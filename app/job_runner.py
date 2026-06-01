from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Callable, Protocol

from app.codex_service import CodexClientService, CodexEditPair, CodexLabel
from app.jobs import CodexJob, JobStore
from app.storage import DatasetStore


CODEX_JOB_TYPES = {
    "generate_images",
    "generate_edit_pairs",
    "label_image",
    "label_batch",
    "import_recent",
    "curate_raw_images",
}


class RunnerCodexService(Protocol):
    async def generate_images_stream(
        self,
        prompt: str,
        count: int,
        reference_paths: list[Path],
        dataset_slug: str,
        on_event: Callable[[str], None],
    ) -> list[Path]: ...

    async def generate_edit_pairs_stream(
        self,
        prompt: str,
        count: int,
        reference_paths: list[Path],
        dataset_slug: str,
        on_event: Callable[[str], None],
    ) -> list[CodexEditPair]: ...

    async def label_image_stream(
        self,
        image_path: Path,
        trigger_token: str,
        on_event: Callable[[str], None],
    ) -> CodexLabel: ...

    async def label_edit_pair_stream(
        self,
        control_path: Path,
        target_path: Path,
        on_event: Callable[[str], None],
    ) -> str: ...

    async def curate_raw_images_stream(
        self,
        instruction: str,
        raw_paths: list[Path],
        reference_paths: list[Path],
        dataset_slug: str,
        on_event: Callable[[str], None],
    ) -> list[Path]: ...

    def recent_generated_images(self, minutes: int = 30) -> list[Path]: ...


class CodexJobRunner:
    def __init__(
        self,
        dataset_store: DatasetStore,
        job_store: JobStore,
        max_parallel_jobs: int | None = None,
        codex_service_factory: Callable[[], RunnerCodexService] | None = None,
    ) -> None:
        self.dataset_store = dataset_store
        self.job_store = job_store
        self.max_parallel_jobs = max_parallel_jobs or int(os.getenv("CODEX_MAX_PARALLEL_JOBS", "2"))
        self.codex_service_factory = codex_service_factory or CodexClientService
        self._semaphore = asyncio.Semaphore(self.max_parallel_jobs)
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        for job in self.job_store.queued_jobs(CODEX_JOB_TYPES):
            self.enqueue_existing(job)

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def enqueue_existing(self, job: CodexJob) -> None:
        task = asyncio.create_task(self._run_with_limit(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def enqueue(self, job: CodexJob) -> None:
        self.enqueue_existing(job)

    async def _run_with_limit(self, job: CodexJob) -> None:
        async with self._semaphore:
            await self._run_job(job)

    async def _run_job(self, job: CodexJob) -> None:
        codex = self.codex_service_factory()
        self.job_store.mark_running(job.dataset_slug, job.id)
        self.job_store.append_log(job.dataset_slug, job.id, f"[job] {job.type} started")
        try:
            if job.type == "generate_images":
                await self._run_generate(job, codex)
            elif job.type == "generate_edit_pairs":
                await self._run_generate_edit_pairs(job, codex)
            elif job.type == "label_image":
                await self._run_label(job, codex)
            elif job.type == "label_batch":
                await self._run_label_batch(job, codex)
            elif job.type == "import_recent":
                await self._run_import_recent(job, codex)
            elif job.type == "curate_raw_images":
                await self._run_curate_raw_images(job, codex)
            else:
                raise ValueError(f"Unsupported job type: {job.type}")
        except Exception as exc:
            self.job_store.append_log(job.dataset_slug, job.id, f"[error] {exc}")
            self.job_store.mark_error(job.dataset_slug, job.id, str(exc))

    async def _run_generate(self, job: CodexJob, codex: RunnerCodexService) -> None:
        prompt = str(job.payload["prompt"])
        count = int(job.payload.get("count", 1))
        references = self.dataset_store.list_references(job.dataset_slug)
        generated = await codex.generate_images_stream(
            prompt,
            count,
            references,
            job.dataset_slug,
            lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
        )
        imported = self.dataset_store.import_generated_images(job.dataset_slug, generated, prompt)
        self.job_store.append_log(job.dataset_slug, job.id, f"[job] imported {len(imported)} images")
        self.job_store.mark_success(
            job.dataset_slug,
            job.id,
            f"Generated {len(imported)} images with Codex",
            generated_count=len(generated),
            imported_count=len(imported),
        )

    async def _run_generate_edit_pairs(self, job: CodexJob, codex: RunnerCodexService) -> None:
        prompt = str(job.payload["prompt"])
        count = int(job.payload.get("count", 1))
        references = self.dataset_store.list_references(job.dataset_slug)
        pairs = await codex.generate_edit_pairs_stream(
            prompt,
            count,
            references,
            job.dataset_slug,
            lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
        )
        if len(pairs) < count:
            self.job_store.append_log(job.dataset_slug, job.id, f"[warning] expected {count} pairs, found {len(pairs)} complete pairs")
        imported = self.dataset_store.import_edit_pairs(
            job.dataset_slug,
            [(pair.control_path, pair.target_path, pair.instruction) for pair in pairs],
            prompt,
        )
        self.job_store.append_log(job.dataset_slug, job.id, f"[job] imported {len(imported)} edit pairs")
        self.job_store.mark_success(
            job.dataset_slug,
            job.id,
            f"Generated {len(imported)} Qwen Edit pairs with Codex",
            generated_count=len(pairs) * 2,
            imported_count=len(imported),
        )

    async def _run_label(self, job: CodexJob, codex: RunnerCodexService) -> None:
        stem = str(job.payload["stem"])
        settings = self.dataset_store.load_settings(job.dataset_slug)
        record = next(record for record in self.dataset_store.list_images(job.dataset_slug) if record.stem == stem)
        image_path = record.image_path
        if settings.dataset_type == "qwen_image_edit_2511":
            control_path = record.control_path
            if control_path is None:
                raise FileNotFoundError(f"control image for {stem}")
            instruction = await codex.label_edit_pair_stream(
                control_path,
                image_path,
                lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
            )
            self.dataset_store.update_caption(job.dataset_slug, stem, instruction, instruction)
        else:
            label = await codex.label_image_stream(
                image_path,
                settings.trigger_token,
                lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
            )
            self.dataset_store.apply_label(job.dataset_slug, stem, label.tags, label.description)
        self.job_store.mark_success(job.dataset_slug, job.id, f"Label applied to {stem}")

    async def _run_label_batch(self, job: CodexJob, codex: RunnerCodexService) -> None:
        stems = [str(stem) for stem in job.payload.get("stems", [])]
        settings = self.dataset_store.load_settings(job.dataset_slug)
        labeled = 0
        for stem in stems:
            record = next(record for record in self.dataset_store.list_images(job.dataset_slug) if record.stem == stem)
            if settings.dataset_type == "qwen_image_edit_2511":
                if record.control_path is None:
                    raise FileNotFoundError(f"control image for {stem}")
                instruction = await codex.label_edit_pair_stream(
                    record.control_path,
                    record.image_path,
                    lambda line, current_stem=stem: self.job_store.append_log(
                        job.dataset_slug, job.id, f"[{current_stem}] {line}"
                    ),
                )
                self.dataset_store.update_caption(job.dataset_slug, stem, instruction, instruction)
            else:
                label = await codex.label_image_stream(
                    record.image_path,
                    settings.trigger_token,
                    lambda line, current_stem=stem: self.job_store.append_log(
                        job.dataset_slug, job.id, f"[{current_stem}] {line}"
                    ),
                )
                self.dataset_store.apply_label(job.dataset_slug, stem, label.tags, label.description)
            labeled += 1
        self.job_store.mark_success(job.dataset_slug, job.id, f"Applied labels to {labeled} images")

    async def _run_import_recent(self, job: CodexJob, codex: RunnerCodexService) -> None:
        minutes = int(job.payload.get("minutes", 60))
        limit = int(job.payload.get("limit", 12))
        recent = codex.recent_generated_images(minutes=minutes)[:limit]
        imported = self.dataset_store.import_generated_images(
            job.dataset_slug,
            recent,
            f"Imported from recent Codex images ({minutes} minutes)",
        )
        self.job_store.append_log(job.dataset_slug, job.id, f"[job] found={len(recent)} imported={len(imported)}")
        self.job_store.mark_success(
            job.dataset_slug,
            job.id,
            f"Imported {len(imported)} recent Codex images",
            generated_count=len(recent),
            imported_count=len(imported),
        )

    async def _run_curate_raw_images(self, job: CodexJob, codex: RunnerCodexService) -> None:
        instruction = str(job.payload.get("instruction", "")).strip()
        raw_names = [str(name) for name in job.payload.get("raw_names", [])]
        reference_names = [str(name) for name in job.payload.get("reference_names", [])]
        raw_paths = [self.dataset_store.raw_image_path(job.dataset_slug, name) for name in raw_names]
        reference_paths = [
            self.dataset_store.reference_image_path(job.dataset_slug, name)
            for name in reference_names
        ]
        if not raw_paths:
            raise ValueError("Select at least one raw image to curate")
        generated = await codex.curate_raw_images_stream(
            instruction,
            raw_paths,
            reference_paths,
            job.dataset_slug,
            lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
        )
        candidates = self.dataset_store.stage_curator_candidates(
            job.dataset_slug,
            job.id,
            generated,
            raw_paths,
            reference_paths,
            instruction,
        )
        self.job_store.append_log(job.dataset_slug, job.id, f"[job] staged {len(candidates)} curator candidates")
        self.job_store.mark_success(
            job.dataset_slug,
            job.id,
            f"Curated {len(candidates)} image candidates",
            generated_count=len(generated),
            imported_count=0,
            output_path=str(self.dataset_store.curator_dir(job.dataset_slug) / job.id),
        )
