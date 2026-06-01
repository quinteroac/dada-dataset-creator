from __future__ import annotations

import asyncio
import os
import signal
from typing import Callable

from app.jobs import CodexJob, JobStore
from app.storage import DatasetStore
from app.training_service import TrainingService


TRAINING_JOB_TYPES = {"setup_sd_scripts", "setup_musubi_tuner", "train_anima_lora", "train_qwen_edit_lora"}


class TrainingJobRunner:
    def __init__(
        self,
        dataset_store: DatasetStore,
        job_store: JobStore,
        max_parallel_jobs: int | None = None,
        training_service_factory: Callable[[], TrainingService] | None = None,
    ) -> None:
        self.dataset_store = dataset_store
        self.job_store = job_store
        self.max_parallel_jobs = max_parallel_jobs or int(os.getenv("TRAINING_MAX_PARALLEL_JOBS", "1"))
        self.training_service_factory = training_service_factory or TrainingService
        self._semaphore = asyncio.Semaphore(self.max_parallel_jobs)
        self._tasks: set[asyncio.Task[None]] = set()
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def start(self) -> None:
        for job in self.job_store.queued_jobs(TRAINING_JOB_TYPES):
            self.enqueue_existing(job)

    async def stop(self) -> None:
        for process in list(self._processes.values()):
            self._terminate_process_group(process)
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

    async def cancel(self, dataset_slug: str, job_id: str) -> None:
        process = self._processes.get(job_id)
        if process is not None and process.returncode is None:
            self._terminate_process_group(process)
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._kill_process_group(process)
                await process.wait()
        self.job_store.append_log(dataset_slug, job_id, "[job] cancelled by user")
        self.job_store.mark_error(dataset_slug, job_id, "cancelled by user")

    async def _run_with_limit(self, job: CodexJob) -> None:
        async with self._semaphore:
            await self._run_job(job)

    async def _run_job(self, job: CodexJob) -> None:
        service = self.training_service_factory()
        self.job_store.mark_running(job.dataset_slug, job.id)
        self.job_store.append_log(job.dataset_slug, job.id, f"[job] {job.type} started")
        try:
            if job.type == "setup_sd_scripts":
                await self._run_setup(job, service)
            elif job.type == "setup_musubi_tuner":
                await self._run_setup_musubi(job, service)
            elif job.type == "train_anima_lora":
                await self._run_training(job, service)
            elif job.type == "train_qwen_edit_lora":
                await self._run_qwen_training(job, service)
            else:
                raise ValueError(f"Unsupported training job type: {job.type}")
        except Exception as exc:
            self.job_store.append_log(job.dataset_slug, job.id, f"[error] {exc}")
            self.job_store.mark_error(job.dataset_slug, job.id, str(exc))
        finally:
            self._processes.pop(job.id, None)

    async def _run_setup(self, job: CodexJob, service: TrainingService) -> None:
        command = service.setup_command()
        self.job_store.update_job(job.dataset_slug, job.id, command=command)
        result = await service.setup_sd_scripts(
            lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
            lambda process: self._set_process(job, process),
        )
        if result.return_code != 0:
            self.job_store.mark_error(job.dataset_slug, job.id, f"setup failed with exit code {result.return_code}")
            return
        self.job_store.mark_success(
            job.dataset_slug,
            job.id,
            "sd-scripts ready",
            output_path=result.output_path,
            return_code=result.return_code,
        )

    async def _run_setup_musubi(self, job: CodexJob, service: TrainingService) -> None:
        command = service.setup_musubi_command()
        self.job_store.update_job(job.dataset_slug, job.id, command=command)
        result = await service.setup_musubi_tuner(
            lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
            lambda process: self._set_process(job, process),
        )
        if result.return_code != 0:
            self.job_store.mark_error(job.dataset_slug, job.id, f"setup failed with exit code {result.return_code}")
            return
        self.job_store.mark_success(
            job.dataset_slug,
            job.id,
            "musubi-tuner ready",
            output_path=result.output_path,
            return_code=result.return_code,
        )
    async def _run_training(self, job: CodexJob, service: TrainingService) -> None:
        self.dataset_store.write_dataset_toml(self.dataset_store.load_settings(job.dataset_slug))
        command = service.build_train_command(job.dataset_slug, job.payload)
        self.job_store.update_job(job.dataset_slug, job.id, command=command)
        result = await service.train_anima_lora(
            job.dataset_slug,
            job.payload,
            lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
            lambda process: self._set_process(job, process),
        )
        if result.return_code != 0:
            self.job_store.mark_error(
                job.dataset_slug,
                job.id,
                f"training failed with exit code {result.return_code}",
            )
            self.job_store.update_job(job.dataset_slug, job.id, return_code=result.return_code)
            return
        self.job_store.mark_success(
            job.dataset_slug,
            job.id,
            "Anima LoRA training finished",
            output_path=result.output_path,
            return_code=result.return_code,
        )

    async def _run_qwen_training(self, job: CodexJob, service: TrainingService) -> None:
        self.dataset_store.write_dataset_toml(self.dataset_store.load_settings(job.dataset_slug))
        training_backend = str(job.payload.get("training_backend", "local"))
        command = (
            service.build_qwen_modal_command(job.dataset_slug, job.payload)
            if training_backend == "modal"
            else service.build_qwen_train_command(job.dataset_slug, job.payload)
        )
        self.job_store.update_job(job.dataset_slug, job.id, command=command)
        train = (
            service.train_qwen_edit_lora_on_modal
            if training_backend == "modal"
            else service.train_qwen_edit_lora
        )
        result = await train(
            job.dataset_slug,
            job.payload,
            lambda line: self.job_store.append_log(job.dataset_slug, job.id, line),
            lambda process: self._set_process(job, process),
        )
        if result.return_code != 0:
            self.job_store.mark_error(
                job.dataset_slug,
                job.id,
                f"Qwen Edit training failed with exit code {result.return_code}",
            )
            self.job_store.update_job(job.dataset_slug, job.id, return_code=result.return_code)
            return
        self.job_store.mark_success(
            job.dataset_slug,
            job.id,
            "Qwen Edit-2511 LoRA training finished",
            output_path=result.output_path,
            return_code=result.return_code,
        )

    def _set_process(self, job: CodexJob, process: asyncio.subprocess.Process) -> None:
        self._processes[job.id] = process
        self.job_store.update_job(job.dataset_slug, job.id, process_pid=process.pid)

    @staticmethod
    def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    @staticmethod
    def _kill_process_group(process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
