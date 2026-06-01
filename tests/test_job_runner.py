import asyncio
from pathlib import Path

from app.codex_service import CodexEditPair, CodexLabel
from app.job_runner import CodexJobRunner
from app.jobs import JobStore
from app.storage import DatasetStore


class FakeCodexService:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    async def generate_images_stream(self, prompt, count, reference_paths, dataset_slug, on_event):
        on_event("fake output")
        generated = self.tmp_path / "generated.png"
        generated.write_bytes(b"fake")
        return [generated]

    async def generate_edit_pairs_stream(self, prompt, count, reference_paths, dataset_slug, on_event):
        on_event("fake edit pair output")
        control = self.tmp_path / "control.png"
        target = self.tmp_path / "target.png"
        control.write_bytes(b"control")
        target.write_bytes(b"target")
        return [CodexEditPair(control, target, "Change the shirt to red.")]

    async def label_image_stream(self, image_path, trigger_token, on_event):
        on_event("fake label output")
        return CodexLabel(tags=[trigger_token, "1girl", "blue hair"], description="A blue-haired character.")

    async def label_edit_pair_stream(self, control_path, target_path, on_event):
        on_event("fake edit label output")
        return "Make the background a beach."

    async def curate_raw_images_stream(self, instruction, raw_paths, reference_paths, dataset_slug, on_event):
        on_event("fake curator output")
        generated = self.tmp_path / "curated.png"
        generated.write_bytes(b"curated")
        return [generated]

    def recent_generated_images(self, minutes=30):
        generated = self.tmp_path / "recent.png"
        generated.write_bytes(b"fake")
        return [generated]


def test_runner_executes_generation_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Blue", "anima", "blue_token")
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(settings.slug, "generate_images", {"prompt": "blue", "count": 1})
    runner = CodexJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        codex_service_factory=lambda: FakeCodexService(tmp_path),
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.imported_count == 1
    assert "fake output" in job_store.read_log(settings.slug, job.id)
    assert (tmp_path / "datasets" / settings.slug / "images" / "000001.png").exists()


def test_runner_executes_label_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Blue", "anima", "blue_token")
    source = tmp_path / "source.png"
    source.write_bytes(b"fake")
    dataset_store.import_generated_images(settings.slug, [source], "")
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(settings.slug, "label_image", {"stem": "000001"})
    runner = CodexJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        codex_service_factory=lambda: FakeCodexService(tmp_path),
    )

    asyncio.run(runner._run_job(job))

    assert job_store.get_job(settings.slug, job.id).status == "success"
    assert "blue_token, 1girl, blue hair" in (
        tmp_path / "datasets" / settings.slug / "images" / "000001.txt"
    ).read_text()


def test_runner_executes_edit_pair_generation_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Qwen", "qwen_image_edit_2511", "Edit the image")
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(settings.slug, "generate_edit_pairs", {"prompt": "red shirt", "count": 1})
    runner = CodexJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        codex_service_factory=lambda: FakeCodexService(tmp_path),
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.imported_count == 1
    assert (tmp_path / "datasets" / settings.slug / "controls" / "000001.png").exists()
    assert (tmp_path / "datasets" / settings.slug / "images" / "000001.txt").read_text().strip() == "Change the shirt to red."


def test_runner_executes_curator_job_without_importing(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Curator", "anima", "cur_token")
    raw = dataset_store.raw_dir(settings.slug) / "raw_000001.png"
    reference = dataset_store.references_dir(settings.slug) / "ref_001.png"
    raw.write_bytes(b"raw")
    reference.write_bytes(b"ref")
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(
        settings.slug,
        "curate_raw_images",
        {
            "instruction": "Use the reference colors.",
            "raw_names": ["raw_000001.png"],
            "reference_names": ["ref_001.png"],
        },
    )
    runner = CodexJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        codex_service_factory=lambda: FakeCodexService(tmp_path),
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.generated_count == 1
    assert loaded.imported_count == 0
    assert "fake curator output" in job_store.read_log(settings.slug, job.id)
    assert (tmp_path / "datasets" / settings.slug / "curator" / job.id / "candidate_000001.png").exists()
    assert not (tmp_path / "datasets" / settings.slug / "images" / "000001.png").exists()
