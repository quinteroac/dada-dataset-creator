import asyncio
from pathlib import Path

from app.jobs import JobStore
from app.storage import DatasetStore
from app.training_runner import TrainingJobRunner
from app.training_service import ProcessResult


class FakeTrainingService:
    def build_train_command(self, dataset_slug, payload):
        return ["accelerate", "launch", "anima_train_network.py", f"--dataset={dataset_slug}"]

    def setup_command(self):
        return ["git", "clone", "repo", "vendor/sd-scripts"]

    def setup_musubi_command(self):
        return ["git", "clone", "repo", "vendor/musubi-tuner"]

    def setup_ai_toolkit_command(self):
        return ["git", "clone", "repo", "vendor/ai-toolkit"]

    def build_qwen_train_command(self, dataset_slug, payload):
        return ["accelerate", "launch", "qwen_image_train_network.py", f"--dataset={dataset_slug}"]

    def build_qwen_modal_command(self, dataset_slug, payload):
        return ["modal", "run", "app/modal_qwen_edit.py", f"--dataset={dataset_slug}"]

    def build_ideogram4_train_command(self, dataset_slug, payload):
        return ["python", "run.py", f"--dataset={dataset_slug}"]

    async def setup_sd_scripts(self, on_line, on_process_start=None):
        on_line("setup output")
        return ProcessResult(return_code=0, output_path="vendor/sd-scripts")

    async def setup_musubi_tuner(self, on_line, on_process_start=None):
        on_line("musubi setup output")
        return ProcessResult(return_code=0, output_path="vendor/musubi-tuner")

    async def setup_ai_toolkit(self, on_line, on_process_start=None):
        on_line("ai toolkit setup output")
        return ProcessResult(return_code=0, output_path="vendor/ai-toolkit")

    async def train_anima_lora(self, dataset_slug, payload, on_line, on_process_start=None):
        on_line("training output")
        output = Path(payload["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        model = output / "model.safetensors"
        model.write_bytes(b"fake")
        return ProcessResult(return_code=0, output_path=str(model))

    async def train_qwen_edit_lora(self, dataset_slug, payload, on_line, on_process_start=None):
        on_line("qwen training output")
        output = Path(payload["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        model = output / "qwen.safetensors"
        model.write_bytes(b"fake")
        return ProcessResult(return_code=0, output_path=str(model))

    async def train_qwen_edit_lora_on_modal(self, dataset_slug, payload, on_line, on_process_start=None):
        on_line("modal qwen training output")
        return ProcessResult(return_code=0, output_path="modal://qwen-volume/outputs/qwen")

    async def train_ideogram4_lora(self, dataset_slug, payload, on_line, on_process_start=None):
        on_line("ideogram4 training output")
        output = Path(payload["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        model = output / "ideogram4.safetensors"
        model.write_bytes(b"fake")
        return ProcessResult(return_code=0, output_path=str(model))


def test_training_runner_executes_training_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Blue", "anima", "blue")
    output_dir = tmp_path / "datasets" / settings.slug / "outputs"
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(
        settings.slug,
        "train_anima_lora",
        {
            "pretrained_model_name_or_path": "model",
            "qwen3": "qwen",
            "vae": "vae",
            "output_dir": str(output_dir),
        },
    )
    runner = TrainingJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        training_service_factory=FakeTrainingService,
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.return_code == 0
    assert loaded.output_path.endswith("model.safetensors")
    assert "training output" in job_store.read_log(settings.slug, job.id)


def test_training_runner_executes_setup_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Blue", "anima", "blue")
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(settings.slug, "setup_sd_scripts", {})
    runner = TrainingJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        training_service_factory=FakeTrainingService,
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.output_path == "vendor/sd-scripts"
    assert "setup output" in job_store.read_log(settings.slug, job.id)


def test_training_runner_executes_qwen_training_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Qwen", "qwen_image_edit_2511", "Edit the image")
    output_dir = tmp_path / "datasets" / settings.slug / "outputs"
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(
        settings.slug,
        "train_qwen_edit_lora",
        {
            "dit": "dit",
            "vae": "vae",
            "text_encoder": "te",
            "output_dir": str(output_dir),
        },
    )
    runner = TrainingJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        training_service_factory=FakeTrainingService,
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.output_path.endswith("qwen.safetensors")
    assert "qwen training output" in job_store.read_log(settings.slug, job.id)


def test_training_runner_executes_modal_qwen_training_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Qwen", "qwen_image_edit_2511", "Edit the image")
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(
        settings.slug,
        "train_qwen_edit_lora",
        {
            "dit": "/data/models/dit.safetensors",
            "vae": "/data/models/vae.safetensors",
            "text_encoder": "/data/models/te.safetensors",
            "training_backend": "modal",
        },
    )
    runner = TrainingJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        training_service_factory=FakeTrainingService,
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.command[0] == "modal"
    assert loaded.output_path == "modal://qwen-volume/outputs/qwen"
    assert "modal qwen training output" in job_store.read_log(settings.slug, job.id)


def test_training_runner_executes_ai_toolkit_setup_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Ideo", "ideogram4", "fallback")
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(settings.slug, "setup_ai_toolkit", {})
    runner = TrainingJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        training_service_factory=FakeTrainingService,
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.output_path == "vendor/ai-toolkit"
    assert "ai toolkit setup output" in job_store.read_log(settings.slug, job.id)


def test_training_runner_executes_ideogram4_training_job(tmp_path: Path) -> None:
    dataset_store = DatasetStore(tmp_path / "datasets")
    settings = dataset_store.create_dataset("Ideo", "ideogram4", "fallback")
    output_dir = tmp_path / "datasets" / settings.slug / "outputs"
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(
        settings.slug,
        "train_ideogram4_lora",
        {
            "model_path": "/models/ideogram-4-fp8",
            "output_dir": str(output_dir),
        },
    )
    runner = TrainingJobRunner(
        dataset_store,
        job_store,
        max_parallel_jobs=1,
        training_service_factory=FakeTrainingService,
    )

    asyncio.run(runner._run_job(job))

    loaded = job_store.get_job(settings.slug, job.id)
    assert loaded.status == "success"
    assert loaded.output_path.endswith("ideogram4.safetensors")
    assert "ideogram4 training output" in job_store.read_log(settings.slug, job.id)
