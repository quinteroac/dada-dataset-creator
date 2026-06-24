from pathlib import Path

from fastapi.testclient import TestClient

from app.job_runner import CodexJobRunner
from app.jobs import JobStore
from app.main import app, get_job_store, get_runner, get_store, get_training_runner
from app.storage import DatasetStore


class NoopRunner:
    def __init__(self) -> None:
        self.enqueued = []

    def enqueue(self, job) -> None:
        self.enqueued.append(job)


class NoopTrainingRunner(NoopRunner):
    async def cancel(self, dataset_slug, job_id) -> None:
        self.cancelled = (dataset_slug, job_id)


def test_fastapi_create_dataset_and_upload_image(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        response = client.post(
            "/datasets",
            data={
                "name": "Web Dataset",
                "dataset_type": "anima",
                "trigger_token": "web_token",
                "resolution_width": 1024,
                "resolution_height": 1024,
                "batch_size": 1,
                "num_repeats": 10,
                "min_bucket_reso": 512,
                "max_bucket_reso": 1536,
                "bucket_reso_steps": 16,
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == "/datasets/web_dataset"

        upload = client.post(
            "/datasets/web_dataset/upload",
            files={"files": ("sample.png", b"fake image", "image/png")},
            follow_redirects=False,
        )
        assert upload.status_code == 303
        assert (tmp_path / "datasets" / "web_dataset" / "images" / "000001.png").exists()
        assert (tmp_path / "datasets" / "web_dataset" / "images" / "000001.txt").read_text().strip() == "web_token"
    finally:
        app.dependency_overrides.clear()


def test_fastapi_home_renders(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        response = client.get("/")
        assert response.status_code == 200
        assert "Create dataset" in response.text
    finally:
        app.dependency_overrides.clear()


def test_fastapi_update_caption_and_download_toml(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Web Dataset", "anima", "web_token")
    generated = tmp_path / "generated.png"
    generated.write_bytes(b"fake")
    store.import_generated_images(settings.slug, [generated], "")

    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        response = client.post(
            f"/datasets/{settings.slug}/images/000001/caption",
            data={"caption": "solo, red dress", "description": "A solo character."},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "web_token, solo, red dress" in (
            tmp_path / "datasets" / settings.slug / "images" / "000001.txt"
        ).read_text()

        toml = client.get(f"/datasets/{settings.slug}/dataset.toml")
        assert toml.status_code == 200
        assert 'class_tokens = "web_token"' in toml.text
    finally:
        app.dependency_overrides.clear()


def test_fastapi_delete_image(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Delete Web", "anima", "web_token")
    generated = tmp_path / "generated.png"
    generated.write_bytes(b"fake")
    store.import_generated_images(settings.slug, [generated], "")

    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        response = client.post(
            f"/datasets/{settings.slug}/images/000001/delete",
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert not (tmp_path / "datasets" / settings.slug / "images" / "000001.png").exists()
        assert not (tmp_path / "datasets" / settings.slug / "images" / "000001.txt").exists()
    finally:
        app.dependency_overrides.clear()


def test_fastapi_delete_selected_images(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Bulk Delete Web", "anima", "web_token")
    generated = []
    for index in range(3):
        path = tmp_path / f"generated-{index}.png"
        path.write_bytes(b"fake")
        generated.append(path)
    store.import_generated_images(settings.slug, generated, "")

    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        response = client.post(
            f"/datasets/{settings.slug}/images/delete-selected",
            data={"stems": ["000001", "000003"]},
            follow_redirects=False,
        )
        assert response.status_code == 303
        images_dir = tmp_path / "datasets" / settings.slug / "images"
        assert not (images_dir / "000001.png").exists()
        assert not (images_dir / "000001.txt").exists()
        assert not (images_dir / "000001.meta.json").exists()
        assert (images_dir / "000002.png").exists()
        assert not (images_dir / "000003.png").exists()
    finally:
        app.dependency_overrides.clear()


def test_fastapi_generate_creates_queued_job(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Async Web", "anima", "web_token")
    job_store = JobStore(tmp_path / "datasets")
    runner = NoopRunner()

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    app.dependency_overrides[get_runner] = lambda: runner
    client = TestClient(app)
    try:
        response = client.post(
            f"/datasets/{settings.slug}/generate",
            data={"prompt": "blue dawn", "count": 1},
            follow_redirects=False,
        )
        assert response.status_code == 303
        jobs = job_store.list_jobs(settings.slug)
        assert jobs[0].status == "queued"
        assert jobs[0].type == "generate_images"
        assert runner.enqueued[0].id == jobs[0].id
    finally:
        app.dependency_overrides.clear()


def test_fastapi_jobs_endpoint_returns_status_and_output(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Jobs Web", "anima", "web_token")
    job_store = JobStore(tmp_path / "datasets")
    job = job_store.create_job(settings.slug, "generate_images", {"prompt": "blue"})
    job_store.append_log(settings.slug, job.id, "hello from codex")

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    client = TestClient(app)
    try:
        response = client.get(f"/datasets/{settings.slug}/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["jobs"][0]["output_tail"] == ["hello from codex"]

        detail = client.get(f"/datasets/{settings.slug}/jobs/{job.id}")
        assert detail.status_code == 200
        assert "hello from codex" in detail.json()["log"]
    finally:
        app.dependency_overrides.clear()


def test_fastapi_training_creates_queued_job(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Train Web", "anima", "web_token")
    job_store = JobStore(tmp_path / "datasets")
    runner = NoopTrainingRunner()

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    app.dependency_overrides[get_training_runner] = lambda: runner
    client = TestClient(app)
    try:
        response = client.post(
            f"/datasets/{settings.slug}/train",
            data={
                "pretrained_model_name_or_path": "/models/anima.safetensors",
                "qwen3": "/models/qwen3",
                "vae": "/models/vae.safetensors",
                "network_dim": 8,
                "max_train_epochs": 1,
                "cache_text_encoder_outputs": "true",
                "qwen_image_vae_2d": "true",
                "cuda_allow_tf32": "true",
                "cuda_cudnn_benchmark": "true",
                "save_state": "true",
                "save_state_on_train_end": "true",
                "resume_state_path": "datasets/train_web/outputs/train_web-state",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        job = job_store.list_jobs(settings.slug)[0]
        assert job.type == "train_anima_lora"
        assert job.status == "queued"
        assert job.payload["cache_text_encoder_outputs"] is True
        assert job.payload["qwen_image_vae_2d"] is True
        assert job.payload["cuda_allow_tf32"] is True
        assert job.payload["cuda_cudnn_benchmark"] is True
        assert job.payload["save_state"] is True
        assert job.payload["save_state_on_train_end"] is True
        assert job.payload["resume_state_path"] == "datasets/train_web/outputs/train_web-state"
        assert runner.enqueued[0].id == job.id
    finally:
        app.dependency_overrides.clear()


def test_fastapi_setup_sd_scripts_creates_job(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Setup Web", "anima", "web_token")
    job_store = JobStore(tmp_path / "datasets")
    runner = NoopTrainingRunner()

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    app.dependency_overrides[get_training_runner] = lambda: runner
    client = TestClient(app)
    try:
        response = client.post(f"/datasets/{settings.slug}/setup-sd-scripts", follow_redirects=False)
        assert response.status_code == 303
        assert job_store.list_jobs(settings.slug)[0].type == "setup_sd_scripts"
    finally:
        app.dependency_overrides.clear()


def test_fastapi_qwen_routes_create_jobs(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Qwen Web", "qwen_image_edit_2511", "Edit the image")
    job_store = JobStore(tmp_path / "datasets")
    runner = NoopRunner()
    training_runner = NoopTrainingRunner()

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    app.dependency_overrides[get_runner] = lambda: runner
    app.dependency_overrides[get_training_runner] = lambda: training_runner
    client = TestClient(app)
    try:
        generate = client.post(
            f"/datasets/{settings.slug}/generate-edit-pairs",
            data={"prompt": "shirt edits", "count": 1},
            follow_redirects=False,
        )
        setup = client.post(f"/datasets/{settings.slug}/setup-musubi-tuner", follow_redirects=False)
        train = client.post(
            f"/datasets/{settings.slug}/train-qwen-edit",
            data={
                "dit": "/models/edit2511.safetensors",
                "vae": "/models/vae.safetensors",
                "text_encoder": "/models/qwen_vl.safetensors",
                "network_dim": 16,
            },
            follow_redirects=False,
        )

        assert generate.status_code == 303
        assert setup.status_code == 303
        assert train.status_code == 303
        types = [job.type for job in job_store.list_jobs(settings.slug)]
        assert "generate_edit_pairs" in types
        assert "setup_musubi_tuner" in types
        assert "train_qwen_edit_lora" in types
    finally:
        app.dependency_overrides.clear()


def test_fastapi_ideogram4_routes_create_jobs(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    job_store = JobStore(tmp_path / "datasets")
    runner = NoopRunner()
    training_runner = NoopTrainingRunner()

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    app.dependency_overrides[get_runner] = lambda: runner
    app.dependency_overrides[get_training_runner] = lambda: training_runner
    client = TestClient(app)
    try:
        create = client.post(
            "/datasets",
            data={
                "name": "Ideogram Web",
                "dataset_type": "ideogram4",
                "trigger_token": "fallback caption",
                "resolution_width": 1024,
                "resolution_height": 1024,
                "batch_size": 1,
                "num_repeats": 10,
                "min_bucket_reso": 512,
                "max_bucket_reso": 1536,
                "bucket_reso_steps": 16,
            },
            follow_redirects=False,
        )
        assert create.status_code == 303
        assert create.headers["location"] == "/datasets/ideogram_web"

        upload = client.post(
            "/datasets/ideogram_web/upload",
            files={"files": ("sample.png", b"fake image", "image/png")},
            follow_redirects=False,
        )
        label = client.post("/datasets/ideogram_web/images/000001/label", follow_redirects=False)
        setup = client.post("/datasets/ideogram_web/setup-ai-toolkit", follow_redirects=False)
        train = client.post(
            "/datasets/ideogram_web/train-ideogram4",
            data={"model_path": "/models/ideogram-4-fp8"},
            follow_redirects=False,
        )

        assert upload.status_code == 303
        assert label.status_code == 303
        assert setup.status_code == 303
        assert train.status_code == 303
        types = [job.type for job in job_store.list_jobs("ideogram_web")]
        assert "label_image" in types
        assert "setup_ai_toolkit" in types
        assert "train_ideogram4_lora" in types
        assert runner.enqueued[0].type == "label_image"
        assert training_runner.enqueued[-1].type == "train_ideogram4_lora"
    finally:
        app.dependency_overrides.clear()


def test_fastapi_qwen_generate_edit_pairs_uses_fallback_prompt(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Qwen Web", "qwen_image_edit_2511", "Edit portraits into watercolor")
    job_store = JobStore(tmp_path / "datasets")
    runner = NoopRunner()

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    app.dependency_overrides[get_runner] = lambda: runner
    client = TestClient(app)
    try:
        response = client.post(f"/datasets/{settings.slug}/generate-edit-pairs", follow_redirects=False)

        assert response.status_code == 303
        job = job_store.list_jobs(settings.slug)[0]
        assert job.type == "generate_edit_pairs"
        assert job.payload["prompt"] == "Edit_portraits_into_watercolor"
    finally:
        app.dependency_overrides.clear()


def test_fastapi_curator_routes_upload_and_create_job(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Curator Web", "anima", "web_token")
    job_store = JobStore(tmp_path / "datasets")
    runner = NoopRunner()

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_job_store] = lambda: job_store
    app.dependency_overrides[get_runner] = lambda: runner
    client = TestClient(app)
    try:
        page = client.get(f"/datasets/{settings.slug}/curator")
        assert page.status_code == 200
        assert "Curate raw images" in page.text

        raw_upload = client.post(
            f"/datasets/{settings.slug}/curator/raw",
            files={"files": ("raw.png", b"raw image", "image/png")},
            follow_redirects=False,
        )
        ref_upload = client.post(
            f"/datasets/{settings.slug}/curator/references",
            files={"files": ("ref.png", b"ref image", "image/png")},
            follow_redirects=False,
        )
        assert raw_upload.status_code == 303
        assert ref_upload.status_code == 303
        assert (tmp_path / "datasets" / settings.slug / "raw" / "raw_000001.png").exists()
        assert (tmp_path / "datasets" / settings.slug / "references" / "ref_001.png").exists()

        process = client.post(
            f"/datasets/{settings.slug}/curator/process",
            data={
                "instruction": "Use the reference colors.",
                "selection_mode": "selected",
                "reference_mode": "selected",
                "raw_names": "raw_000001.png",
                "reference_names": "ref_001.png",
            },
            follow_redirects=False,
        )
        assert process.status_code == 303
        job = job_store.list_jobs(settings.slug)[0]
        assert job.type == "curate_raw_images"
        assert job.payload["raw_names"] == ["raw_000001.png"]
        assert job.payload["reference_names"] == ["ref_001.png"]
        assert runner.enqueued[0].id == job.id
    finally:
        app.dependency_overrides.clear()


def test_fastapi_curator_approve_candidate(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Curator Approve", "anima", "web_token")
    raw = store.raw_dir(settings.slug) / "raw_000001.png"
    generated = tmp_path / "curated.png"
    raw.write_bytes(b"raw")
    generated.write_bytes(b"curated")
    candidate = store.stage_curator_candidates(settings.slug, "job1", [generated], [raw], [], "Make it cinematic.")[0]

    app.dependency_overrides[get_store] = lambda: store
    client = TestClient(app)
    try:
        response = client.post(
            f"/datasets/{settings.slug}/curator/candidates/{candidate.id}/approve",
            data={"instruction": "cinematic portrait"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert (tmp_path / "datasets" / settings.slug / "images" / "000001.png").exists()
        assert "web_token, cinematic portrait" in (
            tmp_path / "datasets" / settings.slug / "images" / "000001.txt"
        ).read_text()
    finally:
        app.dependency_overrides.clear()
