import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.jobs import JobStore
from app.main import app, get_job_store, get_store, get_training_runner
from app.storage import DatasetStore
from app.training_runner import TrainingJobRunner
from app.training_service import ProcessResult, TrainingService


def test_image_config_and_old_toolkit_detection(tmp_path: Path):
    service = TrainingService(tmp_path / "vendor", tmp_path / "datasets")
    config = json.loads(service.build_minimax_h3_config("photos", {
        "output_name": 'portrait "test"', "resolution_list": "512,768",
        "assistant_lora_path": '/models/adapter "test".safetensors',
    }))
    process = config["config"]["process"][0]
    dataset = process["datasets"][0]
    assert dataset["num_frames"] == 1
    assert dataset["do_audio"] is False
    assert dataset["resolution"] == [512, 768]
    assert dataset["caption_ext"] == "txt"
    assert dataset["cache_text_embeddings"] is True
    assert process["model"]["arch"] == "minimax_h3"
    assert process["model"]["layer_offloading_transformer_percent"] == 0.95
    assert process["model"]["assistant_lora_path"] == '/models/adapter "test".safetensors'
    service.ai_toolkit_dir.mkdir(parents=True)
    (service.ai_toolkit_dir / "run.py").touch()
    with pytest.raises(RuntimeError, match="lacks MiniMax H3"):
        service.build_minimax_h3_train_command("photos", {})


@pytest.mark.parametrize("payload", [
    {"resolution_list": "513"}, {"resolution_list": ""}, {"resolution_list": "0"},
    {"output_name": "../escape"}, {"learning_rate": "nan"}, {"steps": 0},
])
def test_reject_invalid_config(payload):
    with pytest.raises(ValueError):
        TrainingService().build_minimax_h3_config("photos", payload)


@pytest.mark.parametrize("return_code", [0, 1])
@pytest.mark.parametrize("backend", ["local", "modal"])
def test_image_training_flow(tmp_path: Path, return_code: int, backend: str):
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Photos", "minimax_h3", "person")
    jobs = JobStore(store.root)
    service = TrainingService(tmp_path / "vendor", store.root)
    service.ai_toolkit_dir.mkdir(parents=True)
    (service.ai_toolkit_dir / "run.py").touch()
    model_dir = service.ai_toolkit_dir / "extensions_built_in/diffusion_models/minimax_h3"
    model_dir.mkdir(parents=True)
    (model_dir / "minimax_h3.py").touch()

    async def fake_process(command, cwd, on_line, on_process_start=None):
        if backend == "modal":
            assert cwd == Path(".")
            if command[:3] == ["modal", "volume", "get"]:
                assert return_code == 0
                Path(command[-1], "photos_minimax_h3_lora.safetensors").write_bytes(b"weights")
                return ProcessResult(0)
            assert "DADA_MODAL_GPU=RTX-PRO-6000" in command
            assert "app/modal_minimax_h3.py" in command
            config = json.loads(command[-1])
            assert config["config"]["process"][0]["training_folder"] == "/data/outputs/photos"
        else:
            assert cwd == service.ai_toolkit_dir
            config = json.loads(Path(command[2]).read_text())
        assert config["config"]["process"][0]["datasets"][0]["num_frames"] == 1
        on_line("image training")
        return ProcessResult(return_code)

    service.run_process = fake_process
    runner = TrainingJobRunner(store, jobs, training_service_factory=lambda: service)
    queued = []
    runner.enqueue = queued.append
    app.dependency_overrides.update({get_store: lambda: store, get_job_store: lambda: jobs,
                                     get_training_runner: lambda: runner})
    try:
        client = TestClient(app)
        url = f"/datasets/{settings.slug}"
        assert "MiniMax H3 Training" in client.get(url).text
        assert client.post(url + "/train-minimax-h3").status_code == 400
        client.post(url + "/upload", files={"files": ("portrait.png", b"image", "image/png")}, follow_redirects=False)
        assert client.post(url + "/train-minimax-h3", data={"resolution_list": "513"}).status_code == 400
        response = client.post(url + "/train-minimax-h3", data={"training_backend": backend}, follow_redirects=False)
        assert response.status_code == 303
        job = queued.pop()
        assert job.type == "train_minimax_h3_lora"
        asyncio.run(runner._run_job(job))
        completed = jobs.get_job(settings.slug, job.id)
        assert completed.status == ("success" if return_code == 0 else "error")
        assert completed.return_code == return_code
        if backend == "modal" and return_code == 0:
            assert Path(completed.output_path).read_bytes() == b"weights"
        assert client.post(url + f"/jobs/{job.id}/cancel", follow_redirects=False).status_code == 303
        assert client.post(url + "/setup-ai-toolkit", follow_redirects=False).status_code == 303
        assert queued[-1].type == "setup_ai_toolkit"
    finally:
        app.dependency_overrides.clear()


def test_modal_config_remaps_paths_without_local_toolkit(tmp_path: Path):
    service = TrainingService(tmp_path / "vendor", tmp_path / "datasets")
    command = service.build_minimax_h3_modal_command("photos", {
        "output_dir": str(tmp_path / "local_outputs"), "modal_gpu": "H100",
        "modal_volume_name": "custom-volume", "modal_output_dir": "/data/outputs/custom",
    })
    assert "DADA_MODAL_GPU=H100" in command
    assert "DADA_MODAL_VOLUME=custom-volume" in command
    process = json.loads(command[-1])["config"]["process"][0]
    assert process["training_folder"] == "/data/outputs/custom"
    assert process["datasets"][0]["folder_path"] == "/data/datasets/photos/images"
    for invalid in ["/tmp/output", "/data/outputs/../../models"]:
        with pytest.raises(ValueError, match="inside /data/outputs"):
            service.build_minimax_h3_modal_command("photos", {"modal_output_dir": invalid})


def test_modal_upload_isolates_images(tmp_path: Path, monkeypatch):
    pytest.importorskip("modal")
    from app import modal_minimax_h3 as remote
    from unittest.mock import MagicMock

    volume = MagicMock()
    monkeypatch.setattr(remote, "data_volume", volume)
    images = tmp_path / "images"
    images.mkdir()
    for name in ["portrait.png", "portrait.txt", "metadata.json", "old.mp4"]:
        (images / name).touch()
    model = tmp_path / "models"
    model.mkdir()
    adapter = tmp_path / "adapter.safetensors"
    adapter.touch()
    config = json.loads(TrainingService().build_minimax_h3_config("photos", {
        "model_path": str(model), "assistant_lora_path": str(adapter),
    }))
    first = remote.upload_inputs(config, tmp_path, "photos")
    second = remote.upload_inputs(config, tmp_path, "photos")
    process = first["config"]["process"][0]
    assert process["datasets"][0]["folder_path"] != second["config"]["process"][0]["datasets"][0]["folder_path"]
    assert process["model"]["name_or_path"].startswith("/data/uploads/")
    assert process["model"]["assistant_lora_path"].endswith(".safetensors")
    calls = volume.batch_upload.return_value.__enter__.return_value.put_file.call_args_list
    assert len(calls) == 6
    assert not any(call.args[0].endswith((".json", ".mp4")) for call in calls)
    assert config["config"]["process"][0]["model"]["name_or_path"] == str(model)


def test_modal_commits_on_training_failure(tmp_path: Path, monkeypatch):
    pytest.importorskip("modal")
    from app import modal_minimax_h3 as remote
    from unittest.mock import MagicMock
    import subprocess

    volume = MagicMock()
    monkeypatch.setattr(remote, "data_volume", volume)
    monkeypatch.setattr(remote.subprocess, "run", MagicMock(side_effect=subprocess.CalledProcessError(1, "trainer")))
    (tmp_path / "photos").mkdir()
    config = json.loads(TrainingService(datasets_root=tmp_path).build_minimax_h3_config("photos", {}))
    with pytest.raises(subprocess.CalledProcessError):
        remote.run_minimax_h3_training.local(config)
    volume.reload.assert_called_once()
    volume.commit.assert_called_once()
