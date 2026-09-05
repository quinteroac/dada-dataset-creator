from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import modal


VOLUME_NAME = os.getenv("DADA_MODAL_VOLUME", "dada-minimax-h3")
GPU = os.getenv("DADA_MODAL_GPU", "RTX-PRO-6000")
TIMEOUT = int(os.getenv("DADA_MODAL_TIMEOUT", "86400"))
TOOLKIT_ROOT = Path("/opt/ai-toolkit")
AI_TOOLKIT_REF = "fea99cf5d019a69bee587babcbeb42afd795fddd"
app = modal.App("dada-minimax-h3")
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install("torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .run_commands(
        "git clone https://github.com/ostris/ai-toolkit.git /opt/ai-toolkit",
        f"git -C /opt/ai-toolkit checkout {AI_TOOLKIT_REF}",
        "git -C /opt/ai-toolkit submodule update --init --recursive",
        "pip install -r /opt/ai-toolkit/requirements.txt",
    )
    .env({"HF_HOME": "/data/huggingface", "MODELS_PATH": "/data/models", "PYTHONUNBUFFERED": "1"})
)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, memory=131072, volumes={"/data": data_volume})
def run_minimax_h3_training(config: dict) -> str:
    data_volume.reload()
    process = config["config"]["process"][0]
    dataset_dir = Path(process["datasets"][0]["folder_path"]).parent
    config_path = dataset_dir / "minimax_h3_training_config.yaml"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    output_dir = Path(process["training_folder"])
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Inherit stdout/stderr so Modal streams the trainer's logs.
        subprocess.run(["python", "run.py", str(config_path)], cwd=TOOLKIT_ROOT, check=True)
    finally:
        # Keep completed checkpoints and caches even when the trainer fails.
        data_volume.commit()
    return str(output_dir / config["config"]["name"])


def upload_inputs(config: dict, local_dataset_dir: Path, dataset_slug: str) -> dict:
    config = json.loads(json.dumps(config))
    process = config["config"]["process"][0]
    run_id = uuid.uuid4().hex
    remote_dataset = f"/datasets/{dataset_slug}/{run_id}"
    images = local_dataset_dir / "images"
    if not images.is_dir():
        raise ValueError(f"Dataset images directory does not exist: {images}")
    process["datasets"][0]["folder_path"] = f"/data{remote_dataset}/images"
    with data_volume.batch_upload(force=True) as batch:
        for path in sorted(images.iterdir()):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".txt"}:
                batch.put_file(str(path), f"{remote_dataset}/images/{path.name}")
        model = process["model"]
        for key, directory in [("name_or_path", True), ("assistant_lora_path", False)]:
            value = str(model.get(key) or "").strip()
            if not value:
                continue
            path = Path(value).expanduser()
            if path.exists():
                remote_path = f"/uploads/{run_id}/{key}"
                if directory:
                    if not path.is_dir():
                        raise ValueError("MiniMax H3 local model must be a Comfy-layout directory")
                    batch.put_directory(str(path.resolve()), remote_path)
                else:
                    if not path.is_file():
                        raise ValueError("Training adapter must be a file")
                    remote_path += ".safetensors"
                    batch.put_file(str(path.resolve()), remote_path)
                model[key] = f"/data{remote_path}"
            elif value.startswith(("/data/",)):
                pass  # An existing model/adapter on the mounted volume.
            elif value.startswith(("/", "~", ".")):
                raise ValueError(f"Local model source does not exist: {value}")
    return config


@app.local_entrypoint()
def main(dataset_slug: str, dataset_dir: str, config_json: str) -> None:
    config = upload_inputs(json.loads(config_json), Path(dataset_dir).resolve(), dataset_slug)
    print(f"[modal] training MiniMax H3 images on {GPU}", flush=True)
    output_path = run_minimax_h3_training.remote(config)
    print(f"[modal] output_path={output_path}", flush=True)
