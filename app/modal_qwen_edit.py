from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
import tempfile
from pathlib import Path

import modal


APP_NAME = "dada-qwen-edit-2511"
DATA_ROOT = Path("/data")
MODEL_ROOT = Path("/models")
MUSUBI_ROOT = Path("/opt/musubi-tuner")
VOLUME_NAME = os.getenv("DADA_MODAL_VOLUME", "dada-qwen-edit")
GPU = os.getenv("DADA_MODAL_GPU", "L40S")
TIMEOUT = int(os.getenv("DADA_MODAL_TIMEOUT", "86400"))
LOCAL_DIT = os.getenv("DADA_MODAL_LOCAL_DIT", "").strip()
LOCAL_VAE = os.getenv("DADA_MODAL_LOCAL_VAE", "").strip()
LOCAL_TEXT_ENCODER = os.getenv("DADA_MODAL_LOCAL_TEXT_ENCODER", "").strip()

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "accelerate",
        "bitsandbytes",
        "opencv-python-headless",
        "pillow",
        "protobuf",
        "safetensors",
        "sentencepiece",
        "tomli-w",
        "torch",
        "torchvision",
        "transformers",
        "uv",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner",
        "cd /opt/musubi-tuner && uv pip install --system -e .",
    )
)


def add_baked_model(base_image: modal.Image, local_path: str, remote_name: str) -> modal.Image:
    if not local_path:
        return base_image
    path = Path(local_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Modal baked model source does not exist: {path}")
    size_gb = path.stat().st_size / (1024**3)
    print(f"[modal] image layer includes {path} -> {MODEL_ROOT / remote_name} ({size_gb:.2f} GiB)", flush=True)
    return base_image.add_local_file(path, str(MODEL_ROOT / remote_name), copy=True)


image = add_baked_model(image, LOCAL_DIT, "dit.safetensors")
image = add_baked_model(image, LOCAL_VAE, "vae.safetensors")
image = add_baked_model(image, LOCAL_TEXT_ENCODER, "text_encoder.safetensors")


def render_remote_toml(dataset_slug: str, payload: dict[str, object]) -> str:
    resolution_width = int(payload.get("resolution_width", 1024))
    resolution_height = int(payload.get("resolution_height", 1024))
    batch_size = int(payload.get("batch_size", 1))
    num_repeats = int(payload.get("num_repeats", 1))
    dataset_dir = DATA_ROOT / "datasets" / dataset_slug
    return "\n".join(
        [
            "[general]",
            f"resolution = [{resolution_width}, {resolution_height}]",
            'caption_extension = ".txt"',
            f"batch_size = {batch_size}",
            "enable_bucket = true",
            "bucket_no_upscale = false",
            "",
            "[[datasets]]",
            f'image_directory = "{dataset_dir / "images"}"',
            f'control_directory = "{dataset_dir / "controls"}"',
            f'cache_directory = "{dataset_dir / "cache"}"',
            f"num_repeats = {num_repeats}",
            "control_resolution = [1024, 1024]",
            "no_resize_control = false",
            "",
        ]
    )


def run_command(command: list[str]) -> None:
    print("[modal] " + shlex.join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=MUSUBI_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ | {"PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"command failed with exit code {return_code}: {shlex.join(command)}")


@app.function(image=image, gpu=GPU, timeout=TIMEOUT, volumes={str(DATA_ROOT): data_volume})
def run_qwen_edit_training(dataset_slug: str, payload: dict[str, object]) -> str:
    dataset_dir = DATA_ROOT / "datasets" / dataset_slug
    dataset_config = dataset_dir / "dataset.toml"
    output_dir = Path(str(payload.get("modal_output_dir") or DATA_ROOT / "outputs" / dataset_slug))
    output_name = str(payload.get("output_name") or f"{dataset_slug}_qwen_edit_lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_config.write_text(render_remote_toml(dataset_slug, payload), encoding="utf-8")

    latent_cache = [
        "python",
        "src/musubi_tuner/qwen_image_cache_latents.py",
        f"--dataset_config={dataset_config}",
        f"--vae={payload['vae']}",
        "--model_version=edit-2511",
    ]
    text_cache = [
        "python",
        "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
        f"--dataset_config={dataset_config}",
        f"--text_encoder={payload['text_encoder']}",
        f"--batch_size={payload.get('text_encoder_batch_size', 1)}",
        "--model_version=edit-2511",
    ]
    if payload.get("fp8_vl", True):
        text_cache.append("--fp8_vl")

    train = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process",
        "1",
        "--mixed_precision",
        str(payload.get("mixed_precision", "bf16")),
        "src/musubi_tuner/qwen_image_train_network.py",
        f"--dit={payload['dit']}",
        f"--vae={payload['vae']}",
        f"--text_encoder={payload['text_encoder']}",
        f"--dataset_config={dataset_config}",
        "--model_version=edit-2511",
        "--sdpa",
        f"--mixed_precision={payload.get('mixed_precision', 'bf16')}",
        f"--timestep_sampling={payload.get('timestep_sampling', 'shift')}",
        f"--weighting_scheme={payload.get('weighting_scheme', 'none')}",
        f"--discrete_flow_shift={payload.get('discrete_flow_shift', '2.2')}",
        f"--optimizer_type={payload.get('optimizer_type', 'adamw8bit')}",
        f"--learning_rate={payload.get('learning_rate', '5e-5')}",
        "--network_module=networks.lora_qwen_image",
        f"--network_dim={payload.get('network_dim', 8)}",
        f"--max_train_epochs={payload.get('max_train_epochs', 16)}",
        f"--save_every_n_epochs={payload.get('save_every_n_epochs', 1)}",
        "--seed=42",
        f"--output_dir={output_dir}",
        f"--output_name={output_name}",
    ]
    if payload.get("gradient_checkpointing", True):
        train.append("--gradient_checkpointing")
    if payload.get("fp8_base", True):
        train.append("--fp8_base")
    if payload.get("fp8_scaled", True):
        train.append("--fp8_scaled")
    blocks_to_swap = str(payload.get("blocks_to_swap", "36")).strip()
    if blocks_to_swap and blocks_to_swap != "0":
        train.append(f"--blocks_to_swap={blocks_to_swap}")
        if payload.get("use_pinned_memory_for_block_swap", True):
            train.append("--use_pinned_memory_for_block_swap")
    extra_args = str(payload.get("extra_args", "")).strip()
    if extra_args:
        train.extend(shlex.split(extra_args))

    for command in [latent_cache, text_cache, train]:
        run_command(command)
    data_volume.commit()
    return str(output_dir / output_name)


@app.local_entrypoint()
def main(dataset_slug: str, dataset_dir: str, payload_json: str) -> None:
    local_dataset_dir = Path(dataset_dir).resolve()
    payload = json.loads(payload_json)
    remote_dataset_dir = f"/datasets/{dataset_slug}"
    print(f"[modal] uploading {local_dataset_dir} to volume {VOLUME_NAME}:{remote_dataset_dir}", flush=True)
    with tempfile.TemporaryDirectory(prefix=f"{dataset_slug}-modal-upload-") as temp_dir:
        staging_dir = Path(temp_dir) / dataset_slug
        shutil.copytree(
            local_dataset_dir,
            staging_dir,
            ignore=shutil.ignore_patterns("cache", "outputs", "jobs_state.json", "jobs_history.json", "jobs"),
        )
        with data_volume.batch_upload(force=True) as batch:
            batch.put_directory(str(staging_dir), remote_dataset_dir)
    output_path = run_qwen_edit_training.remote(dataset_slug, payload)
    print(f"[modal] output_path={output_path}", flush=True)
