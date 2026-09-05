from __future__ import annotations

import asyncio
import json
import os
import shutil
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.lora_utils import create_dit_only_lora


SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts.git"
MUSUBI_TUNER_REPO = "https://github.com/kohya-ss/musubi-tuner.git"
MUSUBI_TUNER_KREA2_REF = "30c658c"
AI_TOOLKIT_REPO = "https://github.com/ostris/ai-toolkit.git"


@dataclass(slots=True)
class ProcessResult:
    return_code: int
    output_path: str = ""


class TrainingService:
    def __init__(self, vendor_dir: Path = Path("vendor"), datasets_root: Path = Path("datasets")) -> None:
        self.vendor_dir = vendor_dir
        self.datasets_root = datasets_root
        self.sd_scripts_dir = self.vendor_dir / "sd-scripts"
        self.musubi_tuner_dir = self.vendor_dir / "musubi-tuner"
        self.ai_toolkit_dir = self.vendor_dir / "ai-toolkit"

    def sd_scripts_ready(self) -> bool:
        return (self.sd_scripts_dir / "anima_train_network.py").exists()

    def musubi_tuner_ready(self) -> bool:
        return (self.musubi_tuner_dir / "src" / "musubi_tuner" / "qwen_image_train_network.py").exists()

    def krea2_ready(self) -> bool:
        return (self.musubi_tuner_dir / "src" / "musubi_tuner" / "krea2_train_network.py").exists()

    def ai_toolkit_ready(self) -> bool:
        return (self.ai_toolkit_dir / "run.py").exists()

    def setup_command(self) -> list[str]:
        return ["git", "clone", "--depth", "1", SD_SCRIPTS_REPO, str(self.sd_scripts_dir)]

    def update_sd_scripts_command(self) -> list[str]:
        return ["git", "-C", str(self.sd_scripts_dir), "pull", "--ff-only", "--tags"]

    def setup_musubi_command(self) -> list[str]:
        return ["git", "clone", "--depth", "1", MUSUBI_TUNER_REPO, str(self.musubi_tuner_dir)]

    def setup_musubi_install_command(self) -> list[str]:
        return ["uv", "pip", "install", "-e", str(self.musubi_tuner_dir)]

    def update_musubi_command(self) -> list[str]:
        return ["git", "-C", str(self.musubi_tuner_dir), "pull", "--ff-only", "--tags"]

    def setup_ai_toolkit_command(self) -> list[str]:
        return ["git", "clone", "--depth", "1", AI_TOOLKIT_REPO, str(self.ai_toolkit_dir)]

    def setup_ai_toolkit_install_command(self) -> list[str]:
        return ["uv", "pip", "install", "-r", str(self.ai_toolkit_dir / "requirements.txt")]

    async def setup_sd_scripts(
        self,
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        self.vendor_dir.mkdir(parents=True, exist_ok=True)
        if self.sd_scripts_ready():
            on_line(f"[setup] updating existing sd-scripts at {self.sd_scripts_dir}")
            result = await self.run_process(
                self.update_sd_scripts_command(),
                cwd=Path("."),
                on_line=on_line,
                on_process_start=on_process_start,
            )
        else:
            result = await self.run_process(
                self.setup_command(),
                cwd=Path("."),
                on_line=on_line,
                on_process_start=on_process_start,
            )
        if result.return_code == 0 and not self.sd_scripts_ready():
            raise RuntimeError("git clone finished but anima_train_network.py was not found")
        result.output_path = str(self.sd_scripts_dir)
        return result

    async def setup_musubi_tuner(
        self,
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        self.vendor_dir.mkdir(parents=True, exist_ok=True)
        if self.musubi_tuner_ready():
            on_line(f"[setup] updating existing musubi-tuner at {self.musubi_tuner_dir}")
            update_result = await self.run_process(
                self.update_musubi_command(),
                cwd=Path("."),
                on_line=on_line,
                on_process_start=on_process_start,
            )
            if update_result.return_code != 0:
                update_result.output_path = str(self.musubi_tuner_dir)
                return update_result
        else:
            result = await self.run_process(
                self.setup_musubi_command(),
                cwd=Path("."),
                on_line=on_line,
                on_process_start=on_process_start,
            )
            if result.return_code != 0:
                result.output_path = str(self.musubi_tuner_dir)
                return result
            if not self.musubi_tuner_ready():
                raise RuntimeError("git clone finished but qwen_image_train_network.py was not found")
        install_result = await self.run_process(
            self.setup_musubi_install_command(),
            cwd=Path("."),
            on_line=on_line,
            on_process_start=on_process_start,
        )
        install_result.output_path = str(self.musubi_tuner_dir)
        return install_result

    async def setup_ai_toolkit(
        self,
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        self.vendor_dir.mkdir(parents=True, exist_ok=True)
        if self.ai_toolkit_ready():
            on_line(f"[setup] ai-toolkit already exists at {self.ai_toolkit_dir}")
        else:
            result = await self.run_process(
                self.setup_ai_toolkit_command(),
                cwd=Path("."),
                on_line=on_line,
                on_process_start=on_process_start,
            )
            if result.return_code != 0:
                result.output_path = str(self.ai_toolkit_dir)
                return result
            if not self.ai_toolkit_ready():
                raise RuntimeError("git clone finished but ai-toolkit run.py was not found")
        install_result = await self.run_process(
            self.setup_ai_toolkit_install_command(),
            cwd=Path("."),
            on_line=on_line,
            on_process_start=on_process_start,
        )
        install_result.output_path = str(self.ai_toolkit_dir)
        return install_result

    def build_train_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_ready()
        dataset_dir = (self.datasets_root / dataset_slug).resolve()
        dataset_config = dataset_dir / "dataset.toml"
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()

        command = [
            "accelerate",
            "launch",
            "--num_cpu_threads_per_process",
            "1",
            "anima_train_network.py",
            f"--pretrained_model_name_or_path={payload['pretrained_model_name_or_path']}",
            f"--qwen3={payload['qwen3']}",
            f"--vae={payload['vae']}",
            f"--dataset_config={dataset_config}",
            f"--output_dir={output_dir}",
            f"--output_name={payload.get('output_name') or dataset_slug + '_anima_lora'}",
            "--save_model_as=safetensors",
            "--network_module=networks.lora_anima",
            f"--network_dim={payload.get('network_dim', 8)}",
            f"--learning_rate={payload.get('learning_rate', '1e-4')}",
            f"--optimizer_type={payload.get('optimizer_type', 'AdamW8bit')}",
            f"--lr_scheduler={payload.get('lr_scheduler', 'constant')}",
            f"--timestep_sampling={payload.get('timestep_sampling', 'sigmoid')}",
            f"--discrete_flow_shift={payload.get('discrete_flow_shift', '1.0')}",
            f"--max_train_epochs={payload.get('max_train_epochs', 10)}",
            f"--save_every_n_epochs={payload.get('save_every_n_epochs', 1)}",
            f"--mixed_precision={payload.get('mixed_precision', 'bf16')}",
        ]
        vae_chunk_size = int(payload.get("vae_chunk_size", 0) or 0)
        if vae_chunk_size > 0:
            command.append(f"--vae_chunk_size={vae_chunk_size}")
        optional_paths = {
            "llm_adapter_path": "--llm_adapter_path",
            "t5_tokenizer_path": "--t5_tokenizer_path",
        }
        for payload_key, arg_name in optional_paths.items():
            value = str(payload.get(payload_key, "")).strip()
            if value:
                command.append(f"{arg_name}={value}")
        for flag in ["gradient_checkpointing", "cache_latents", "vae_disable_cache"]:
            if payload.get(flag):
                command.append(f"--{flag}")
        for flag in ["qwen_image_vae_2d", "compile", "cuda_allow_tf32", "cuda_cudnn_benchmark"]:
            if payload.get(flag):
                command.append(f"--{flag}")
        if payload.get("compile"):
            compile_mode = str(payload.get("compile_mode", "default")).strip()
            compile_cache_size_limit = int(payload.get("compile_cache_size_limit", 32) or 0)
            if compile_mode:
                command.append(f"--compile_mode={compile_mode}")
            if compile_cache_size_limit > 0:
                command.append(f"--compile_cache_size_limit={compile_cache_size_limit}")
        for flag in ["save_state", "save_state_on_train_end"]:
            if payload.get(flag):
                command.append(f"--{flag}")
        resume_state_path = str(payload.get("resume_state_path", "")).strip()
        if resume_state_path:
            resume_path = Path(resume_state_path).expanduser()
            if not resume_path.is_absolute():
                resume_path = resume_path.resolve()
            command.append(f"--resume={resume_path}")
        if not payload.get("train_text_encoder", False):
            command.append("--network_train_unet_only")
        if payload.get("cache_text_encoder_outputs") and not self._dataset_uses_shuffle_caption(dataset_config):
            command.append("--cache_text_encoder_outputs")
        extra_args = str(payload.get("extra_args", "")).strip()
        if extra_args:
            command.extend(shlex.split(extra_args))
        return command

    def build_qwen_latent_cache_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_musubi_ready()
        dataset_config = (self.datasets_root / dataset_slug / "dataset.toml").resolve()
        return [
            "python",
            "src/musubi_tuner/qwen_image_cache_latents.py",
            f"--dataset_config={dataset_config}",
            f"--vae={payload['vae']}",
            "--model_version=edit-2511",
        ]

    def build_qwen_text_cache_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_musubi_ready()
        dataset_config = (self.datasets_root / dataset_slug / "dataset.toml").resolve()
        command = [
            "python",
            "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py",
            f"--dataset_config={dataset_config}",
            f"--text_encoder={payload['text_encoder']}",
            f"--batch_size={payload.get('text_encoder_batch_size', 1)}",
            "--model_version=edit-2511",
        ]
        if payload.get("fp8_vl", True):
            command.append("--fp8_vl")
        return command

    def build_qwen_train_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_musubi_ready()
        dataset_dir = (self.datasets_root / dataset_slug).resolve()
        dataset_config = dataset_dir / "dataset.toml"
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        command = [
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
            f"--output_name={payload.get('output_name') or dataset_slug + '_qwen_edit_lora'}",
        ]
        if payload.get("gradient_checkpointing", True):
            command.append("--gradient_checkpointing")
        if payload.get("fp8_base", True):
            command.append("--fp8_base")
        if payload.get("fp8_scaled", True):
            command.append("--fp8_scaled")
        blocks_to_swap = str(payload.get("blocks_to_swap", "36")).strip()
        if blocks_to_swap and blocks_to_swap != "0":
            command.append(f"--blocks_to_swap={blocks_to_swap}")
            if payload.get("use_pinned_memory_for_block_swap", True):
                command.append("--use_pinned_memory_for_block_swap")
        extra_args = str(payload.get("extra_args", "")).strip()
        if extra_args:
            command.extend(shlex.split(extra_args))
        return command

    def build_krea2_latent_cache_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_krea2_ready()
        dataset_config = (self.datasets_root / dataset_slug / "dataset.toml").resolve()
        return [
            "python",
            "src/musubi_tuner/krea2_cache_latents.py",
            f"--dataset_config={dataset_config}",
            f"--vae={payload['vae']}",
        ]

    def build_krea2_text_cache_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_krea2_ready()
        dataset_config = (self.datasets_root / dataset_slug / "dataset.toml").resolve()
        command = [
            "python",
            "src/musubi_tuner/krea2_cache_text_encoder_outputs.py",
            f"--dataset_config={dataset_config}",
            f"--text_encoder={payload['text_encoder']}",
            f"--batch_size={payload.get('text_encoder_batch_size', 1)}",
        ]
        return command

    def build_krea2_train_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_krea2_ready()
        dataset_dir = (self.datasets_root / dataset_slug).resolve()
        dataset_config = dataset_dir / "dataset.toml"
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        command = [
            "accelerate",
            "launch",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            str(payload.get("mixed_precision", "bf16")),
            "src/musubi_tuner/krea2_train_network.py",
            f"--dit={payload['dit']}",
            f"--vae={payload['vae']}",
            f"--dataset_config={dataset_config}",
            "--sdpa",
            f"--mixed_precision={payload.get('mixed_precision', 'bf16')}",
            f"--timestep_sampling={payload.get('timestep_sampling', 'shift')}",
            f"--weighting_scheme={payload.get('weighting_scheme', 'none')}",
            f"--optimizer_type={payload.get('optimizer_type', 'adamw8bit')}",
            f"--learning_rate={payload.get('learning_rate', '1e-4')}",
            "--network_module=networks.lora_krea2",
            f"--network_dim={payload.get('network_dim', 32)}",
            f"--network_alpha={payload.get('network_alpha', payload.get('network_dim', 32))}",
            f"--max_train_epochs={payload.get('max_train_epochs', 16)}",
            f"--save_every_n_epochs={payload.get('save_every_n_epochs', 1)}",
            "--seed=42",
            f"--output_dir={output_dir}",
            f"--output_name={payload.get('output_name') or dataset_slug + '_krea2_lora'}",
        ]
        blocks_to_swap = str(payload.get("blocks_to_swap", "")).strip()
        turbo_dit = str(payload.get("turbo_dit", "")).strip()
        if payload.get("gradient_checkpointing", True):
            command.append("--gradient_checkpointing")
        if str(payload.get("timestep_sampling", "shift")).strip() != "krea2_shift":
            command.append(f"--discrete_flow_shift={payload.get('discrete_flow_shift', '2.5')}")
        fp8_base = bool(payload.get("fp8_base", True))
        fp8_scaled = bool(payload.get("fp8_scaled", True))
        if fp8_base or fp8_scaled:
            # Krea 2 requires both flags together; plain fp8 without scaled is rejected.
            command.append("--fp8_base")
            command.append("--fp8_scaled")
        if not blocks_to_swap and not turbo_dit:
            blocks_to_swap = "26"
        if turbo_dit and blocks_to_swap and blocks_to_swap != "0":
            raise ValueError(
                "Krea 2 Turbo DiT sampling cannot be combined with blocks_to_swap. "
                "Use Turbo sampling without block swap, or omit turbo_dit to sample on RAW."
            )
        if blocks_to_swap and blocks_to_swap != "0":
            command.append(f"--blocks_to_swap={blocks_to_swap}")
            if payload.get("use_pinned_memory_for_block_swap", True):
                command.append("--use_pinned_memory_for_block_swap")
            if payload.get("block_swap_h2d_only", True):
                command.append("--block_swap_h2d_only")
                command.append(f"--block_swap_ring_size={int(payload.get('block_swap_ring_size', 1) or 1)}")
        if payload.get("compile", False):
            command.append("--compile")
        # Optional Turbo DiT for sample generation during training.
        if turbo_dit:
            command.append(f"--turbo_dit={turbo_dit}")
            if payload.get("turbo_dit_cache", False):
                command.append("--turbo_dit_cache")
        extra_args = str(payload.get("extra_args", "")).strip()
        if extra_args:
            command.extend(shlex.split(extra_args))
        return command

    def build_qwen_modal_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        dataset_dir = (self.datasets_root / dataset_slug).resolve()
        modal_payload = self._qwen_modal_payload(payload)
        payload_json = json.dumps(modal_payload, separators=(",", ":"))
        volume_name = str(payload.get("modal_volume_name") or "dada-qwen-edit").strip()
        gpu = str(payload.get("modal_gpu") or "L40S").strip()
        timeout = str(payload.get("modal_timeout") or "86400").strip()
        command = [
            "env",
            f"DADA_MODAL_VOLUME={volume_name}",
            f"DADA_MODAL_GPU={gpu}",
            f"DADA_MODAL_TIMEOUT={timeout}",
        ]
        if payload.get("modal_bake_models", True):
            command.extend(
                [
                    f"DADA_MODAL_LOCAL_DIT={payload['dit']}",
                    f"DADA_MODAL_LOCAL_VAE={payload['vae']}",
                    f"DADA_MODAL_LOCAL_TEXT_ENCODER={payload['text_encoder']}",
                ]
            )
        command.extend(
            [
                "modal",
                "run",
                "app/modal_qwen_edit.py",
                "--dataset-slug",
                dataset_slug,
                "--dataset-dir",
                str(dataset_dir),
                "--payload-json",
                payload_json,
            ]
        )
        return command

    def build_krea2_modal_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        dataset_dir = (self.datasets_root / dataset_slug).resolve()
        modal_payload = self._krea2_modal_payload(payload)
        payload_json = json.dumps(modal_payload, separators=(",", ":"))
        volume_name = str(payload.get("modal_volume_name") or "dada-krea2").strip()
        gpu = str(payload.get("modal_gpu") or "RTX-PRO-6000").strip()
        timeout = str(payload.get("modal_timeout") or "86400").strip()
        command = [
            "env",
            f"DADA_MODAL_VOLUME={volume_name}",
            f"DADA_MODAL_GPU={gpu}",
            f"DADA_MODAL_TIMEOUT={timeout}",
            f"DADA_MODAL_MUSUBI_REF={self._modal_musubi_ref(payload)}",
        ]
        if payload.get("modal_bake_models", True):
            command.extend(
                [
                    f"DADA_MODAL_LOCAL_DIT={payload['dit']}",
                    f"DADA_MODAL_LOCAL_VAE={payload['vae']}",
                    f"DADA_MODAL_LOCAL_TEXT_ENCODER={payload['text_encoder']}",
                ]
            )
        command.extend(
            [
                "modal",
                "run",
                "app/modal_krea2.py",
                "--dataset-slug",
                dataset_slug,
                "--dataset-dir",
                str(dataset_dir),
                "--payload-json",
                payload_json,
            ]
        )
        return command

    def build_minimax_h3_modal_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        config = json.loads(self.build_minimax_h3_config(dataset_slug, payload))
        remote_output = str(payload.get("modal_output_dir") or f"/data/outputs/{dataset_slug}").strip()
        if not remote_output.startswith("/data/outputs/") or ".." in Path(remote_output).parts:
            raise ValueError("Modal output directory must be inside /data/outputs/")
        process = config["config"]["process"][0]
        process["training_folder"] = remote_output
        # The Modal entrypoint replaces this with an isolated upload directory.
        process["datasets"][0]["folder_path"] = f"/data/datasets/{dataset_slug}/images"
        return [
            "env",
            f"DADA_MODAL_VOLUME={str(payload.get('modal_volume_name') or 'dada-minimax-h3').strip()}",
            f"DADA_MODAL_GPU={str(payload.get('modal_gpu') or 'RTX-PRO-6000').strip()}",
            f"DADA_MODAL_TIMEOUT={int(payload.get('modal_timeout') or 86400)}",
            "modal", "run", "app/modal_minimax_h3.py",
            "--dataset-slug", dataset_slug,
            "--dataset-dir", str((self.datasets_root / dataset_slug).resolve()),
            "--config-json", json.dumps(config),
        ]

    async def train_minimax_h3_lora_on_modal(
        self, dataset_slug: str, payload: dict[str, object], on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        command = self.build_minimax_h3_modal_command(dataset_slug, payload)
        (self.datasets_root / dataset_slug / "minimax_h3_modal_training_config.json").write_text(
            command[-1], encoding="utf-8",
        )
        on_line("[modal] " + shlex.join(command))
        result = await self.run_process(command, cwd=Path("."), on_line=on_line, on_process_start=on_process_start)
        volume = str(payload.get("modal_volume_name") or "dada-minimax-h3").strip()
        remote_output = json.loads(command[-1])["config"]["process"][0]["training_folder"]
        name = json.loads(command[-1])["config"]["name"]
        remote_run = f"{remote_output.rstrip('/')}/{name}"
        result.output_path = f"modal://{volume}{remote_run.removeprefix('/data')}"
        if result.return_code == 0:
            result.output_path = await self.download_modal_outputs(
                dataset_slug, payload, volume, remote_run, on_line, on_process_start,
                fallback_output_path=result.output_path,
            )
        return result

    def build_minimax_h3_config(self, dataset_slug: str, payload: dict[str, object]) -> str:
        """JSON is valid YAML; serialize user paths and captions without interpolation."""
        dataset_dir = (self.datasets_root / dataset_slug).resolve()
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser().resolve()
        resolutions = [int(v.strip()) for v in str(payload.get("resolution_list", "512")).split(",")]
        if not resolutions or any(v < 32 or v % 32 for v in resolutions):
            raise ValueError("MiniMax H3 resolutions must be positive multiples of 32")
        model_path = str(payload.get("model_path", "Comfy-Org/MiniMax-H3")).strip()
        if not model_path:
            raise ValueError("MiniMax H3 model path is required")
        name = str(payload.get("output_name") or dataset_slug + "_minimax_h3_lora")
        if name in {".", ".."} or any(c in name for c in '/\\'):
            raise ValueError("Output name must be a name, not a path")
        numeric = {key: int(payload.get(key, default)) for key, default in {
            "network_dim": 16, "steps": 1000, "save_every": 250,
            "batch_size": 1, "num_repeats": 10,
        }.items()}
        if any(v < 1 for v in numeric.values()):
            raise ValueError("Rank, steps, save interval, batch size and repeats must be positive")
        lr = float(payload.get("learning_rate", 1e-4))
        if not 0 < lr < float("inf"):
            raise ValueError("Learning rate must be finite and positive")
        model = {
            "name_or_path": model_path, "arch": "minimax_h3",
            "dtype": "bf16", "quantize": False, "quantize_te": False,
            "low_vram": True, "layer_offloading": bool(payload.get("layer_offloading", True)),
            "layer_offloading_text_encoder_percent": 0.95,
            "layer_offloading_transformer_percent": 0.95,
            "model_kwargs": {"partition": "fl2va_pruned", "sample_audio": False},
        }
        if payload.get("assistant_lora_path"):
            model["assistant_lora_path"] = str(payload["assistant_lora_path"])
        process = {
            "type": "sd_trainer", "training_folder": str(output_dir), "device": "cuda:0",
            "network": {"type": "lora", "linear": numeric["network_dim"], "linear_alpha": numeric["network_dim"]},
            "save": {"dtype": "float16", "save_every": numeric["save_every"], "max_step_saves_to_keep": 2, "push_to_hub": False},
            "datasets": [{
                "folder_path": str(dataset_dir / "images"), "caption_ext": "txt",
                "caption_dropout_rate": 0.0, "shuffle_tokens": False,
                "cache_latents_to_disk": True, "cache_text_embeddings": True,
                "resolution": resolutions, "num_repeats": numeric["num_repeats"],
                "num_frames": 1, "do_audio": False,
            }],
            "train": {
                "batch_size": numeric["batch_size"], "steps": numeric["steps"],
                "gradient_accumulation_steps": 1, "train_unet": True,
                "train_text_encoder": False, "gradient_checkpointing": True,
                "noise_scheduler": "flowmatch", "optimizer": "adamw8bit", "lr": lr,
                "dtype": "bf16", "disable_sampling": True,
            },
            "model": model,
        }
        return json.dumps({"job": "extension", "config": {"name": name, "process": [process]}}, indent=2) + "\n"

    def build_minimax_h3_train_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_ai_toolkit_ready()
        model_file = self.ai_toolkit_dir / "extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py"
        if not model_file.is_file():
            raise RuntimeError("Installed AI Toolkit lacks MiniMax H3 support; update vendor/ai-toolkit and rerun setup")
        config_path = (self.datasets_root / dataset_slug / "minimax_h3_training_config.yaml").resolve()
        config_path.write_text(self.build_minimax_h3_config(dataset_slug, payload), encoding="utf-8")
        return ["python", "run.py", str(config_path)]

    async def train_minimax_h3_lora(
        self, dataset_slug: str, payload: dict[str, object], on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        command = self.build_minimax_h3_train_command(dataset_slug, payload)
        dataset_dir = self.datasets_root / dataset_slug
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "minimax_h3_training_payload.json").write_text(
            json.dumps(payload | {"command": command}, indent=2), encoding="utf-8",
        )
        on_line("[minimax_h3] " + shlex.join(command))
        result = await self.run_process(command, cwd=self.ai_toolkit_dir, on_line=on_line, on_process_start=on_process_start)
        run_dir = output_dir / str(payload.get("output_name") or dataset_slug + "_minimax_h3_lora")
        models = sorted(run_dir.glob("**/*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
        result.output_path = str(models[0] if models else run_dir)
        return result

    def build_ideogram4_config(self, dataset_slug: str, payload: dict[str, object]) -> str:
        dataset_dir = (self.datasets_root / dataset_slug).resolve()
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        resolution_values = str(payload.get("resolution_list", "512,768,1024")).strip() or "512,768,1024"
        resolutions = [
            int(value.strip())
            for value in resolution_values.replace("[", "").replace("]", "").split(",")
            if value.strip()
        ]
        if not resolutions:
            resolutions = [1024]
        bool_value = lambda key, default=True: "true" if payload.get(key, default) else "false"
        return "\n".join(
            [
                "---",
                "job: extension",
                "config:",
                f"  name: \"{payload.get('output_name') or dataset_slug + '_ideogram4_lora'}\"",
                "  process:",
                "    - type: 'sd_trainer'",
                f"      training_folder: \"{output_dir}\"",
                "      device: cuda:0",
                "      network:",
                "        type: \"lora\"",
                f"        linear: {int(payload.get('network_dim', 16) or 16)}",
                f"        linear_alpha: {int(payload.get('network_alpha', payload.get('network_dim', 16)) or 16)}",
                "      save:",
                "        dtype: float16",
                f"        save_every: {int(payload.get('save_every', 250) or 250)}",
                "        max_step_saves_to_keep: 2",
                "        push_to_hub: false",
                "      datasets:",
                f"        - folder_path: \"{dataset_dir / 'images'}\"",
                "          caption_ext: \"txt\"",
                "          caption_dropout_rate: 0.0",
                "          shuffle_tokens: false",
                "          cache_latents_to_disk: true",
                f"          resolution: [ {', '.join(str(value) for value in resolutions)} ]",
                "      train:",
                f"        batch_size: {int(payload.get('batch_size', 1) or 1)}",
                f"        steps: {int(payload.get('steps', 1000) or 1000)}",
                "        gradient_accumulation_steps: 1",
                "        train_unet: true",
                "        train_text_encoder: false",
                f"        gradient_checkpointing: {bool_value('gradient_checkpointing', True)}",
                "        noise_scheduler: \"flowmatch\"",
                f"        optimizer: \"{payload.get('optimizer', 'adamw8bit')}\"",
                f"        lr: {payload.get('learning_rate', '1e-4')}",
                "        ema_config:",
                "          use_ema: true",
                "          ema_decay: 0.99",
                f"        dtype: {payload.get('dtype', 'bf16')}",
                "      model:",
                f"        name_or_path: \"{payload['model_path']}\"",
                "        arch: ideogram4",
                f"        quantize: {bool_value('quantize', True)}",
                f"        quantize_te: {bool_value('quantize_te', True)}",
                f"        low_vram: {bool_value('low_vram', True)}",
                f"        layer_offloading: {bool_value('layer_offloading', True)}",
                f"        layer_offloading_text_encoder_percent: {int(payload.get('layer_offloading_text_encoder_percent', 95) or 95)}",
                f"        layer_offloading_transformer_percent: {int(payload.get('layer_offloading_transformer_percent', 95) or 95)}",
                "",
            ]
        )

    def write_ideogram4_config(self, dataset_slug: str, payload: dict[str, object]) -> Path:
        dataset_dir = self.datasets_root / dataset_slug
        config_path = dataset_dir / "ideogram4_training_config.yaml"
        config_path.write_text(self.build_ideogram4_config(dataset_slug, payload), encoding="utf-8")
        return config_path

    def build_ideogram4_train_command(self, dataset_slug: str, payload: dict[str, object]) -> list[str]:
        self._require_ai_toolkit_ready()
        config_path = self.write_ideogram4_config(dataset_slug, payload).resolve()
        command = ["python", "run.py", str(config_path)]
        extra_args = str(payload.get("extra_args", "")).strip()
        if extra_args:
            command.extend(shlex.split(extra_args))
        return command

    async def train_anima_lora(
        self,
        dataset_slug: str,
        payload: dict[str, object],
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        command = self.build_train_command(dataset_slug, payload)
        dataset_dir = self.datasets_root / dataset_slug
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "training_config.json").write_text(
            json.dumps(payload | {"command": command}, indent=2),
            encoding="utf-8",
        )
        on_line("[training] " + shlex.join(command))
        if payload.get("cache_text_encoder_outputs") and "--cache_text_encoder_outputs" not in command:
            on_line(
                "[training] skipped --cache_text_encoder_outputs because dataset.toml uses shuffle_caption=true"
            )
        result = await self.run_process(
            command,
            cwd=self.sd_scripts_dir,
            on_line=on_line,
            on_process_start=on_process_start,
        )
        safetensors = sorted(output_dir.glob("*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
        if safetensors:
            result.output_path = str(create_dit_only_lora(safetensors[0]))
        else:
            result.output_path = str(output_dir)
        return result

    async def train_qwen_edit_lora(
        self,
        dataset_slug: str,
        payload: dict[str, object],
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        self._require_musubi_ready()
        self._validate_qwen_training_models(payload)
        dataset_dir = self.datasets_root / dataset_slug
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "qwen_training_config.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        phases = [
            ("latent-cache", self.build_qwen_latent_cache_command(dataset_slug, payload)),
            ("text-cache", self.build_qwen_text_cache_command(dataset_slug, payload)),
            ("training", self.build_qwen_train_command(dataset_slug, payload)),
        ]
        final_result = ProcessResult(return_code=0, output_path=str(output_dir))
        for phase, command in phases:
            on_line(f"[{phase}] " + shlex.join(command))
            result = await self.run_process(
                command,
                cwd=self.musubi_tuner_dir,
                on_line=on_line,
                on_process_start=on_process_start,
            )
            final_result = result
            if result.return_code != 0:
                return result
        safetensors = sorted(output_dir.glob("*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
        final_result.output_path = str(safetensors[0] if safetensors else output_dir)
        return final_result

    async def train_qwen_edit_lora_on_modal(
        self,
        dataset_slug: str,
        payload: dict[str, object],
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        self._validate_qwen_training_models(payload)
        dataset_dir = self.datasets_root / dataset_slug
        (dataset_dir / "qwen_modal_training_config.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        if payload.get("modal_bake_models", True):
            for key, remote_path in [
                ("dit", "/models/dit.safetensors"),
                ("vae", "/models/vae.safetensors"),
                ("text_encoder", "/models/text_encoder.safetensors"),
            ]:
                model_path = Path(str(payload[key])).expanduser().resolve()
                if not model_path.exists():
                    raise FileNotFoundError(f"Modal baked model source does not exist: {model_path}")
                size_gb = model_path.stat().st_size / (1024**3)
                on_line(f"[modal] baking {key}: {model_path} -> {remote_path} ({size_gb:.2f} GiB)")
        command = self.build_qwen_modal_command(dataset_slug, payload)
        on_line("[modal] " + shlex.join(command))
        result = await self.run_process(
            command,
            cwd=Path("."),
            on_line=on_line,
            on_process_start=on_process_start,
        )
        volume_name = str(payload.get("modal_volume_name") or "dada-qwen-edit").strip()
        output_dir = str(payload.get("modal_output_dir") or f"/data/outputs/{dataset_slug}").strip()
        result.output_path = f"modal://{volume_name}{output_dir.removeprefix('/data')}"
        if result.return_code == 0:
            result.output_path = await self.download_modal_outputs(
                dataset_slug,
                payload,
                volume_name,
                output_dir,
                on_line,
                on_process_start,
                fallback_output_path=result.output_path,
            )
        return result

    async def train_krea2_lora_on_modal(
        self,
        dataset_slug: str,
        payload: dict[str, object],
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        self._validate_krea2_training_models(payload)
        dataset_dir = self.datasets_root / dataset_slug
        (dataset_dir / "krea2_modal_training_config.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        if payload.get("modal_bake_models", True):
            for key, remote_path in [
                ("dit", "/models/dit.safetensors"),
                ("vae", "/models/vae.safetensors"),
                ("text_encoder", "/models/text_encoder.safetensors"),
            ]:
                model_path = Path(str(payload[key])).expanduser().resolve()
                if not model_path.exists():
                    raise FileNotFoundError(f"Modal baked model source does not exist: {model_path}")
                size_gb = model_path.stat().st_size / (1024**3)
                on_line(f"[modal] baking {key}: {model_path} -> {remote_path} ({size_gb:.2f} GiB)")
        command = self.build_krea2_modal_command(dataset_slug, payload)
        on_line("[modal] " + shlex.join(command))
        result = await self.run_process(
            command,
            cwd=Path("."),
            on_line=on_line,
            on_process_start=on_process_start,
        )
        volume_name = str(payload.get("modal_volume_name") or "dada-krea2").strip()
        output_dir = str(payload.get("modal_output_dir") or f"/data/outputs/{dataset_slug}").strip()
        result.output_path = f"modal://{volume_name}{output_dir.removeprefix('/data')}"
        if result.return_code == 0:
            result.output_path = await self.download_modal_outputs(
                dataset_slug,
                payload,
                volume_name,
                output_dir,
                on_line,
                on_process_start,
                fallback_output_path=result.output_path,
            )
        return result

    async def train_ideogram4_lora(
        self,
        dataset_slug: str,
        payload: dict[str, object],
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        self._require_ai_toolkit_ready()
        dataset_dir = self.datasets_root / dataset_slug
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_ideogram4_train_command(dataset_slug, payload)
        (dataset_dir / "ideogram4_training_payload.json").write_text(
            json.dumps(payload | {"command": command}, indent=2),
            encoding="utf-8",
        )
        on_line("[ideogram4] " + shlex.join(command))
        result = await self.run_process(
            command,
            cwd=self.ai_toolkit_dir,
            on_line=on_line,
            on_process_start=on_process_start,
        )
        safetensors = sorted(output_dir.glob("**/*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
        result.output_path = str(safetensors[0] if safetensors else output_dir)
        return result

    async def train_krea2_lora(
        self,
        dataset_slug: str,
        payload: dict[str, object],
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        self._require_krea2_ready()
        self._validate_krea2_training_models(payload)
        dataset_dir = self.datasets_root / dataset_slug
        output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not output_dir.is_absolute():
            output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "krea2_training_config.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        phases = [
            ("latent-cache", self.build_krea2_latent_cache_command(dataset_slug, payload)),
            ("text-cache", self.build_krea2_text_cache_command(dataset_slug, payload)),
            ("training", self.build_krea2_train_command(dataset_slug, payload)),
        ]
        final_result = ProcessResult(return_code=0, output_path=str(output_dir))
        for phase, command in phases:
            on_line(f"[{phase}] " + shlex.join(command))
            result = await self.run_process(
                command,
                cwd=self.musubi_tuner_dir,
                on_line=on_line,
                on_process_start=on_process_start,
            )
            final_result = result
            if result.return_code != 0:
                return result
        safetensors = sorted(output_dir.glob("*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
        final_result.output_path = str(safetensors[0] if safetensors else output_dir)
        return final_result

    def _qwen_modal_payload(self, payload: dict[str, object]) -> dict[str, object]:
        modal_payload = dict(payload)
        if payload.get("modal_bake_models", True):
            modal_payload["dit"] = "/models/dit.safetensors"
            modal_payload["vae"] = "/models/vae.safetensors"
            modal_payload["text_encoder"] = "/models/text_encoder.safetensors"
        return modal_payload

    def _krea2_modal_payload(self, payload: dict[str, object]) -> dict[str, object]:
        modal_payload = dict(payload)
        if payload.get("modal_bake_models", True):
            modal_payload["dit"] = "/models/dit.safetensors"
            modal_payload["vae"] = "/models/vae.safetensors"
            modal_payload["text_encoder"] = "/models/text_encoder.safetensors"
        return modal_payload

    async def download_modal_outputs(
        self,
        dataset_slug: str,
        payload: dict[str, object],
        volume_name: str,
        remote_output_dir: str,
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
        fallback_output_path: str = "",
    ) -> str:
        dataset_dir = self.datasets_root / dataset_slug
        local_output_dir = Path(str(payload.get("output_dir") or dataset_dir / "outputs")).expanduser()
        if not local_output_dir.is_absolute():
            local_output_dir = local_output_dir.resolve()
        local_output_dir.mkdir(parents=True, exist_ok=True)
        remote_path = str(remote_output_dir).strip() or f"/data/outputs/{dataset_slug}"
        if remote_path.startswith("/data"):
            remote_path = remote_path.removeprefix("/data") or "/"
        if not remote_path.startswith("/"):
            remote_path = "/" + remote_path
        with tempfile.TemporaryDirectory(prefix="modal-download-", dir=dataset_dir) as temp_dir:
            staging_dir = Path(temp_dir)
            command = [
                "modal",
                "volume",
                "get",
                "--force",
                volume_name,
                remote_path,
                str(staging_dir),
            ]
            on_line("[modal-download] " + shlex.join(command))
            download_result = await self.run_process(
                command,
                cwd=Path("."),
                on_line=on_line,
                on_process_start=on_process_start,
            )
            if download_result.return_code != 0:
                on_line(
                    f"[modal-download] download failed with exit code {download_result.return_code}; "
                    f"remote output remains at {fallback_output_path}"
                )
                return fallback_output_path
            downloaded = sorted(staging_dir.rglob("*.safetensors"))
            for source in downloaded:
                target = local_output_dir / source.name
                if target.exists():
                    target.unlink()
                shutil.move(str(source), str(target))
        safetensors = sorted(local_output_dir.glob("*.safetensors"), key=lambda path: path.stat().st_mtime, reverse=True)
        final_output = local_output_dir / f"{payload.get('output_name')}.safetensors"
        if payload.get("output_name") and final_output.exists():
            on_line(f"[modal-download] downloaded final safetensors: {final_output}")
            return str(final_output)
        if safetensors:
            on_line(f"[modal-download] downloaded latest safetensors: {safetensors[0]}")
            return str(safetensors[0])
        on_line(f"[modal-download] downloaded outputs to {local_output_dir}")
        return str(local_output_dir)

    def _modal_musubi_ref(self, payload: dict[str, object]) -> str:
        ref = str(payload.get("modal_musubi_ref") or "").strip()
        if ref:
            return ref
        if (self.musubi_tuner_dir / ".git").exists():
            try:
                return (
                    subprocess.check_output(
                        ["git", "-C", str(self.musubi_tuner_dir), "rev-parse", "HEAD"],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    )
                    .strip()
                    or MUSUBI_TUNER_KREA2_REF
                )
            except (OSError, subprocess.CalledProcessError):
                pass
        return MUSUBI_TUNER_KREA2_REF

    async def run_process(
        self,
        command: list[str],
        cwd: Path,
        on_line: Callable[[str], None],
        on_process_start: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> ProcessResult:
        env = os.environ | {"PYTHONUNBUFFERED": "1"}
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        if on_process_start is not None:
            on_process_start(process)
        assert process.stdout is not None
        pending = ""
        while chunk := await process.stdout.read(4096):
            pending += chunk.decode(errors="replace")
            pending = pending.replace("\r", "\n")
            *lines, pending = pending.split("\n")
            for line in lines:
                if line:
                    on_line(line.rstrip())
        if pending.strip():
            on_line(pending.rstrip())
        return_code = await process.wait()
        return ProcessResult(return_code=return_code)

    def _require_ready(self) -> None:
        if not self.sd_scripts_ready():
            raise FileNotFoundError(
                f"{self.sd_scripts_dir / 'anima_train_network.py'} not found. Run setup sd-scripts first."
            )

    def _require_musubi_ready(self) -> None:
        if not self.musubi_tuner_ready():
            raise FileNotFoundError(
                f"{self.musubi_tuner_dir / 'src' / 'musubi_tuner' / 'qwen_image_train_network.py'} not found. "
                "Run setup musubi-tuner first."
            )

    def _require_ai_toolkit_ready(self) -> None:
        if not self.ai_toolkit_ready():
            raise FileNotFoundError(
                f"{self.ai_toolkit_dir / 'run.py'} not found. Run setup AI Toolkit first."
            )

    def _require_krea2_ready(self) -> None:
        if not self.krea2_ready():
            raise FileNotFoundError(
                f"{self.musubi_tuner_dir / 'src' / 'musubi_tuner' / 'krea2_train_network.py'} not found. "
                "Run setup musubi-tuner (it pulls the latest musubi-tuner with Krea 2 support) first."
            )

    def _validate_krea2_training_models(self, payload: dict[str, object]) -> None:
        fp8_paths = []
        for key in ["dit", "text_encoder"]:
            value = str(payload.get(key, "")).strip()
            if value and "fp8" in Path(value).name.lower():
                fp8_paths.append(f"{key}={value}")
        if fp8_paths:
            raise ValueError(
                "Musubi Krea 2 training requires bf16 checkpoint files for dit/text_encoder; "
                "use --fp8_base/--fp8_scaled for DiT VRAM savings instead of fp8 model files. "
                "Unsupported paths: " + ", ".join(fp8_paths)
            )

    def _validate_qwen_training_models(self, payload: dict[str, object]) -> None:
        fp8_paths = []
        for key in ["dit", "text_encoder"]:
            value = str(payload.get(key, "")).strip()
            if value and "fp8" in Path(value).name.lower():
                fp8_paths.append(f"{key}={value}")
        if fp8_paths:
            raise ValueError(
                "Musubi Qwen training requires bf16 checkpoint files for dit/text_encoder; "
                "use --fp8_base/--fp8_scaled/--fp8_vl for VRAM savings instead of fp8 model files. "
                "Unsupported paths: " + ", ".join(fp8_paths)
            )

    def _dataset_uses_shuffle_caption(self, dataset_config: Path) -> bool:
        if not dataset_config.exists():
            return False
        return "shuffle_caption = true" in dataset_config.read_text(encoding="utf-8")
