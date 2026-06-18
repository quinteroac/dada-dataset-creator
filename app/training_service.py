from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.lora_utils import create_dit_only_lora


SD_SCRIPTS_REPO = "https://github.com/kohya-ss/sd-scripts.git"
MUSUBI_TUNER_REPO = "https://github.com/kohya-ss/musubi-tuner.git"


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

    def sd_scripts_ready(self) -> bool:
        return (self.sd_scripts_dir / "anima_train_network.py").exists()

    def musubi_tuner_ready(self) -> bool:
        return (self.musubi_tuner_dir / "src" / "musubi_tuner" / "qwen_image_train_network.py").exists()

    def setup_command(self) -> list[str]:
        return ["git", "clone", "--depth", "1", SD_SCRIPTS_REPO, str(self.sd_scripts_dir)]

    def update_sd_scripts_command(self) -> list[str]:
        return ["git", "-C", str(self.sd_scripts_dir), "pull", "--ff-only", "--tags"]

    def setup_musubi_command(self) -> list[str]:
        return ["git", "clone", "--depth", "1", MUSUBI_TUNER_REPO, str(self.musubi_tuner_dir)]

    def setup_musubi_install_command(self) -> list[str]:
        return ["uv", "pip", "install", "-e", str(self.musubi_tuner_dir)]

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
            on_line(f"[setup] musubi-tuner already exists at {self.musubi_tuner_dir}")
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
        return result

    def _qwen_modal_payload(self, payload: dict[str, object]) -> dict[str, object]:
        modal_payload = dict(payload)
        if payload.get("modal_bake_models", True):
            modal_payload["dit"] = "/models/dit.safetensors"
            modal_payload["vae"] = "/models/vae.safetensors"
            modal_payload["text_encoder"] = "/models/text_encoder.safetensors"
        return modal_payload

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
