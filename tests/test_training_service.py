from pathlib import Path

import asyncio
import pytest

from app.training_service import TrainingService


def test_build_train_command_uses_safe_arg_list_and_preset(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    sd_scripts = vendor / "sd-scripts"
    sd_scripts.mkdir(parents=True)
    (sd_scripts / "anima_train_network.py").write_text("# script", encoding="utf-8")
    datasets = tmp_path / "datasets"
    (datasets / "blue").mkdir(parents=True)
    (datasets / "blue" / "dataset.toml").write_text("[general]", encoding="utf-8")
    service = TrainingService(vendor_dir=vendor, datasets_root=datasets)

    command = service.build_train_command(
        "blue",
        {
            "pretrained_model_name_or_path": "/models/anima.safetensors",
            "qwen3": "/models/qwen3",
            "vae": "/models/vae.safetensors",
            "output_name": "blue_lora",
            "network_dim": 16,
            "learning_rate": "2e-4",
            "gradient_checkpointing": True,
            "extra_args": '--network_args "train_llm_adapter=True"',
        },
    )

    assert command[:5] == ["accelerate", "launch", "--num_cpu_threads_per_process", "1", "anima_train_network.py"]
    assert "--network_module=networks.lora_anima" in command
    assert "--dataset_config=" + str((datasets / "blue" / "dataset.toml").resolve()) in command
    assert "--gradient_checkpointing" in command
    assert "--network_train_unet_only" in command
    assert "train_llm_adapter=True" in command


def test_build_train_command_skips_text_encoder_cache_when_shuffle_caption(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    sd_scripts = vendor / "sd-scripts"
    sd_scripts.mkdir(parents=True)
    (sd_scripts / "anima_train_network.py").write_text("# script", encoding="utf-8")
    datasets = tmp_path / "datasets"
    (datasets / "blue").mkdir(parents=True)
    (datasets / "blue" / "dataset.toml").write_text("[general]\nshuffle_caption = true\n", encoding="utf-8")
    service = TrainingService(vendor_dir=vendor, datasets_root=datasets)

    command = service.build_train_command(
        "blue",
        {
            "pretrained_model_name_or_path": "model",
            "qwen3": "qwen",
            "vae": "vae",
            "cache_text_encoder_outputs": True,
        },
    )

    assert "--cache_text_encoder_outputs" not in command


def test_build_train_command_can_train_text_encoder_when_requested(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    sd_scripts = vendor / "sd-scripts"
    sd_scripts.mkdir(parents=True)
    (sd_scripts / "anima_train_network.py").write_text("# script", encoding="utf-8")
    datasets = tmp_path / "datasets"
    (datasets / "blue").mkdir(parents=True)
    (datasets / "blue" / "dataset.toml").write_text("[general]\n", encoding="utf-8")
    service = TrainingService(vendor_dir=vendor, datasets_root=datasets)

    command = service.build_train_command(
        "blue",
        {
            "pretrained_model_name_or_path": "model",
            "qwen3": "qwen",
            "vae": "vae",
            "train_text_encoder": True,
        },
    )

    assert "--network_train_unet_only" not in command


def test_setup_command_targets_vendor_sd_scripts(tmp_path: Path) -> None:
    service = TrainingService(vendor_dir=tmp_path / "vendor")

    command = service.setup_command()

    assert command[:3] == ["git", "clone", "--depth"]
    assert command[-1].endswith("vendor/sd-scripts")


def test_build_qwen_commands_use_musubi_and_vram_defaults(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    musubi = vendor / "musubi-tuner" / "src" / "musubi_tuner"
    musubi.mkdir(parents=True)
    (musubi / "qwen_image_train_network.py").write_text("# script", encoding="utf-8")
    datasets = tmp_path / "datasets"
    (datasets / "qwen").mkdir(parents=True)
    (datasets / "qwen" / "dataset.toml").write_text("[general]", encoding="utf-8")
    service = TrainingService(vendor_dir=vendor, datasets_root=datasets)
    payload = {"dit": "dit", "vae": "vae", "text_encoder": "te"}

    latent = service.build_qwen_latent_cache_command("qwen", payload)
    text = service.build_qwen_text_cache_command("qwen", payload)
    train = service.build_qwen_train_command("qwen", payload)

    assert "src/musubi_tuner/qwen_image_cache_latents.py" in latent
    assert "--vae=vae" in latent
    assert "--model_version=edit-2511" in latent
    assert "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py" in text
    assert "--text_encoder=te" in text
    assert "--fp8_vl" in text
    assert "src/musubi_tuner/qwen_image_train_network.py" in train
    assert "--model_version=edit-2511" in train
    assert "--network_module=networks.lora_qwen_image" in train
    assert "--fp8_base" in train
    assert "--fp8_scaled" in train
    assert "--network_dim=8" in train
    assert "--blocks_to_swap=36" in train
    assert "--use_pinned_memory_for_block_swap" in train


def test_build_qwen_modal_command_uses_modal_cli_and_volume(tmp_path: Path) -> None:
    datasets = tmp_path / "datasets"
    (datasets / "qwen").mkdir(parents=True)
    service = TrainingService(datasets_root=datasets)
    payload = {
        "dit": "/data/models/dit.safetensors",
        "vae": "/data/models/vae.safetensors",
        "text_encoder": "/data/models/te.safetensors",
        "modal_volume_name": "qwen-volume",
        "modal_gpu": "A100-40GB",
        "modal_timeout": 7200,
    }

    command = service.build_qwen_modal_command("qwen", payload)

    assert "DADA_MODAL_VOLUME=qwen-volume" in command
    assert "DADA_MODAL_GPU=A100-40GB" in command
    assert "DADA_MODAL_TIMEOUT=7200" in command
    assert "DADA_MODAL_LOCAL_DIT=/data/models/dit.safetensors" in command
    assert "modal" in command
    assert "app/modal_qwen_edit.py" in command
    assert "--dataset-slug" in command
    assert "--payload-json" in command
    payload_json = command[command.index("--payload-json") + 1]
    assert '"/models/dit.safetensors"' in payload_json
    assert '"/models/vae.safetensors"' in payload_json
    assert '"/models/text_encoder.safetensors"' in payload_json


def test_validate_qwen_training_models_rejects_fp8_checkpoints() -> None:
    service = TrainingService()

    with pytest.raises(ValueError, match="requires bf16 checkpoint files"):
        service._validate_qwen_training_models(
            {
                "dit": "/models/qwen_image_edit_2511_fp8mixed.safetensors",
                "text_encoder": "/models/qwen_2.5_vl_7b_fp8_scaled.safetensors",
            }
        )


def test_setup_musubi_command_installs_editable_package(tmp_path: Path) -> None:
    service = TrainingService(vendor_dir=tmp_path / "vendor")

    command = service.setup_musubi_install_command()

    assert command[:4] == ["uv", "pip", "install", "-e"]
    assert command[-1].endswith("vendor/musubi-tuner")


def test_setup_musubi_installs_when_repo_already_exists(tmp_path: Path) -> None:
    class RecordingTrainingService(TrainingService):
        def __init__(self) -> None:
            super().__init__(vendor_dir=tmp_path / "vendor")
            self.commands = []

        async def run_process(self, command, cwd, on_line, on_process_start=None):
            self.commands.append(command)
            return type("Result", (), {"return_code": 0, "output_path": ""})()

    service = RecordingTrainingService()
    musubi = service.musubi_tuner_dir / "src" / "musubi_tuner"
    musubi.mkdir(parents=True)
    (musubi / "qwen_image_train_network.py").write_text("# script", encoding="utf-8")

    result = asyncio.run(service.setup_musubi_tuner(lambda line: None))

    assert result.return_code == 0
    assert service.commands == [service.setup_musubi_install_command()]
