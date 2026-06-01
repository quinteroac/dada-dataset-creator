from __future__ import annotations

from pathlib import Path


def create_dit_only_lora(input_path: Path, output_path: Path | None = None) -> Path:
    try:
        from safetensors.torch import safe_open, save_file
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Filtering LoRA checkpoints requires safetensors and torch. "
            "Install the training dependencies before running Anima training."
        ) from exc

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_dit_only{input_path.suffix}")
    tensors = {}
    metadata = {}
    with safe_open(input_path, framework="pt", device="cpu") as source:
        metadata = dict(source.metadata() or {})
        for key in source.keys():
            if key.startswith("lora_te"):
                continue
            tensors[key] = source.get_tensor(key)
    metadata["dada_filtered"] = "removed lora_te text encoder weights"
    save_file(tensors, str(output_path), metadata=metadata)
    return output_path
