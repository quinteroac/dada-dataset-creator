from __future__ import annotations

import re
from pathlib import Path

from app.models import SUPPORTED_IMAGE_EXTENSIONS


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "dataset"


def safe_trigger_token(value: str) -> str:
    token = re.sub(r"\s+", "_", value.strip())
    return token or "trigger_token"


def normalize_caption(trigger_token: str, caption: str) -> str:
    parts = [part.strip() for part in caption.split(",") if part.strip()]
    if not parts or parts[0] != trigger_token:
        parts = [part for part in parts if part != trigger_token]
        parts.insert(0, trigger_token)
    return ", ".join(parts)


def is_supported_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def next_numbered_stem(existing_files: list[Path], width: int = 6) -> str:
    max_seen = 0
    for file_path in existing_files:
        if file_path.stem.isdigit():
            max_seen = max(max_seen, int(file_path.stem))
    return f"{max_seen + 1:0{width}d}"
