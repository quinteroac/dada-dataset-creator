from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DATASET_TYPE_QWEN_IMAGE_EDIT_2511 = "qwen_image_edit_2511"
DATASET_TYPE_IDEOGRAM4 = "ideogram4"


@dataclass(slots=True)
class DatasetSettings:
    name: str
    slug: str
    dataset_type: str
    trigger_token: str
    resolution_width: int = 1024
    resolution_height: int = 1024
    batch_size: int = 1
    num_repeats: int = 10
    min_bucket_reso: int = 512
    max_bucket_reso: int = 1536
    bucket_reso_steps: int = 16
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.resolution_width, self.resolution_height)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "slug": self.slug,
            "dataset_type": self.dataset_type,
            "trigger_token": self.trigger_token,
            "resolution_width": self.resolution_width,
            "resolution_height": self.resolution_height,
            "batch_size": self.batch_size,
            "num_repeats": self.num_repeats,
            "min_bucket_reso": self.min_bucket_reso,
            "max_bucket_reso": self.max_bucket_reso,
            "bucket_reso_steps": self.bucket_reso_steps,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DatasetSettings":
        return cls(
            name=str(data["name"]),
            slug=str(data["slug"]),
            dataset_type=str(data.get("dataset_type", "anima")),
            trigger_token=str(data["trigger_token"]),
            resolution_width=int(data.get("resolution_width", 1024)),
            resolution_height=int(data.get("resolution_height", 1024)),
            batch_size=int(data.get("batch_size", 1)),
            num_repeats=int(data.get("num_repeats", 10)),
            min_bucket_reso=int(data.get("min_bucket_reso", 512)),
            max_bucket_reso=int(data.get("max_bucket_reso", 1536)),
            bucket_reso_steps=int(data.get("bucket_reso_steps", 16)),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(slots=True)
class ImageRecord:
    stem: str
    image_path: Path
    caption_path: Path
    meta_path: Path
    control_path: Path | None = None
    caption: str = ""
    description: str = ""
    source_type: str = "uploaded"
    source_prompt: str = ""
    caption_format: str = ""

    @property
    def filename(self) -> str:
        return self.image_path.name

    @property
    def caption_filename(self) -> str:
        return self.caption_path.name

    @property
    def control_filename(self) -> str:
        return self.control_path.name if self.control_path is not None else ""

    @property
    def control_exists(self) -> bool:
        return self.control_path is not None and self.control_path.exists()


@dataclass(slots=True)
class CuratorCandidate:
    id: str
    job_id: str
    image_path: Path
    meta_path: Path
    raw_path: Path | None = None
    reference_paths: list[Path] = field(default_factory=list)
    instruction: str = ""
    approved: bool = False

    @property
    def filename(self) -> str:
        return self.image_path.name

    @property
    def raw_filename(self) -> str:
        return self.raw_path.name if self.raw_path is not None else ""

    @property
    def reference_filenames(self) -> list[str]:
        return [path.name for path in self.reference_paths]
