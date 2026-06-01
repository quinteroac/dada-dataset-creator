from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.models import (
    DATASET_TYPE_QWEN_IMAGE_EDIT_2511,
    CuratorCandidate,
    DatasetSettings,
    ImageRecord,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from app.toml_export import render_dataset_toml
from app.utils import is_supported_image, next_numbered_stem, normalize_caption, safe_trigger_token, slugify


class DatasetNotFoundError(Exception):
    pass


class DatasetStore:
    def __init__(self, root: Path = Path("datasets")) -> None:
        self.root = root

    def list_datasets(self) -> list[DatasetSettings]:
        if not self.root.exists():
            return []
        datasets = []
        for settings_path in sorted(self.root.glob("*/settings.json")):
            datasets.append(self.load_settings(settings_path.parent.name))
        return datasets

    def create_dataset(
        self,
        name: str,
        dataset_type: str,
        trigger_token: str,
        resolution_width: int = 1024,
        resolution_height: int = 1024,
        batch_size: int = 1,
        num_repeats: int = 10,
        min_bucket_reso: int = 512,
        max_bucket_reso: int = 1536,
        bucket_reso_steps: int = 16,
    ) -> DatasetSettings:
        slug = self._unique_slug(slugify(name))
        settings = DatasetSettings(
            name=name.strip() or slug,
            slug=slug,
            dataset_type=dataset_type,
            trigger_token=safe_trigger_token(trigger_token),
            resolution_width=resolution_width,
            resolution_height=resolution_height,
            batch_size=batch_size,
            num_repeats=num_repeats,
            min_bucket_reso=min_bucket_reso,
            max_bucket_reso=max_bucket_reso,
            bucket_reso_steps=bucket_reso_steps,
        )
        dataset_dir = self.dataset_dir(slug)
        (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "raw").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "curator").mkdir(parents=True, exist_ok=True)
        if dataset_type == DATASET_TYPE_QWEN_IMAGE_EDIT_2511:
            (dataset_dir / "controls").mkdir(parents=True, exist_ok=True)
            (dataset_dir / "cache").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "references").mkdir(parents=True, exist_ok=True)
        self.save_settings(settings)
        self.write_dataset_toml(settings)
        return settings

    def load_settings(self, slug: str) -> DatasetSettings:
        settings_path = self.dataset_dir(slug) / "settings.json"
        if not settings_path.exists():
            raise DatasetNotFoundError(slug)
        return DatasetSettings.from_json(json.loads(settings_path.read_text()))

    def save_settings(self, settings: DatasetSettings) -> None:
        dataset_dir = self.dataset_dir(settings.slug)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "settings.json").write_text(
            json.dumps(settings.to_json(), indent=2), encoding="utf-8"
        )
        self.write_dataset_toml(settings)

    def update_settings(self, slug: str, **updates: object) -> DatasetSettings:
        settings = self.load_settings(slug)
        data = settings.to_json()
        data.update({key: value for key, value in updates.items() if value is not None})
        data["trigger_token"] = safe_trigger_token(str(data["trigger_token"]))
        updated = DatasetSettings.from_json(data)
        self.save_settings(updated)
        self.rewrite_captions_for_trigger(updated)
        return updated

    async def save_uploaded_images(self, slug: str, files: list[UploadFile]) -> list[ImageRecord]:
        settings = self.load_settings(slug)
        saved = []
        for upload in files:
            if not upload.filename or not is_supported_image(upload.filename):
                continue
            suffix = Path(upload.filename).suffix.lower()
            stem = self._next_image_stem(slug)
            image_path = self.images_dir(slug) / f"{stem}{suffix}"
            content = await upload.read()
            image_path.write_bytes(content)
            record = self._write_caption_and_meta(
                settings,
                image_path=image_path,
                caption=settings.trigger_token,
                description="",
                source_type="uploaded",
                source_prompt="",
            )
            saved.append(record)
        return saved

    async def save_uploaded_edit_pairs(
        self,
        slug: str,
        control_files: list[UploadFile],
        target_files: list[UploadFile],
        instructions: str = "",
    ) -> list[ImageRecord]:
        settings = self.load_settings(slug)
        saved = []
        instruction_lines = [line.strip() for line in instructions.splitlines() if line.strip()]
        for index, (control_upload, target_upload) in enumerate(zip(control_files, target_files)):
            if (
                not control_upload.filename
                or not target_upload.filename
                or not is_supported_image(control_upload.filename)
                or not is_supported_image(target_upload.filename)
            ):
                continue
            stem = self._next_image_stem(slug)
            control_suffix = Path(control_upload.filename).suffix.lower()
            target_suffix = Path(target_upload.filename).suffix.lower()
            control_path = self.controls_dir(slug) / f"{stem}{control_suffix}"
            target_path = self.images_dir(slug) / f"{stem}{target_suffix}"
            control_path.write_bytes(await control_upload.read())
            target_path.write_bytes(await target_upload.read())
            instruction = instruction_lines[index] if index < len(instruction_lines) else settings.trigger_token
            saved.append(
                self._write_caption_and_meta(
                    settings,
                    target_path,
                    caption=instruction,
                    description=instruction,
                    source_type="uploaded_edit_pair",
                    source_prompt="",
                    control_path=control_path,
                )
            )
        return saved

    async def save_reference_images(self, slug: str, files: list[UploadFile]) -> list[Path]:
        self.load_settings(slug)
        saved = []
        references_dir = self.references_dir(slug)
        for upload in files:
            if not upload.filename or not is_supported_image(upload.filename):
                continue
            suffix = Path(upload.filename).suffix.lower()
            stem = self._next_reference_stem(slug)
            reference_path = references_dir / f"{stem}{suffix}"
            reference_path.write_bytes(await upload.read())
            saved.append(reference_path)
        return saved

    async def save_raw_images(self, slug: str, files: list[UploadFile]) -> list[Path]:
        self.load_settings(slug)
        saved = []
        raw_dir = self.raw_dir(slug)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for upload in files:
            if not upload.filename or not is_supported_image(upload.filename):
                continue
            suffix = Path(upload.filename).suffix.lower()
            stem = self._next_raw_stem(slug)
            raw_path = raw_dir / f"{stem}{suffix}"
            raw_path.write_bytes(await upload.read())
            saved.append(raw_path)
        return saved

    def list_raw_images(self, slug: str) -> list[Path]:
        self.load_settings(slug)
        raw_dir = self.raw_dir(slug)
        raw_dir.mkdir(parents=True, exist_ok=True)
        return [
            path
            for path in sorted(raw_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

    def delete_raw_image(self, slug: str, filename: str) -> None:
        raw_path = self.raw_image_path(slug, filename)
        if raw_path.exists():
            raw_path.unlink()

    def raw_image_path(self, slug: str, filename: str) -> Path:
        self.load_settings(slug)
        return self._supported_image_path_by_name(self.raw_dir(slug), filename)

    def reference_image_path(self, slug: str, filename: str) -> Path:
        self.load_settings(slug)
        return self._supported_image_path_by_name(self.references_dir(slug), filename)

    def import_generated_images(
        self,
        slug: str,
        generated_paths: list[Path],
        source_prompt: str,
    ) -> list[ImageRecord]:
        settings = self.load_settings(slug)
        imported = []
        for generated_path in generated_paths:
            if generated_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            stem = self._next_image_stem(slug)
            target = self.images_dir(slug) / f"{stem}{generated_path.suffix.lower()}"
            shutil.copy2(generated_path, target)
            imported.append(
                self._write_caption_and_meta(
                    settings,
                    target,
                    caption=settings.trigger_token,
                    description="",
                    source_type="generated",
                    source_prompt=source_prompt,
                )
            )
        return imported

    def import_edit_pairs(
        self,
        slug: str,
        pairs: list[tuple[Path, Path, str]],
        source_prompt: str,
    ) -> list[ImageRecord]:
        settings = self.load_settings(slug)
        imported = []
        for control_source, target_source, instruction in pairs:
            if (
                control_source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
                or target_source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
            ):
                continue
            stem = self._next_image_stem(slug)
            control_target = self.controls_dir(slug) / f"{stem}{control_source.suffix.lower()}"
            image_target = self.images_dir(slug) / f"{stem}{target_source.suffix.lower()}"
            shutil.copy2(control_source, control_target)
            shutil.copy2(target_source, image_target)
            imported.append(
                self._write_caption_and_meta(
                    settings,
                    image_target,
                    caption=instruction.strip() or settings.trigger_token,
                    description=instruction.strip(),
                    source_type="generated_edit_pair",
                    source_prompt=source_prompt,
                    control_path=control_target,
                )
            )
        return imported

    def stage_curator_candidates(
        self,
        slug: str,
        job_id: str,
        generated_paths: list[Path],
        raw_paths: list[Path],
        reference_paths: list[Path],
        instruction: str,
    ) -> list[CuratorCandidate]:
        self.load_settings(slug)
        candidates = []
        job_dir = self.curator_dir(slug) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        for index, generated_path in enumerate(generated_paths):
            if generated_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            stem = f"candidate_{len(candidates) + 1:06d}"
            target = job_dir / f"{stem}{generated_path.suffix.lower()}"
            shutil.copy2(generated_path, target)
            raw_path = raw_paths[index] if index < len(raw_paths) else (raw_paths[0] if raw_paths else None)
            meta_path = target.with_suffix(".meta.json")
            meta = {
                "job_id": job_id,
                "raw_filename": raw_path.name if raw_path is not None else "",
                "reference_filenames": [path.name for path in reference_paths],
                "instruction": instruction,
                "approved": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            candidates.append(self._read_curator_candidate(slug, target))
        return candidates

    def list_curator_candidates(self, slug: str) -> list[CuratorCandidate]:
        self.load_settings(slug)
        curator_dir = self.curator_dir(slug)
        curator_dir.mkdir(parents=True, exist_ok=True)
        candidates = [
            self._read_curator_candidate(slug, image_path)
            for image_path in sorted(curator_dir.glob("*/*"))
            if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        candidates.sort(key=lambda candidate: candidate.image_path.stat().st_mtime, reverse=True)
        return candidates

    def approve_curator_candidate(
        self,
        slug: str,
        candidate_id: str,
        instruction: str = "",
    ) -> ImageRecord:
        settings = self.load_settings(slug)
        candidate = self.curator_candidate(slug, candidate_id)
        if candidate.approved:
            raise ValueError(f"Candidate {candidate_id} is already approved")
        if settings.dataset_type == DATASET_TYPE_QWEN_IMAGE_EDIT_2511:
            caption = instruction.strip() or candidate.instruction.strip() or settings.trigger_token
        else:
            caption = instruction.strip() or settings.trigger_token
        stem = self._next_image_stem(slug)
        image_target = self.images_dir(slug) / f"{stem}{candidate.image_path.suffix.lower()}"
        shutil.copy2(candidate.image_path, image_target)
        control_path = None
        if settings.dataset_type == DATASET_TYPE_QWEN_IMAGE_EDIT_2511:
            if candidate.raw_path is None:
                raise FileNotFoundError(f"raw image for {candidate_id}")
            control_path = self.controls_dir(slug) / f"{stem}{candidate.raw_path.suffix.lower()}"
            shutil.copy2(candidate.raw_path, control_path)
        record = self._write_caption_and_meta(
            settings,
            image_target,
            caption=caption,
            description=caption if settings.dataset_type == DATASET_TYPE_QWEN_IMAGE_EDIT_2511 else "",
            source_type="curated",
            source_prompt=candidate.instruction,
            control_path=control_path,
        )
        meta = json.loads(candidate.meta_path.read_text(encoding="utf-8")) if candidate.meta_path.exists() else {}
        meta.update(
            {
                "approved": True,
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "approved_stem": record.stem,
                "approved_filename": record.filename,
            }
        )
        candidate.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return record

    def approve_curator_candidates(
        self,
        slug: str,
        candidate_ids: list[str],
        instruction: str = "",
    ) -> list[ImageRecord]:
        approved = []
        for candidate_id in candidate_ids:
            approved.append(self.approve_curator_candidate(slug, candidate_id, instruction))
        return approved

    def curator_candidate(self, slug: str, candidate_id: str) -> CuratorCandidate:
        self.load_settings(slug)
        try:
            job_id, stem = candidate_id.split("__", 1)
        except ValueError as exc:
            raise FileNotFoundError(candidate_id) from exc
        job_dir = self.curator_dir(slug) / job_id
        for suffix in SUPPORTED_IMAGE_EXTENSIONS:
            candidate_path = job_dir / f"{stem}{suffix}"
            if candidate_path.exists():
                return self._read_curator_candidate(slug, candidate_path)
        raise FileNotFoundError(candidate_id)

    def list_images(self, slug: str) -> list[ImageRecord]:
        settings = self.load_settings(slug)
        records = []
        for image_path in sorted(self.images_dir(slug).iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            records.append(self._read_image_record(settings, image_path))
        return records

    def list_references(self, slug: str) -> list[Path]:
        self.load_settings(slug)
        return [
            path
            for path in sorted(self.references_dir(slug).iterdir())
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

    def update_caption(
        self,
        slug: str,
        stem: str,
        caption: str,
        description: str,
    ) -> ImageRecord:
        settings = self.load_settings(slug)
        image_path = self._image_path_for_stem(slug, stem)
        return self._write_caption_and_meta(
            settings,
            image_path,
            caption=caption,
            description=description,
            source_type=self._read_meta(image_path).get("source_type", "uploaded"),
            source_prompt=str(self._read_meta(image_path).get("source_prompt", "")),
            control_path=self._control_path_for_stem(slug, stem),
        )

    def apply_label(
        self,
        slug: str,
        stem: str,
        tags: list[str],
        description: str,
    ) -> ImageRecord:
        settings = self.load_settings(slug)
        image_path = self._image_path_for_stem(slug, stem)
        caption = ", ".join(tags)
        meta = self._read_meta(image_path)
        return self._write_caption_and_meta(
            settings,
            image_path,
            caption=caption,
            description=description,
            source_type=str(meta.get("source_type", "uploaded")),
            source_prompt=str(meta.get("source_prompt", "")),
            control_path=self._control_path_for_stem(slug, stem),
        )

    def delete_image(self, slug: str, stem: str) -> None:
        self.load_settings(slug)
        image_path = self._image_path_for_stem(slug, stem)
        related_paths = [
            image_path,
            image_path.with_suffix(".txt"),
            image_path.with_suffix(".meta.json"),
        ]
        control_path = self._control_path_for_stem(slug, stem)
        if control_path is not None:
            related_paths.append(control_path)
        for path in related_paths:
            if path.exists():
                path.unlink()

    def rewrite_captions_for_trigger(self, settings: DatasetSettings) -> None:
        for record in self.list_images(settings.slug):
            self._write_caption_and_meta(
                settings,
                record.image_path,
                caption=record.caption,
                description=record.description,
                source_type=record.source_type,
                source_prompt=record.source_prompt,
            )

    def write_dataset_toml(self, settings: DatasetSettings) -> Path:
        target = self.dataset_dir(settings.slug) / "dataset.toml"
        target.write_text(
            render_dataset_toml(
                settings,
                self.images_dir(settings.slug),
                self.controls_dir(settings.slug),
                self.cache_dir(settings.slug),
            ),
            encoding="utf-8",
        )
        return target

    def dataset_dir(self, slug: str) -> Path:
        return self.root / slug

    def images_dir(self, slug: str) -> Path:
        return self.dataset_dir(slug) / "images"

    def references_dir(self, slug: str) -> Path:
        return self.dataset_dir(slug) / "references"

    def raw_dir(self, slug: str) -> Path:
        return self.dataset_dir(slug) / "raw"

    def curator_dir(self, slug: str) -> Path:
        return self.dataset_dir(slug) / "curator"

    def controls_dir(self, slug: str) -> Path:
        return self.dataset_dir(slug) / "controls"

    def cache_dir(self, slug: str) -> Path:
        return self.dataset_dir(slug) / "cache"

    def _unique_slug(self, base_slug: str) -> str:
        candidate = base_slug
        index = 2
        while self.dataset_dir(candidate).exists():
            candidate = f"{base_slug}_{index}"
            index += 1
        return candidate

    def _next_image_stem(self, slug: str) -> str:
        return next_numbered_stem(list(self.images_dir(slug).glob("*")))

    def _next_reference_stem(self, slug: str) -> str:
        existing = list(self.references_dir(slug).glob("ref_*"))
        max_seen = 0
        for path in existing:
            value = path.stem.replace("ref_", "")
            if value.isdigit():
                max_seen = max(max_seen, int(value))
        return f"ref_{max_seen + 1:03d}"

    def _next_raw_stem(self, slug: str) -> str:
        existing = list(self.raw_dir(slug).glob("raw_*"))
        max_seen = 0
        for path in existing:
            value = path.stem.replace("raw_", "")
            if value.isdigit():
                max_seen = max(max_seen, int(value))
        return f"raw_{max_seen + 1:06d}"

    def _write_caption_and_meta(
        self,
        settings: DatasetSettings,
        image_path: Path,
        caption: str,
        description: str,
        source_type: str,
        source_prompt: str,
        control_path: Path | None = None,
    ) -> ImageRecord:
        if settings.dataset_type == DATASET_TYPE_QWEN_IMAGE_EDIT_2511:
            normalized = caption.strip() or settings.trigger_token
        else:
            normalized = normalize_caption(settings.trigger_token, caption)
        caption_path = image_path.with_suffix(".txt")
        meta_path = image_path.with_suffix(".meta.json")
        caption_path.write_text(normalized + "\n", encoding="utf-8")
        meta = self._read_meta(image_path)
        meta.update(
            {
                "description": description,
                "tags": [part.strip() for part in normalized.split(",") if part.strip()],
                "trigger_token": settings.trigger_token,
                "source_prompt": source_prompt,
                "source_type": source_type,
                "control_filename": control_path.name if control_path is not None else "",
                "edit_instruction": normalized if settings.dataset_type == DATASET_TYPE_QWEN_IMAGE_EDIT_2511 else "",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return ImageRecord(
            stem=image_path.stem,
            image_path=image_path,
            caption_path=caption_path,
            meta_path=meta_path,
            control_path=control_path,
            caption=normalized,
            description=description,
            source_type=source_type,
            source_prompt=source_prompt,
        )

    def _read_image_record(self, settings: DatasetSettings, image_path: Path) -> ImageRecord:
        caption_path = image_path.with_suffix(".txt")
        meta = self._read_meta(image_path)
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else settings.trigger_token
        return ImageRecord(
            stem=image_path.stem,
            image_path=image_path,
            caption_path=caption_path,
            meta_path=image_path.with_suffix(".meta.json"),
            control_path=self._control_path_for_stem(settings.slug, image_path.stem),
            caption=caption if settings.dataset_type == DATASET_TYPE_QWEN_IMAGE_EDIT_2511 else normalize_caption(settings.trigger_token, caption),
            description=str(meta.get("description", "")),
            source_type=str(meta.get("source_type", "uploaded")),
            source_prompt=str(meta.get("source_prompt", "")),
        )

    def _read_meta(self, image_path: Path) -> dict[str, object]:
        meta_path = image_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _read_curator_candidate(self, slug: str, image_path: Path) -> CuratorCandidate:
        meta_path = image_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        raw_filename = str(meta.get("raw_filename", ""))
        reference_filenames = [str(value) for value in meta.get("reference_filenames", [])]
        raw_path = None
        if raw_filename:
            try:
                raw_path = self.raw_image_path(slug, raw_filename)
            except FileNotFoundError:
                raw_path = None
        reference_paths = []
        for filename in reference_filenames:
            try:
                reference_paths.append(self.reference_image_path(slug, filename))
            except FileNotFoundError:
                continue
        return CuratorCandidate(
            id=f"{image_path.parent.name}__{image_path.stem}",
            job_id=image_path.parent.name,
            image_path=image_path,
            meta_path=meta_path,
            raw_path=raw_path,
            reference_paths=reference_paths,
            instruction=str(meta.get("instruction", "")),
            approved=bool(meta.get("approved", False)),
        )

    def _supported_image_path_by_name(self, directory: Path, filename: str) -> Path:
        safe_name = Path(filename).name
        candidate = directory / safe_name
        if candidate.exists() and candidate.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            return candidate
        raise FileNotFoundError(filename)

    def _image_path_for_stem(self, slug: str, stem: str) -> Path:
        for suffix in SUPPORTED_IMAGE_EXTENSIONS:
            candidate = self.images_dir(slug) / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(stem)

    def _control_path_for_stem(self, slug: str, stem: str) -> Path | None:
        controls_dir = self.controls_dir(slug)
        if not controls_dir.exists():
            return None
        for suffix in SUPPORTED_IMAGE_EXTENSIONS:
            candidate = controls_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None
