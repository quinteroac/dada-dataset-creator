from __future__ import annotations

from pathlib import Path

from app.models import DATASET_TYPE_QWEN_IMAGE_EDIT_2511, DatasetSettings


def render_dataset_toml(
    settings: DatasetSettings,
    image_dir: Path | None = None,
    control_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> str:
    rendered_image_dir = str(image_dir.resolve()) if image_dir is not None else "./images"
    if settings.dataset_type == DATASET_TYPE_QWEN_IMAGE_EDIT_2511:
        rendered_control_dir = str(control_dir.resolve()) if control_dir is not None else "./controls"
        rendered_cache_dir = str(cache_dir.resolve()) if cache_dir is not None else "./cache"
        return "\n".join(
            [
                "[general]",
                f"resolution = [{settings.resolution_width}, {settings.resolution_height}]",
                'caption_extension = ".txt"',
                f"batch_size = {settings.batch_size}",
                "enable_bucket = true",
                "bucket_no_upscale = false",
                "",
                "[[datasets]]",
                f'image_directory = "{rendered_image_dir}"',
                f'control_directory = "{rendered_control_dir}"',
                f'cache_directory = "{rendered_cache_dir}"',
                f"num_repeats = {settings.num_repeats}",
                "control_resolution = [1024, 1024]",
                "no_resize_control = false",
                "",
            ]
        )

    return "\n".join(
        [
            "[general]",
            'caption_extension = ".txt"',
            "shuffle_caption = true",
            "keep_tokens = 1",
            "",
            "[[datasets]]",
            f"resolution = [{settings.resolution_width}, {settings.resolution_height}]",
            f"batch_size = {settings.batch_size}",
            "enable_bucket = true",
            f"bucket_reso_steps = {settings.bucket_reso_steps}",
            f"min_bucket_reso = {settings.min_bucket_reso}",
            f"max_bucket_reso = {settings.max_bucket_reso}",
            "",
            "  [[datasets.subsets]]",
            f'  image_dir = "{rendered_image_dir}"',
            f"  num_repeats = {settings.num_repeats}",
            f'  class_tokens = "{settings.trigger_token}"',
            "",
        ]
    )
