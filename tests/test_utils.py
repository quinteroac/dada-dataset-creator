from app.utils import normalize_caption, slugify


def test_slugify_human_dataset_name() -> None:
    assert slugify("  Mi Dataset Ánima!!  ") == "mi_dataset_nima"
    assert slugify("!!!") == "dataset"


def test_normalize_caption_keeps_trigger_first() -> None:
    caption = normalize_caption("my_token", "1girl, my_token, blue hair")
    assert caption == "my_token, 1girl, blue hair"
