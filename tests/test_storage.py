from pathlib import Path

from app.storage import DatasetStore


def test_create_dataset_writes_anima_toml(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path)
    settings = store.create_dataset("My Dataset", "anima", "my_token")

    toml = (tmp_path / settings.slug / "dataset.toml").read_text(encoding="utf-8")

    assert 'caption_extension = ".txt"' in toml
    assert "resolution = [1024, 1024]" in toml
    assert f'image_dir = "{tmp_path / settings.slug / "images"}"' in toml
    assert 'class_tokens = "my_token"' in toml


def test_import_generated_images_uses_sequential_names_and_metadata(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Generated", "anima", "gen_token")
    source = tmp_path / "source.png"
    source.write_bytes(b"fake image")

    records = store.import_generated_images(settings.slug, [source], "blue hair prompt")

    assert records[0].filename == "000001.png"
    assert records[0].caption == "gen_token"
    assert (tmp_path / "datasets" / settings.slug / "images" / "000001.txt").read_text().strip() == "gen_token"
    meta = (tmp_path / "datasets" / settings.slug / "images" / "000001.meta.json").read_text()
    assert '"source_type": "generated"' in meta
    assert '"source_prompt": "blue hair prompt"' in meta


def test_update_caption_always_adds_trigger_token(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Manual", "anima", "manual_token")
    source = tmp_path / "source.webp"
    source.write_bytes(b"fake image")
    store.import_generated_images(settings.slug, [source], "")

    record = store.update_caption(settings.slug, "000001", "1girl, white dress", "standing")

    assert record.caption == "manual_token, 1girl, white dress"
    assert record.description == "standing"


def test_delete_image_removes_image_caption_and_metadata(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Delete", "anima", "delete_token")
    source = tmp_path / "source.png"
    source.write_bytes(b"fake image")
    store.import_generated_images(settings.slug, [source], "")

    store.delete_image(settings.slug, "000001")

    images_dir = tmp_path / "datasets" / settings.slug / "images"
    assert not (images_dir / "000001.png").exists()
    assert not (images_dir / "000001.txt").exists()
    assert not (images_dir / "000001.meta.json").exists()


def test_qwen_edit_dataset_writes_musubi_toml_and_imports_pairs(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Qwen Edit", "qwen_image_edit_2511", "Change the image")
    control = tmp_path / "control.png"
    target = tmp_path / "target.png"
    control.write_bytes(b"control")
    target.write_bytes(b"target")

    records = store.import_edit_pairs(settings.slug, [(control, target, "Make the jacket red.")], "red jacket prompt")

    dataset_dir = tmp_path / "datasets" / settings.slug
    assert (dataset_dir / "controls" / "000001.png").exists()
    assert (dataset_dir / "images" / "000001.png").exists()
    assert (dataset_dir / "images" / "000001.txt").read_text().strip() == "Make the jacket red."
    assert records[0].control_filename == "000001.png"
    toml = (dataset_dir / "dataset.toml").read_text(encoding="utf-8")
    assert 'image_directory = "' in toml
    assert 'control_directory = "' in toml
    assert 'cache_directory = "' in toml
    assert "control_resolution = [1024, 1024]" in toml

    store.delete_image(settings.slug, "000001")

    assert not (dataset_dir / "controls" / "000001.png").exists()


def test_curator_raw_references_and_anima_approval(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Curator", "anima", "cur_token")
    raw = store.raw_dir(settings.slug) / "raw_000001.png"
    reference = store.references_dir(settings.slug) / "ref_001.png"
    generated = tmp_path / "curated.png"
    raw.write_bytes(b"raw")
    reference.write_bytes(b"ref")
    generated.write_bytes(b"curated")

    candidates = store.stage_curator_candidates(
        settings.slug,
        "job1",
        [generated],
        [raw],
        [reference],
        "Use the reference colors.",
    )
    record = store.approve_curator_candidate(settings.slug, candidates[0].id, "")

    dataset_dir = tmp_path / "datasets" / settings.slug
    assert (dataset_dir / "raw").exists()
    assert (dataset_dir / "curator" / "job1" / "candidate_000001.png").exists()
    assert record.filename == "000001.png"
    assert (dataset_dir / "images" / "000001.txt").read_text().strip() == "cur_token"
    assert '"source_type": "curated"' in (dataset_dir / "images" / "000001.meta.json").read_text()


def test_curator_qwen_approval_imports_raw_as_control(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path / "datasets")
    settings = store.create_dataset("Qwen Curator", "qwen_image_edit_2511", "Edit the image")
    raw = store.raw_dir(settings.slug) / "raw_000001.png"
    generated = tmp_path / "curated.png"
    raw.write_bytes(b"raw")
    generated.write_bytes(b"curated")

    candidate = store.stage_curator_candidates(
        settings.slug,
        "job2",
        [generated],
        [raw],
        [],
        "Make this look like flash photography.",
    )[0]
    record = store.approve_curator_candidate(settings.slug, candidate.id, "")

    dataset_dir = tmp_path / "datasets" / settings.slug
    assert (dataset_dir / "controls" / "000001.png").read_bytes() == b"raw"
    assert (dataset_dir / "images" / "000001.png").read_bytes() == b"curated"
    assert (dataset_dir / "images" / "000001.txt").read_text().strip() == "Make this look like flash photography."
    assert record.control_filename == "000001.png"
