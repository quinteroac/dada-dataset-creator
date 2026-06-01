from pathlib import Path

from app.jobs import JobStore


def test_job_store_creates_updates_and_persists_logs(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    (root / "blue").mkdir(parents=True)
    store = JobStore(root)

    job = store.create_job("blue", "generate_images", {"prompt": "blue dawn"})
    store.mark_running("blue", job.id)
    store.append_log("blue", job.id, "Codex says hello")
    store.mark_success("blue", job.id, "done", generated_count=2, imported_count=2)

    loaded = store.get_job("blue", job.id)
    assert loaded.status == "success"
    assert loaded.output_tail[-1] == "Codex says hello"
    assert loaded.imported_count == 2
    assert "Codex says hello" in store.read_log("blue", job.id)
    assert (root / "blue" / "jobs.jsonl").exists()


def test_job_store_recovers_interrupted_running_jobs(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    (root / "blue").mkdir(parents=True)
    store = JobStore(root)
    job = store.create_job("blue", "label_image", {"stem": "000001"})
    store.mark_running("blue", job.id)

    recovered = store.recover_interrupted_jobs()

    assert recovered[0].id == job.id
    assert store.get_job("blue", job.id).status == "error"
    assert "server restart" in store.get_job("blue", job.id).error
