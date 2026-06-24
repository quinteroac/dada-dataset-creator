from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


JobStatus = Literal["queued", "running", "success", "error"]
JobType = Literal[
    "generate_images",
    "generate_edit_pairs",
    "label_image",
    "label_batch",
    "import_recent",
    "curate_raw_images",
    "setup_sd_scripts",
    "setup_musubi_tuner",
    "setup_ai_toolkit",
    "train_anima_lora",
    "train_qwen_edit_lora",
    "train_ideogram4_lora",
    "cancel_training",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CodexJob:
    id: str
    dataset_slug: str
    type: JobType
    status: JobStatus
    payload: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    message: str = ""
    error: str = ""
    generated_count: int = 0
    imported_count: int = 0
    output_path: str = ""
    command: list[str] = field(default_factory=list)
    return_code: int | None = None
    process_pid: int | None = None
    output_tail: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_slug": self.dataset_slug,
            "type": self.type,
            "status": self.status,
            "payload": self.payload,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "message": self.message,
            "error": self.error,
            "generated_count": self.generated_count,
            "imported_count": self.imported_count,
            "output_path": self.output_path,
            "command": self.command,
            "return_code": self.return_code,
            "process_pid": self.process_pid,
            "output_tail": self.output_tail,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CodexJob":
        return cls(
            id=str(data["id"]),
            dataset_slug=str(data["dataset_slug"]),
            type=data["type"],
            status=data["status"],
            payload=dict(data.get("payload", {})),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            message=str(data.get("message", "")),
            error=str(data.get("error", "")),
            generated_count=int(data.get("generated_count", 0)),
            imported_count=int(data.get("imported_count", 0)),
            output_path=str(data.get("output_path", "")),
            command=[str(part) for part in data.get("command", [])],
            return_code=data.get("return_code"),
            process_pid=data.get("process_pid"),
            output_tail=[str(line) for line in data.get("output_tail", [])],
        )


class JobStore:
    def __init__(self, datasets_root: Path = Path("datasets")) -> None:
        self.datasets_root = datasets_root
        self._lock = threading.Lock()

    def create_job(self, dataset_slug: str, job_type: JobType, payload: dict[str, Any]) -> CodexJob:
        now = utc_now()
        job = CodexJob(
            id=uuid.uuid4().hex,
            dataset_slug=dataset_slug,
            type=job_type,
            status="queued",
            payload=payload,
            created_at=now,
            updated_at=now,
            message="Job queued",
        )
        self.save_job(job)
        return job

    def list_jobs(self, dataset_slug: str, limit: int = 25) -> list[CodexJob]:
        jobs = list(self._read_state(dataset_slug).values())
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    def get_job(self, dataset_slug: str, job_id: str) -> CodexJob:
        state = self._read_state(dataset_slug)
        if job_id not in state:
            raise KeyError(job_id)
        return state[job_id]

    def save_job(self, job: CodexJob) -> None:
        with self._lock:
            job.updated_at = utc_now()
            state = self._read_state_unlocked(job.dataset_slug)
            state[job.id] = job
            self._write_state_unlocked(job.dataset_slug, state)
            self._append_history_unlocked(job)

    def mark_running(self, dataset_slug: str, job_id: str) -> CodexJob:
        job = self.get_job(dataset_slug, job_id)
        job.status = "running"
        job.started_at = utc_now()
        job.message = "Job running"
        self.save_job(job)
        return job

    def update_job(self, dataset_slug: str, job_id: str, **updates: Any) -> CodexJob:
        job = self.get_job(dataset_slug, job_id)
        for key, value in updates.items():
            if hasattr(job, key):
                setattr(job, key, value)
        self.save_job(job)
        return job

    def mark_success(
        self,
        dataset_slug: str,
        job_id: str,
        message: str,
        generated_count: int = 0,
        imported_count: int = 0,
        output_path: str = "",
        return_code: int | None = None,
    ) -> CodexJob:
        job = self.get_job(dataset_slug, job_id)
        job.status = "success"
        job.finished_at = utc_now()
        job.message = message
        job.error = ""
        job.generated_count = generated_count
        job.imported_count = imported_count
        job.output_path = output_path
        job.return_code = return_code
        job.process_pid = None
        self.save_job(job)
        return job

    def mark_error(self, dataset_slug: str, job_id: str, error: str) -> CodexJob:
        job = self.get_job(dataset_slug, job_id)
        job.status = "error"
        job.finished_at = utc_now()
        job.message = "Job failed"
        job.error = error
        job.process_pid = None
        self.save_job(job)
        return job

    def append_log(self, dataset_slug: str, job_id: str, line: str) -> None:
        clean_line = line.rstrip()
        if not clean_line:
            return
        with self._lock:
            state = self._read_state_unlocked(dataset_slug)
            job = state[job_id]
            job.output_tail = (job.output_tail + [clean_line])[-80:]
            job.updated_at = utc_now()
            self._log_path(dataset_slug, job_id).parent.mkdir(parents=True, exist_ok=True)
            with self._log_path(dataset_slug, job_id).open("a", encoding="utf-8") as file:
                file.write(clean_line + "\n")
            self._write_state_unlocked(dataset_slug, state)
            self._append_history_unlocked(job)

    def read_log(self, dataset_slug: str, job_id: str) -> str:
        log_path = self._log_path(dataset_slug, job_id)
        if not log_path.exists():
            return ""
        return log_path.read_text(encoding="utf-8")

    def recover_interrupted_jobs(self) -> list[CodexJob]:
        recovered = []
        for state_path in self.datasets_root.glob("*/jobs_state.json"):
            dataset_slug = state_path.parent.name
            state = self._read_state(dataset_slug)
            changed = False
            for job in state.values():
                if job.status == "running":
                    job.status = "error"
                    job.finished_at = utc_now()
                    job.message = "Job interrupted"
                    job.error = "interrupted by server restart"
                    changed = True
                    recovered.append(job)
            if changed:
                with self._lock:
                    self._write_state_unlocked(dataset_slug, state)
                    for job in recovered:
                        if job.dataset_slug == dataset_slug:
                            self._append_history_unlocked(job)
        return recovered

    def queued_jobs(self, job_types: set[str] | None = None) -> list[CodexJob]:
        jobs = []
        for state_path in self.datasets_root.glob("*/jobs_state.json"):
            jobs.extend(
                job
                for job in self._read_state(state_path.parent.name).values()
                if job.status == "queued" and (job_types is None or job.type in job_types)
            )
        jobs.sort(key=lambda job: job.created_at)
        return jobs

    def _read_state(self, dataset_slug: str) -> dict[str, CodexJob]:
        with self._lock:
            return self._read_state_unlocked(dataset_slug)

    def _read_state_unlocked(self, dataset_slug: str) -> dict[str, CodexJob]:
        state_path = self._state_path(dataset_slug)
        if not state_path.exists():
            return {}
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return {job_id: CodexJob.from_json(job_data) for job_id, job_data in data.items()}

    def _write_state_unlocked(self, dataset_slug: str, state: dict[str, CodexJob]) -> None:
        state_path = self._state_path(dataset_slug)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({job_id: job.to_json() for job_id, job in state.items()}, indent=2),
            encoding="utf-8",
        )

    def _append_history_unlocked(self, job: CodexJob) -> None:
        history_path = self._history_path(job.dataset_slug)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(job.to_json()) + "\n")

    def _dataset_dir(self, dataset_slug: str) -> Path:
        return self.datasets_root / dataset_slug

    def _state_path(self, dataset_slug: str) -> Path:
        return self._dataset_dir(dataset_slug) / "jobs_state.json"

    def _history_path(self, dataset_slug: str) -> Path:
        return self._dataset_dir(dataset_slug) / "jobs.jsonl"

    def _log_path(self, dataset_slug: str, job_id: str) -> Path:
        return self._dataset_dir(dataset_slug) / "jobs" / f"{job_id}.log"
