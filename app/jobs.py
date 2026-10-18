"""Tiny in-process background job registry.

Long operations (summarize a large document, embed a folder) return a job id immediately 
and are polled for status, instead of blocking the caller.

Deliberately minimal: threads + a dict. No persistence, no external queue. If the
workload grows, swap this module for a real queue without touching capabilities.
"""

import threading
import time
import uuid

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def submit(name: str, fn, *args, **kwargs) -> str:
    """Run ``fn(*args, **kwargs)`` in a thread; return the job id at once."""
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "name": name,
            "status": "queued",
            "created_at": time.time(),
        }

    def run() -> None:
        with _lock:
            _jobs[job_id]["status"] = "running"
            _jobs[job_id]["started_at"] = time.time()
        try:
            result = fn(*args, **kwargs)
            with _lock:
                _jobs[job_id].update(status="done", result=result)
        except Exception as error:  # keep the failure visible to the caller
            with _lock:
                _jobs[job_id].update(status="error", error=str(error))
        finally:
            with _lock:
                _jobs[job_id]["finished_at"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job_id


def get(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs() -> list[dict]:
    with _lock:
        return [dict(j) for j in _jobs.values()]