from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def summarize(result: dict | None) -> dict | None:
    if result is None:
        return None
    return {key: value for key, value in result.items() if key != "sessions"}


class EvaluationService:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        dataset_path: str | Path = "data/public_set.jsonl",
        artifact_path: str | Path = "artifacts/latest_evaluation.json",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.dataset_path = Path(dataset_path)
        self.artifact_path = Path(artifact_path)
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def latest(self) -> dict | None:
        if not self.artifact_path.exists():
            return None
        return summarize(json.loads(self.artifact_path.read_text(encoding="utf-8")))

    def start(self) -> dict:
        with self.lock:
            active = next(
                (job for job in self.jobs.values() if job["status"] in {"queued", "running"}),
                None,
            )
            if active:
                return dict(active)
            job_id = uuid.uuid4().hex
            job = {"evaluation_id": job_id, "status": "queued", "result": None, "error": None}
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return dict(job)

    def get(self, job_id: str) -> dict:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown evaluation: {job_id}")
            return dict(job)

    def _run(self, job_id: str) -> None:
        with self.lock:
            self.jobs[job_id]["status"] = "running"
        try:
            samples = load_jsonl(self.dataset_path)
            catalog_ids, categories, products = catalog_index(self.catalog_path)
            result = evaluate(
                Agent(self.catalog_path),
                samples,
                catalog_ids,
                categories,
                products,
            )
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self.artifact_path.write_text(
                json.dumps(result, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.lock:
                self.jobs[job_id].update(status="completed", result=summarize(result))
        except Exception as error:  # pragma: no cover - surfaced through the API
            with self.lock:
                self.jobs[job_id].update(status="failed", error=str(error))
