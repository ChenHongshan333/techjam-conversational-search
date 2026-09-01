"""Stateless Vercel API for the optional Seekly demonstration site.

This adapter imports the existing dashboard and Agent as read-only dependencies.
It does not change the competition entry point or retrieval implementation.
"""

from __future__ import annotations

import json
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dashboard.service import DashboardService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.jsonl"
DATASET_PATH = PROJECT_ROOT / "data" / "public_set.jsonl"
BASELINE_PATH = PROJECT_ROOT / "docs" / "baseline_results.json"
VALIDATED_RESULTS_PATH = PROJECT_ROOT / "vercel_app" / "validated_results.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dashboard() -> DashboardService:
    """Build request-local state so SQLite never crosses serverless threads."""
    return DashboardService(CATALOG_PATH, DATASET_PATH)


@lru_cache(maxsize=1)
def _test_case_rows() -> tuple[dict, ...]:
    """Load the public case selector without constructing the SQLite index."""
    samples: list[dict] = []
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(json.loads(line))

    target_ids = {
        str(sample["ground_truth"]["parent_asin"])
        for sample in samples
    }
    titles: dict[str, str] = {}
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            parent_asin = str(product["parent_asin"])
            if parent_asin in target_ids:
                titles[parent_asin] = str(product.get("title") or parent_asin)

    return tuple({
        "sample_id": sample["sample_id"],
        "scenario_type": sample["scenario_type"],
        "difficulty_bucket": sample.get("difficulty_bucket"),
        "category_bucket": sample.get("category_bucket"),
        "target_title": titles[str(sample["ground_truth"]["parent_asin"])],
    } for sample in samples)


def _list_test_cases(
    scenario: str | None,
    difficulty: str | None,
    query: str | None,
) -> list[dict]:
    query_value = (query or "").casefold().strip()
    return [
        row
        for row in _test_case_rows()
        if (not scenario or row["scenario_type"] == scenario)
        and (not difficulty or row["difficulty_bucket"] == difficulty)
        and (
            not query_value
            or query_value in f"{row['sample_id']} {row['target_title']}".casefold()
        )
    ]


def _replay_payload(sample_id: str) -> dict:
    """Calculate a complete replay in one request, then discard server state."""
    dashboard = _dashboard()
    session = dashboard.create_session(sample_id)
    try:
        initial = session.snapshot()
        events = session.run_all()
        final = session.snapshot()
        return {"initial": initial, "events": events, "snapshot": final}
    finally:
        dashboard.sessions.pop(session.session_id, None)


class handler(BaseHTTPRequestHandler):
    """Vercel Python function entry point."""

    def do_GET(self) -> None:  # noqa: N802
        try:
            path, query = self._route()
            if path == "/api/health":
                self._json({"status": "ok", "mode": "vercel-stateless"})
                return
            if path == "/api/metrics":
                self._json({
                    "baseline": _load_json(BASELINE_PATH),
                    "current": _load_json(VALIDATED_RESULTS_PATH),
                    "evaluation_enabled": False,
                })
                return
            if path == "/api/test-cases":
                rows = _list_test_cases(
                    (query.get("scenario") or [None])[0],
                    (query.get("difficulty") or [None])[0],
                    (query.get("q") or [None])[0],
                )
                self._json({"test_cases": rows, "count": len(rows)})
                return
            raise KeyError(f"Unknown endpoint: {path}")
        except KeyError as error:
            self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except ValueError as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - serverless boundary
            self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        try:
            path, _ = self._route()
            if path == "/api/replay":
                sample_id = str(self._body().get("sample_id") or "").strip()
                if not sample_id:
                    raise ValueError("sample_id is required")
                self._json(_replay_payload(sample_id), HTTPStatus.CREATED)
                return
            if path == "/api/evaluations":
                self._json(
                    {"error": "Live evaluation is local-only; validated metrics are shown."},
                    HTTPStatus.CONFLICT,
                )
                return
            raise KeyError(f"Unknown endpoint: {path}")
        except KeyError as error:
            self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - serverless boundary
            self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _route(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        rewritten = (query.get("route") or [None])[0]
        path = f"/api/{rewritten.strip('/')}" if rewritten else parsed.path
        return path, query

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return
