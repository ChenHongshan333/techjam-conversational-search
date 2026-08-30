from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .evaluation_service import EvaluationService
from .service import DashboardService


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "static"
BASELINE_PATH = Path("docs/baseline_results.json")


class DashboardApplication:
    def __init__(self) -> None:
        self.dashboard = DashboardService()
        self.evaluations = EvaluationService()
        self.baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    def metrics(self) -> dict:
        return {"baseline": self.baseline, "current": self.evaluations.latest()}


def make_handler(application: DashboardApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        app = application

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._get()
            except KeyError as error:
                self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
            except ValueError as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:  # pragma: no cover - defensive server boundary
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            try:
                self._post()
            except KeyError as error:
                self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except RuntimeError as error:
                self._json({"error": str(error)}, HTTPStatus.CONFLICT)
            except Exception as error:  # pragma: no cover - defensive server boundary
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def _get(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            if path == "/api/health":
                self._json({"status": "ok"})
                return
            if path == "/api/metrics":
                self._json(self.app.metrics())
                return
            if path == "/api/test-cases":
                rows = self.app.dashboard.list_test_cases(
                    scenario=(query.get("scenario") or [None])[0],
                    difficulty=(query.get("difficulty") or [None])[0],
                    query=(query.get("q") or [None])[0],
                )
                self._json({"test_cases": rows, "count": len(rows)})
                return
            if path.startswith("/api/sessions/"):
                session_id = path.removeprefix("/api/sessions/")
                self._json(self.app.dashboard.get_session(session_id).snapshot())
                return
            if path.startswith("/api/evaluations/"):
                evaluation_id = path.removeprefix("/api/evaluations/")
                self._json(self.app.evaluations.get(evaluation_id))
                return
            self._static(path)

        def _post(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/sessions":
                body = self._body()
                sample_id = str(body.get("sample_id") or "").strip()
                if not sample_id:
                    raise ValueError("sample_id is required")
                self._json(self.app.dashboard.create_session(sample_id).snapshot(), HTTPStatus.CREATED)
                return
            if path.endswith("/step") and path.startswith("/api/sessions/"):
                session_id = path.removeprefix("/api/sessions/").removesuffix("/step")
                event = self.app.dashboard.get_session(session_id).step()
                self._json(event)
                return
            if path.endswith("/run") and path.startswith("/api/sessions/"):
                session_id = path.removeprefix("/api/sessions/").removesuffix("/run")
                session = self.app.dashboard.get_session(session_id)
                self._json({"events": session.run_all(), "snapshot": session.snapshot()})
                return
            if path == "/api/evaluations":
                self._json(self.app.evaluations.start(), HTTPStatus.ACCEPTED)
                return
            raise KeyError(f"Unknown endpoint: {path}")

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

        def _static(self, path: str) -> None:
            relative = "index.html" if path == "/" else path.lstrip("/")
            target = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT.resolve() not in target.parents and target != STATIC_ROOT.resolve():
                raise KeyError("Invalid static path")
            if not target.is_file():
                raise KeyError(f"Static file not found: {relative}")
            payload = target.read_bytes()
            mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local TechJam agent dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    application = DashboardApplication()
    server = HTTPServer((args.host, args.port), make_handler(application))
    print(f"Dashboard ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
