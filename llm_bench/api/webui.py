"""Loopback-only HTTP adapter for the API workbench. Run with python -m llm_bench.api.webui."""
from __future__ import annotations

import argparse
import json
import secrets
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from llm_bench.api.config import BenchmarkConfig
from llm_bench.api.service import Workbench


ROOT = Path(__file__).resolve().parents[2]


class WorkbenchServer(ThreadingHTTPServer):
    def __init__(self, port: int, workbench: Workbench, assets: Path):
        self.workbench = workbench
        self.assets = assets.resolve()
        self.session_token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), partial(Handler, directory=str(self.assets)))
        self.origin = f"http://127.0.0.1:{self.server_port}"


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # URLs, request bodies, and credentials are never written to access logs.
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        super().end_headers()

    def _trusted(self, token: bool = True) -> bool:
        expected_host = urlsplit(self.server.origin).netloc
        if (
            self.headers.get("Host") != expected_host
            or self.headers.get("Origin", self.server.origin) != self.server.origin
            or self.headers.get("Sec-Fetch-Site") == "cross-site"
            or (token and not secrets.compare_digest(
                self.headers.get("X-Session-Token", ""), self.server.session_token
            ))
        ):
            self._json({"error": "Local session authorization required"}, 403)
            return False
        return True

    def _json(self, value, status: int = 200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if not self._trusted(token=path.startswith("/api/") and path != "/api/bootstrap"):
            return
        try:
            if path == "/api/bootstrap":
                self._json({**self.server.workbench.bootstrap(), "token": self.server.session_token})
            elif path == "/api/job":
                self._json(self.server.workbench.status())
            elif path == "/api/history":
                self._json(self.server.workbench.history())
            elif path.startswith("/api/results/"):
                self._json(self.server.workbench.result(unquote(path.removeprefix("/api/results/"))))
            elif path.startswith("/api/"):
                self._json({"error": "Unknown API route"}, 404)
            else:
                target = Path(self.translate_path(path)).resolve()
                if not target.is_relative_to(self.server.assets) or not target.is_file():
                    self._json({"error": "Not found"}, 404)
                    return
                super().do_GET()
        except FileNotFoundError:
            self._json({"error": "Result not found"}, 404)
        except (ValueError, OSError) as exc:
            self._json({"error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}, 400)

    def do_HEAD(self):
        self._json({"error": "Method not supported"}, 405)

    def do_POST(self):
        if not self._trusted():
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536 or self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON body of at most 64 KiB")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            path = urlsplit(self.path).path
            workbench = self.server.workbench
            if path == "/api/preview":
                config = BenchmarkConfig.from_dict(body.get("config"))
                self._json(config.plan())
            elif path == "/api/run":
                self._json(workbench.start(body.get("config"), body.get("apiKey"), body.get("useSavedKey", False)), 202)
            elif path == "/api/cancel":
                self._json(workbench.cancel())
            elif path == "/api/profiles":
                self._json(workbench.save_profile(body.get("name"), body.get("config")))
            else:
                self._json({"error": "Unknown API route"}, 404)
        except (ValueError, OSError) as exc:
            self._json({"error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the local LLM API benchmark workbench.")
    parser.add_argument("--port", type=int, default=0, help="Loopback port (default: choose an available port)")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "docs-src" / "dist-workbench")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "llm-api-bench")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "outputs" / "llm-api-ui")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if not (args.assets_dir / "workbench.html").is_file():
        parser.error("UI assets missing. Build once with: cd docs-src && npm run build:workbench")
    server = WorkbenchServer(args.port, Workbench(args.output_dir, args.state_dir, args.env_file), args.assets_dir)
    url = server.origin + "/workbench.html"
    print(f"LLM API workbench: {url}", flush=True)
    print("Press Ctrl+C to stop. Running requests finish before exit.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.workbench.cancel()
        server.server_close()
        if server.workbench.worker is not None:
            server.workbench.worker.join()


if __name__ == "__main__":
    main()
