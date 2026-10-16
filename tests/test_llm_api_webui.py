import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, request

from llm_bench.api.config import BenchmarkConfig
from llm_bench.api.data import dashboard_entry
from llm_bench.api.service import Workbench
from llm_bench.api.webui import WorkbenchServer


class FakeProvider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.server.calls += 1
        if getattr(self.server, "reject", False):
            self.send_error(401)
            return
        self.server.started.set()
        self.server.release.wait(5)
        events = [
            {"choices": [{"delta": {"content": "Hello "}}]},
            {"choices": [{"delta": {"content": "world"}}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 3}},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.001)
        self.wfile.write(b"data: [DONE]\n\n")


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        assets = self.root / "assets"
        assets.mkdir()
        (assets / "workbench.html").write_text("<h1>Workbench</h1>")
        self.provider = ThreadingHTTPServer(("127.0.0.1", 0), FakeProvider)
        self.provider.calls = 0
        self.provider.started = threading.Event()
        self.provider.release = threading.Event()
        self.provider.release.set()
        self.workbench = Workbench(self.root / "results", self.root / "state", self.root / ".env")
        self.server = WorkbenchServer(0, self.workbench, assets)
        for server in (self.provider, self.server):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        self.token = self.call("/api/bootstrap")["token"]
        self.config = {
            "provider": "compatible", "model": "fixture-model",
            "base_url": f"http://127.0.0.1:{self.provider.server_port}/v1",
        }

    def tearDown(self):
        self.workbench.cancel()
        self.provider.release.set()
        if self.workbench.worker:
            self.workbench.worker.join(5)
        for server in (self.server, self.provider):
            server.shutdown()
            server.server_close()
        self.temporary.cleanup()

    def call(self, path, body=None, headers=None):
        headers = {"X-Session-Token": getattr(self, "token", ""), **(headers or {})}
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = request.Request(self.server.origin + path, headers=headers,
                              data=json.dumps(body).encode() if body is not None else None)
        try:
            with request.urlopen(req, timeout=5) as response:
                return json.load(response)
        except error.HTTPError as exc:
            exc.close()
            raise

    def finish(self):
        self.workbench.worker.join(5)
        self.assertFalse(self.workbench.worker.is_alive())
        return self.call("/api/job")

    def test_ui_to_provider_to_payload_and_history(self):
        plan = self.call("/api/preview", {"config": self.config})
        self.assertEqual(plan["requests"], 23)
        self.assertEqual(self.provider.calls, 0)
        self.call("/api/run", {"config": self.config, "apiKey": "test-session-key"})
        job = self.finish()
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["completed"], 23)
        self.assertEqual(self.provider.calls, 23)
        history = self.call("/api/history")
        self.assertEqual(len(history), 1)
        result = self.call("/api/results/" + history[0]["file"])
        payload = result["payload"]
        self.assertEqual(payload["summary"]["successes"], 21)
        self.assertEqual(dashboard_entry(payload)["model"], "fixture-model")
        self.assertEqual(json.loads(Path(result["path"]).read_text()), payload)
        for path in self.root.rglob("*.json"):
            self.assertNotIn("test-session-key", path.read_text())
        self.assertNotIn("test-session-key", json.dumps(job))

    def test_failed_warmup_is_visible_without_a_collectable_result(self):
        self.provider.reject = True
        self.call("/api/run", {"config": self.config})
        job = self.finish()
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"], "Warmup failed: HTTP 401")
        self.assertIsNone(job["resultFile"])
        self.assertEqual(self.call("/api/history"), [])
        self.assertEqual(self.provider.calls, 1)

    def test_workload_overrides_are_rejected_at_the_http_boundary(self):
        with self.assertRaises(error.HTTPError) as caught:
            self.call("/api/run", {"config": {**self.config, "repetitions": 2}})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.provider.calls, 0)

    def test_cancel_while_warmup_running_and_reject_duplicate(self):
        self.provider.release.clear()
        self.call("/api/run", {"config": self.config})
        self.assertTrue(self.provider.started.wait(2))
        with self.assertRaises(error.HTTPError) as caught:
            self.call("/api/run", {"config": self.config})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.call("/api/cancel", {})["status"], "stopping")
        self.provider.release.set()
        job = self.finish()
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(self.provider.calls, 1)
        result = self.call("/api/results/" + job["resultFile"])
        self.assertEqual(len(result["payload"]["samples"]), 1)

    def test_authorization_and_file_boundaries(self):
        for path, headers in (
            ("/api/bootstrap", {"Origin": "https://untrusted.example"}),
            ("/api/bootstrap", {"Host": "untrusted.example"}),
            ("/api/history", {"X-Session-Token": "wrong"}),
            ("/api/bootstrap", {"Sec-Fetch-Site": "cross-site"}),
        ):
            with self.subTest(path=path, headers=headers), self.assertRaises(error.HTTPError) as caught:
                self.call(path, headers=headers)
            self.assertEqual(caught.exception.code, 403)
        with self.assertRaises(error.HTTPError):
            self.call("/api/profiles", {"name": "unsafe", "config": {**self.config, "api_key": "do-not-save"}})
        with self.assertRaises(ValueError):
            self.workbench.result("../../.env")
        (self.root / "results").mkdir()
        (self.root / "private.json").write_text('{}')
        (self.root / "results" / "llm-api-escape.json").symlink_to(self.root / "private.json")
        with self.assertRaises(ValueError):
            self.workbench.result("llm-api-escape.json")

    def test_profiles_survive_restart_without_keys(self):
        profiles = self.call("/api/profiles", {"name": "Local server", "config": self.config})
        self.assertEqual(profiles["Local server"]["model"], "fixture-model")
        other = Workbench(self.workbench.output_dir, self.workbench.state_dir, self.workbench.env_file)
        self.assertEqual(other.profiles(), profiles)
        self.assertEqual(BenchmarkConfig.from_dict(profiles["Local server"]).plan()["requests"], 23)

    def test_legacy_profile_migration_only_preserves_connections(self):
        self.workbench.state_dir.mkdir()
        (self.workbench.state_dir / "profiles.json").write_text(json.dumps({
            "Previous profile": {**self.config, "repetitions": 20, "concurrency": 4, "prompt_chars": [1, 2, 3]},
        }))
        bootstrap = self.call("/api/bootstrap")
        self.assertEqual(bootstrap["suite"]["id"], "api-standard-v2")
        self.assertEqual(bootstrap["profiles"]["Previous profile"]["model"], "fixture-model")
        self.assertNotIn("repetitions", bootstrap["profiles"]["Previous profile"])
        self.assertNotIn("concurrency", bootstrap["defaults"])


if __name__ == "__main__":
    unittest.main()
