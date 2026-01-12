import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from advisor.llm_client import LLMAdvisor
from config.manager import config_exists, load_config, save_config
from detector import DetectorManager, list_detectors
from git_utils.models import CommitData, FileDiff
from git_utils.repo import get_latest_commit

DEFAULT_THRESHOLD = 0.6


def _commit_to_dict(commit):
    return {
        "commit_id": commit.commit_id,
        "author": commit.author,
        "message": commit.message,
        "timestamp": commit.timestamp,
        "files_changed": commit.files_changed,
        "total_lines_added": commit.total_lines_added,
        "total_lines_deleted": commit.total_lines_deleted,
        "total_lines_changed": commit.total_lines_changed,
        "files": [
            {
                "file_path": item.file_path,
                "diff_text": item.diff_text,
                "lines_added": item.lines_added,
                "lines_deleted": item.lines_deleted,
                "file_type": item.file_type,
            }
            for item in commit.files
        ],
    }


def _coerce_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _commit_from_dict(data):
    if not isinstance(data, dict):
        return None

    files = []
    for item in data.get("files", []) or []:
        if not isinstance(item, dict):
            continue
        files.append(
            FileDiff(
                file_path=str(item.get("file_path", "")),
                diff_text=str(item.get("diff_text", "")),
                lines_added=_coerce_int(item.get("lines_added", 0)),
                lines_deleted=_coerce_int(item.get("lines_deleted", 0)),
                file_type=str(item.get("file_type", "other")),
            )
        )

    total_lines_added = _coerce_int(
        data.get("total_lines_added", sum(item.lines_added for item in files))
    )
    total_lines_deleted = _coerce_int(
        data.get("total_lines_deleted", sum(item.lines_deleted for item in files))
    )
    total_lines_changed = _coerce_int(
        data.get(
            "total_lines_changed",
            sum(item.lines_added + item.lines_deleted for item in files),
        )
    )

    return CommitData(
        commit_id=str(data.get("commit_id", "")),
        author=str(data.get("author", "")),
        message=str(data.get("message", "")),
        timestamp=_coerce_int(data.get("timestamp", 0)),
        files=files,
        files_changed=_coerce_int(data.get("files_changed", len(files))),
        total_lines_added=total_lines_added,
        total_lines_deleted=total_lines_deleted,
        total_lines_changed=total_lines_changed,
    )


class CommitTutorHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.manager = DetectorManager()
        self.threshold = DEFAULT_THRESHOLD
        self.repo_path = None
        if config_exists():
            config = load_config()
            if config.detector_model:
                self.manager.set_current(config.detector_model)

    def server_close(self):
        self.manager.shutdown()
        super().server_close()


class CommitTutorRequestHandler(BaseHTTPRequestHandler):
    server_version = "CommitTutorHTTP/0.1"

    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status_code, code, message):
        self._send_json(status_code, {"error": {"code": code, "message": message}})

    def _read_json(self):
        length = _coerce_int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return None
        raw = self.rfile.read(length).decode("utf-8")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def do_OPTIONS(self):
        self._send_json(204, {})

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/config":
            self._handle_get_config()
            return
        if path == "/detectors":
            self._handle_get_detectors()
            return
        if path == "/commits/latest":
            self._handle_get_latest_commit()
            return
        if path == "/threshold":
            self._handle_get_threshold()
            return
        if path == "/workspace":
            self._handle_get_workspace()
            return
        if path == "/health":
            self._handle_get_health()
            return
        self._send_error(404, "not_found", "Unknown endpoint.")

    def do_PUT(self):
        path = urlparse(self.path).path
        if path == "/config":
            self._handle_put_config()
            return
        if path == "/detectors/current":
            self._handle_put_detector()
            return
        if path == "/threshold":
            self._handle_put_threshold()
            return
        if path == "/workspace":
            self._handle_put_workspace()
            return
        self._send_error(404, "not_found", "Unknown endpoint.")

    def do_PATCH(self):
        path = urlparse(self.path).path
        if path == "/config/llm":
            self._handle_patch_llm()
            return
        self._send_error(404, "not_found", "Unknown endpoint.")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/analysis":
            self._handle_post_analysis()
            return
        if path == "/advice":
            self._handle_post_advice()
            return
        self._send_error(404, "not_found", "Unknown endpoint.")

    def _handle_get_config(self):
        if not config_exists():
            self._send_json(200, {"exists": False, "config": None})
            return
        config = load_config()
        self._send_json(200, {"exists": True, "config": config.to_dict()})

    def _handle_put_config(self):
        data = self._read_json()
        if not isinstance(data, dict):
            self._send_error(400, "invalid_json", "Request body must be JSON.")
            return

        required = ["base_url", "api_key", "model_name"]
        missing = [key for key in required if not data.get(key)]
        if missing:
            self._send_error(400, "missing_fields", f"Missing fields: {', '.join(missing)}.")
            return

        config = save_config(data)
        if config.detector_model:
            self.server.manager.set_current(config.detector_model)
        self._send_json(200, {"config": config.to_dict()})

    def _handle_patch_llm(self):
        if not config_exists():
            self._send_error(400, "config_missing", "Config not initialized.")
            return

        data = self._read_json()
        if not isinstance(data, dict):
            self._send_error(400, "invalid_json", "Request body must be JSON.")
            return

        config = load_config()
        if "base_url" in data:
            config.base_url = str(data.get("base_url", ""))
        if "api_key" in data:
            config.api_key = str(data.get("api_key", ""))
        if "model_name" in data:
            config.model_name = str(data.get("model_name", ""))

        save_config(config.to_dict())
        self._send_json(200, {"config": config.to_dict()})

    def _handle_get_detectors(self):
        detectors = list_detectors()
        current = ""
        if config_exists():
            current = load_config().detector_model
        self._send_json(200, {"detectors": detectors, "current": current})

    def _handle_put_detector(self):
        if not config_exists():
            self._send_error(400, "config_missing", "Config not initialized.")
            return

        data = self._read_json()
        if not isinstance(data, dict):
            self._send_error(400, "invalid_json", "Request body must be JSON.")
            return

        name = data.get("name")
        detectors = list_detectors()
        if name not in detectors:
            self._send_error(400, "invalid_detector", "Detector not found.")
            return

        config = load_config()
        config.detector_model = name
        save_config(config.to_dict())
        self.server.manager.set_current(name)
        self._send_json(200, {"current": name})

    def _handle_get_latest_commit(self):
        try:
            commit = get_latest_commit(repo_path=self.server.repo_path)
        except RuntimeError as exc:
            self._send_error(400, "git_error", str(exc))
            return
        self._send_json(200, {"commit": _commit_to_dict(commit)})

    def _handle_post_analysis(self):
        if not config_exists():
            self._send_error(400, "config_missing", "Config not initialized.")
            return

        config = load_config()
        if not config.detector_model:
            self._send_error(400, "detector_missing", "Detector model not selected.")
            return

        detector = self.server.manager.set_current(config.detector_model)
        if detector is None:
            self._send_error(400, "detector_missing", "Detector model not available.")
            return

        try:
            commit = get_latest_commit(repo_path=self.server.repo_path)
        except RuntimeError as exc:
            self._send_error(400, "git_error", str(exc))
            return

        score = detector.analyze(commit)
        threshold = self.server.threshold
        self._send_json(
            200,
            {
                "score": score,
                "threshold": threshold,
                "should_request_advice": score >= threshold,
                "commit": _commit_to_dict(commit),
            },
        )

    def _handle_post_advice(self):
        if not config_exists():
            self._send_error(400, "config_missing", "Config not initialized.")
            return

        data = self._read_json()
        if not isinstance(data, dict):
            self._send_error(400, "invalid_json", "Request body must be JSON.")
            return

        commit = _commit_from_dict(data.get("commit"))
        if commit is None:
            self._send_error(400, "missing_commit", "Commit payload is required.")
            return

        score = data.get("score")
        if score is None:
            self._send_error(400, "missing_score", "Score is required.")
            return

        history = data.get("history") if isinstance(data.get("history"), list) else []

        config = load_config()
        advisor = LLMAdvisor(config)
        feedback = advisor.generate_feedback(commit, float(score), history=history)
        self._send_json(200, {"feedback": feedback})

    def _handle_get_threshold(self):
        self._send_json(200, {"threshold": self.server.threshold})

    def _handle_put_threshold(self):
        data = self._read_json()
        if not isinstance(data, dict):
            self._send_error(400, "invalid_json", "Request body must be JSON.")
            return

        threshold = data.get("threshold")
        try:
            value = float(threshold)
        except (TypeError, ValueError):
            self._send_error(400, "invalid_threshold", "Threshold must be a number.")
            return

        if value < 0 or value > 1:
            self._send_error(400, "invalid_threshold", "Threshold must be between 0 and 1.")
            return

        self.server.threshold = value
        self._send_json(200, {"threshold": value})

    def _handle_get_workspace(self):
        self._send_json(200, {"path": self.server.repo_path})

    def _handle_get_health(self):
        self._send_json(200, {"status": "ok"})

    def _handle_put_workspace(self):
        data = self._read_json()
        if not isinstance(data, dict):
            self._send_error(400, "invalid_json", "Request body must be JSON.")
            return

        path = data.get("path")
        if not path:
            self._send_error(400, "missing_path", "Workspace path is required.")
            return

        if not os.path.isdir(path):
            self._send_error(400, "invalid_path", "Workspace path must be a directory.")
            return

        self.server.repo_path = path
        self._send_json(200, {"path": self.server.repo_path})


def run(host="127.0.0.1", port=8765):
    server = CommitTutorHTTPServer((host, port), CommitTutorRequestHandler)
    try:
        print(f"CommitTutor HTTP server listening on http://{host}:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["run"]
