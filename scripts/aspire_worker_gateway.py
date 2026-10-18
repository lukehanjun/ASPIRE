#!/usr/bin/env python3
"""Record native Haiku coding-agent calls with a shared conservative cost cap."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import threading
import uuid

import requests


def now():
    return datetime.now(timezone.utc).isoformat()


def usage_from_response(body, content_type):
    usage = {}
    if "event-stream" in content_type:
        for line in body.decode().splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except ValueError:
                continue
            usage.update(event.get("message", {}).get("usage", {}))
            usage.update(event.get("usage", {}))
    else:
        try:
            usage = json.loads(body).get("usage", {})
        except ValueError:
            pass
    return usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--port", type=int, default=8231)
    args = parser.parse_args()
    root = args.run_root.resolve()
    transport = root / "transport"
    transport.mkdir(parents=True, exist_ok=True)
    credentials = {}
    repo = Path(__file__).resolve().parents[1]
    for line in (repo / ".env").read_text().splitlines():
        name, equal, value = line.strip().removeprefix("export ").partition("=")
        if equal and name == "VECTOR_API_KEY":
            credentials[name] = " ".join(shlex.split(value, comments=True))
    key = credentials["VECTOR_API_KEY"]
    state_file = transport / "budget.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        "calls": 0, "conservative_list_usd": 0.0, "reserved_usd": 0.0,
    }
    state["reserved_usd"] = 0.0
    lock = threading.Lock()

    def save():
        tmp = state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({**state, "updated_at": now(), "cap_usd": args.budget_usd}, indent=2) + "\n")
        tmp.replace(state_file)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, content_type="application/json"):
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            with lock:
                self.send(200, {"status": "ready", **state})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path not in ("/v1/messages", "/v1/messages?beta=true", "/v1/messages/count_tokens", "/v1/messages/count_tokens?beta=true"):
                self.send(404, {"error": {"type": "invalid_request_error", "message": "unsupported coding-agent route"}})
                return
            if payload.get("model") != "claude-haiku-5-5":
                self.send(400, {"error": {"type": "invalid_request_error", "message": "only Haiku 5.5 is allowed"}})
                return
            maximum = int(payload.get("max_tokens", 0))
            if maximum > 200000:
                self.send(400, {"error": {"type": "invalid_request_error", "message": "completion token limit exceeds 200000"}})
                return
            # Reserve every possible input byte at the one-hour long-context
            # cache-write rate (2 x uncached), plus the full output limit. Charge reported
            # cached usage at its corresponding long-context catalog rate.
            reserve = (len(json.dumps(payload).encode()) * 1.0 + maximum * 2.5) / 1e6
            with lock:
                if state["conservative_list_usd"] + state["reserved_usd"] + reserve > args.budget_usd:
                    self.send(402, {"type": "error", "error": {"type": "invalid_request_error", "message": "shared experiment API budget exhausted"}})
                    return
                state["reserved_usd"] += reserve
                save()
            scope = self.headers.get("x-api-key", "unscoped")
            headers = {"x-api-key": key, "Authorization": "Bearer " + key, "Content-Type": "application/json"}
            for name in ("anthropic-version", "anthropic-beta"):
                if self.headers.get(name):
                    headers[name] = self.headers[name]
            started = now()
            accounted = False
            try:
                response = requests.post("https://proxy.vectorinstitute.ai" + self.path, headers=headers, json=payload, timeout=500)
                body = response.content
                content_type = response.headers.get("Content-Type", "application/json")
                usage = usage_from_response(body, content_type)
                tokens_out = usage.get("output_tokens", 0)
                charge = (
                    usage.get("input_tokens", 0) * 0.5
                    + usage.get("cache_read_input_tokens", 0) * 0.05
                    + usage.get("cache_creation_input_tokens", 0) * 0.625
                    + usage.get("cache_creation", {}).get("ephemeral_1h_input_tokens", 0) * 0.375
                    + tokens_out * 2.5
                ) / 1e6
                # Successful calls with absent usage retain the complete reserve.
                if response.ok and not usage and "/count_tokens" not in self.path:
                    charge = reserve
                with lock:
                    state["reserved_usd"] -= reserve
                    state["conservative_list_usd"] += charge
                    state["calls"] += 1
                    save()
                    accounted = True
                record = {"scope": scope, "started_at": started, "finished_at": now(), "status": response.status_code, "path": self.path, "request": payload, "response_content_type": content_type, "response": body.decode(), "usage": usage, "conservative_list_usd": charge}
                raw = json.dumps(record)
                for secret in credentials.values():
                    raw = raw.replace(secret, "[REDACTED]")
                with gzip.open(transport / (uuid.uuid4().hex + ".json.gz"), "wt") as out:
                    out.write(raw)
                self.send(response.status_code, body, content_type)
            except Exception as exc:
                with lock:
                    if not accounted:
                        state["reserved_usd"] -= reserve
                        # The provider may have processed an interrupted request.
                        state["conservative_list_usd"] += reserve
                        save()
                try:
                    self.send(502, {"type": "error", "error": {"type": "api_error", "message": type(exc).__name__}})
                except (BrokenPipeError, ConnectionResetError):
                    pass

    save()
    print(f"Haiku coding-agent gateway ready on localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
