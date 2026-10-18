#!/usr/bin/env python3
"""Record ASPIRE v1 calls and share the remaining authorized comparison budget."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8230)
    parser.add_argument("--budget-usd", type=float, default=58)
    args = parser.parse_args()
    root = args.run_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    credentials = {}
    for line in (repo / ".env").read_text().splitlines():
        name, equal, value = line.strip().removeprefix("export ").partition("=")
        if equal and name in ("VECTOR_API_KEY", "OPENAI_API_KEY"):
            credentials[name] = " ".join(shlex.split(value, comments=True))
    key = credentials["VECTOR_API_KEY"]
    lock = threading.Lock()
    state_file = root / "transport/budget.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        "estimated_list_usd": 0, "upper_bound_list_usd": 0, "reserved_upper_usd": 0,
        "calls": 0, "upstream_reported_usd": 0, "upstream_cost_headers_seen": 0,
    }
    state["reserved_upper_usd"] = 0

    def save():
        temporary = state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps({**state, "updated_at": now()}, indent=2) + "\n")
        temporary.replace(state_file)

    def campaign_budget():
        manifest = json.loads((root / "baseline.json").read_text())
        previous = Path(manifest["robospoon_reference_root"])
        previous_summary = json.loads((previous / "summary.json").read_text())
        # Reserve the entire other campaign allowance until its workers exit.
        other = manifest["robo_campaign_cap_usd"]
        if previous_summary["status"] == "complete":
            other = json.loads((previous / "transport/budget.json").read_text())["upper_bound_list_usd"]
        remaining = 60.0 - manifest["prior_campaigns_upper_bound_usd"] - manifest["cancelled_coding_preflights_allowance_usd"] - other
        return min(args.budget_usd, remaining)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, message_format, *arguments):
            pass

        def send(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            with lock:
                self.send(200, {"status": "ready", "model": "claude-haiku-5-5", **state})

        def do_POST(self):
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path not in ("/chat/completions", "/v1/chat/completions"):
                    self.send(404, {"error": "unknown baseline gateway route"})
                    return
                if payload.get("model") != "claude-haiku-5-5":
                    self.send(400, {"error": "this run allows only claude-haiku-5-5"})
                    return
                payload.pop("temperature", None)
                maximum = int(payload.get("max_tokens", 20480))
                if maximum > 20480:
                    self.send(400, {"error": "completion limit exceeds released baseline"})
                    return
                # Conservative reserve: at most one token per UTF-8 text byte, plus
                # 4,784 tokens/image (installed Claude Code's Haiku 5.5 model catalog).
                prompt_bound = 0
                for message in payload["messages"]:
                    content = message.get("content", "")
                    if isinstance(content, str):
                        prompt_bound += len(content.encode())
                    else:
                        for item in content:
                            prompt_bound += 4784 if item.get("type") == "image_url" else len(
                                item.get("text", "").encode()
                            )
                # Reserve at the more expensive >100k-context list-price tier.
                reserve = (prompt_bound * 0.5 + maximum * 2.5) / 1e6
                with lock:
                    if state["upper_bound_list_usd"] + state["reserved_upper_usd"] + reserve > campaign_budget():
                        self.send(402, {"error": "baseline API budget exhausted"})
                        return
                    state["reserved_upper_usd"] += reserve
                    save()
                request_id = uuid.uuid4().hex
                scope = self.headers.get("Authorization", "Bearer unscoped").removeprefix("Bearer ")
                started = now()
                accounted = False
                try:
                    response = requests.post(
                        "https://proxy.vectorinstitute.ai/v1/chat/completions",
                        headers={"Authorization": "Bearer " + key}, json=payload, timeout=500,
                    )
                    body = response.json()
                    usage = body.get("usage") or {}
                    input_count = usage.get("prompt_tokens", 0)
                    output_count = usage.get("completion_tokens", 0)
                    upper = (input_count * 0.5 + output_count * 2.5) / 1e6
                    estimate = upper if input_count > 100000 else (
                        input_count * 0.1 + output_count * 0.5
                    ) / 1e6
                    cost_header = response.headers.get("x-litellm-response-cost")
                    with lock:
                        state["reserved_upper_usd"] -= reserve
                        state["estimated_list_usd"] += estimate
                        state["upper_bound_list_usd"] += upper
                        state["calls"] += 1
                        if cost_header is not None:
                            state["upstream_reported_usd"] += float(cost_header)
                            state["upstream_cost_headers_seen"] += 1
                        save()
                        accounted = True
                    record = {
                        "id": request_id, "scope": scope, "started_at": started,
                        "finished_at": now(), "status": response.status_code,
                        "request": payload, "response": body,
                        "estimated_list_usd": estimate, "upper_bound_list_usd": upper,
                    }
                    clean = json.dumps(record)
                    for secret in credentials.values():
                        if secret:
                            clean = clean.replace(secret, "[REDACTED]")
                    with gzip.open(root / "transport" / (request_id + ".json.gz"), "wt") as output:
                        output.write(clean)
                    print(json.dumps({k: record[k] for k in (
                        "id", "scope", "status", "estimated_list_usd",
                    )}), flush=True)
                    self.send(response.status_code, body)
                except Exception:
                    if not accounted:
                        with lock:
                            # A request without usable usage may still have been billed.
                            state["reserved_upper_usd"] = max(0, state["reserved_upper_usd"] - reserve)
                            state["upper_bound_list_usd"] += reserve
                            save()
                    raise
            except Exception as error:
                self.send(502, {"error": type(error).__name__, "detail": str(error)[:500]})

    with lock:
        save()
    print(f"Baseline gateway ready on localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
