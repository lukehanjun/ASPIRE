#!/usr/bin/env python3
"""Run a recorded Claude Code worker through Vector's Anthropic-compatible API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
from urllib.parse import urlparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--timeout-minutes", type=float, default=90)
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--preflight", action="store_true", help="allow only the Read tool")
    parser.add_argument("--coordinator", action="store_true", help="also allow native subagents")
    parser.add_argument("--gateway-url", help="local recorded native-messages gateway")
    parser.add_argument("--scope", default="aspire-worker", help="local bookkeeping label")
    parser.add_argument("--cwd", type=Path, help="worker working directory (default: repository)")
    args = parser.parse_args()
    if args.budget_usd <= 0 or args.timeout_minutes <= 0:
        parser.error("budget and timeout must be positive")

    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "invocation.json").exists():
        parser.error("output directory already contains an invocation; use a new directory")
    prompt = args.prompt_file.resolve().read_text()
    credentials = {}
    if args.gateway_url:
        parsed = urlparse(args.gateway_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            parser.error("the recording gateway must be on http://127.0.0.1")
        key = args.scope
    else:
        for line in (repo / ".env").read_text().splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and name in {"VECTOR_API_KEY", "OPENAI_API_KEY"}:
                tokens = shlex.split(value, comments=True)
                credentials[name] = " ".join(tokens)
        key = credentials.get("VECTOR_API_KEY")
        if not key:
            parser.error("VECTOR_API_KEY is missing from the repository .env")
    env = os.environ.copy()
    for name in (
        "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY",
        "ANTHROPIC_CUSTOM_HEADERS", "OPENAI_API_KEY", "VECTOR_API_KEY",
        "NVIDIA_API_KEY", "OPENROUTER_API_KEY",
    ):
        env.pop(name, None)
    env.update(
        ANTHROPIC_API_KEY=key,
        ANTHROPIC_CUSTOM_HEADERS="Authorization: Bearer " + key,
        ANTHROPIC_BASE_URL=args.gateway_url or "https://proxy.vectorinstitute.ai",
        CLAUDE_CONFIG_DIR=str(output / "claude-config"),
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        ANTHROPIC_DEFAULT_HAIKU_MODEL=args.model,
        ANTHROPIC_DEFAULT_SONNET_MODEL=args.model,
        ANTHROPIC_DEFAULT_OPUS_MODEL=args.model,
        CLAUDE_CODE_SUBAGENT_MODEL=args.model,
    )
    allowed = "Read" if args.preflight else "Bash,Read,Write,Edit,Glob,Grep"
    if args.coordinator:
        allowed += ",Agent,SendMessage,TaskOutput,TaskStop"
    command = [
        "claude", "--bare", "--model", args.model, "--print", "--verbose",
        "--output-format", "stream-json", "--max-budget-usd", str(args.budget_usd),
        "--tools", allowed, "--allowedTools", allowed,
        "--permission-mode", "acceptEdits", "--setting-sources", "",
    ]
    invocation = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "prompt_file": str(args.prompt_file.resolve()),
        "endpoint": env["ANTHROPIC_BASE_URL"],
        "credential_source": "gateway-only; worker has a bookkeeping label" if args.gateway_url else ".env:VECTOR_API_KEY",
        "cwd": str(args.cwd.resolve() if args.cwd else repo),
        "auth": "x-api-key plus Authorization Bearer",
        "timeout_minutes": args.timeout_minutes,
    }
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    (output / "prompt.md").write_text(prompt)

    def record(source, destination: Path) -> None:
        with destination.open("w", buffering=1) as target:
            for line in source:
                clean = line
                for secret in credentials.values():
                    if secret:
                        clean = clean.replace(secret, "[REDACTED]")
                target.write(clean)

    child = subprocess.Popen(
        command, cwd=args.cwd or repo, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True,
    )
    (output / "worker.pid").write_text(str(child.pid) + "\n")
    threads = [
        threading.Thread(target=record, args=(child.stdout, output / "transcript.jsonl")),
        threading.Thread(target=record, args=(child.stderr, output / "stderr.log")),
    ]
    for thread in threads:
        thread.start()
    child.stdin.write(prompt)
    child.stdin.close()
    timed_out = False
    try:
        child.wait(timeout=args.timeout_minutes * 60)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    finally:
        for thread in threads:
            thread.join()
    results = []
    for line in (output / "transcript.jsonl").read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "result":
            results.append(event)
    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "returncode": child.returncode,
        "timed_out": timed_out,
        "result": results[-1] if results else None,
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output_dir": str(output), **summary}))
    return 124 if timed_out else child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
