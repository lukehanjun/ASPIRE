#!/usr/bin/env python3
"""Validate paired CaP-Agent0 trials and report native success rates."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


def load_rows(root):
    return [json.loads(line) for line in (root / "trial-results.jsonl").read_text().splitlines() if "/evaluation/" in json.loads(line).get("output_dir", "")]


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / denominator
    delta = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [middle - delta, middle + delta]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = json.loads((root / "baseline.json").read_text())
    reference = Path(manifest["aspire_reference_root"])
    rows = load_rows(root)
    other = load_rows(reference)
    counts = Counter((row["task_id"], row["trial"]) for row in rows)
    expected = {(task, seed) for task in range(10) for seed in range(51, 66)}
    failures = []
    if set(counts) != expected or len(rows) != 150 or any(n != 1 for n in counts.values()):
        failures.append("missing, duplicate or unexpected task/seed trials")
    source_changes = [name for name, digest in json.loads((root / "source-sha256.json").read_text()).items() if hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() != digest]
    if source_changes:
        failures.append("frozen source files changed")
    directory_mismatches = []
    for row in rows:
        match = re.search(r"taskcompleted_(\d+)", str(row["code_path"]))
        if not match or bool(int(match.group(1))) != row["task_completed"] or not Path(row["code_path"]).is_file():
            directory_mismatches.append([row["task_id"], row["trial"]])
    if directory_mismatches:
        failures.append("native result-directory scoring mismatch")
    initial_state_mismatches = [[r["task_id"], r["trial"]] for r in rows if not r.get("matches_aspire_initial_state")]
    if initial_state_mismatches:
        failures.append("initial states differ from ASPIRE reference")
    if any(row["max_steps"] != 8000 for row in rows):
        failures.append("incorrect simulation horizon")
    transports = [json.load(gzip.open(p, "rt")) for p in (root / "transport").glob("*.json.gz")]
    statuses = Counter(r["status"] for r in transports)
    unexpected_models = sorted({r["request"]["model"] for r in transports if r["request"]["model"] != "claude-haiku-5-5"})
    if unexpected_models:
        failures.append("unexpected model calls")
    with (root / "results.csv").open("w") as out:
        columns = ["task_id", "trial", "task_completed", "program_ok", "simulator_steps", "num_code_blocks", "num_regenerations", "initial_state_sha256", "matches_aspire_initial_state", "code_path"]
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda row: (row["task_id"], row["trial"])):
            writer.writerow({**{key: row.get(key) for key in columns}, "program_ok": row["success"]})
    result = {
        "valid_complete_comparison": not failures, "failures": failures,
        "trials": len(rows), "source_changes": source_changes,
        "initial_state_mismatches": initial_state_mismatches,
        "directory_scoring_mismatches": directory_mismatches,
        "transport_statuses_including_preflight": dict(statuses),
        "unexpected_models": unexpected_models,
        "scope": "RoboSpoon source under the same frozen simulator runtime/assets/providers as the ASPIRE v0.1 reference; native-runtime checks reported separately",
    }
    if not failures:
        baseline = {(r["task_id"], r["trial"]): r["task_completed"] for r in other}
        pairs = [(baseline[(r["task_id"], r["trial"])], r["task_completed"]) for r in rows]
        only_aspire = sum(a and not b for a, b in pairs)
        only_robo = sum(b and not a for a, b in pairs)
        discordant = only_aspire + only_robo
        pvalue = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(only_aspire, only_robo) + 1)) / 2 ** discordant) if discordant else 1.0
        a, b = sum(x for x, _ in pairs), sum(y for _, y in pairs)
        result.update(
            aspire={"successes": a, "trials": 150, "success_rate": a / 150, "wilson_95_ci": wilson(a, 150)},
            robospoon={"successes": b, "trials": 150, "success_rate": b / 150, "wilson_95_ci": wilson(b, 150)},
            paired={"both_succeed": sum(x and y for x, y in pairs), "only_aspire_succeeds": only_aspire, "only_robospoon_succeeds": only_robo, "neither_succeeds": sum(not x and not y for x, y in pairs), "mcnemar_exact_two_sided_p": pvalue, "success_rate_difference_robospoon_minus_aspire": (b - a) / 150},
            interpretation="A non-significant result is not proof of equivalence or absence of bugs. Prompts/runner behavior differ, and sampling noise remains.",
        )
        result["per_task"] = [{"task_id": task, "aspire": sum(x["task_completed"] for x in other if x["task_id"] == task), "robospoon": sum(x["task_completed"] for x in rows if x["task_id"] == task), "trials_per_condition": 15} for task in range(10)]
    (root / "final-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
