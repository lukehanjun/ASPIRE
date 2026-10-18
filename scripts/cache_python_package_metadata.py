#!/usr/bin/env python3
"""Mirror package metadata locally while linking unchanged package code/data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = args.site_packages.resolve()
    target = args.destination.resolve()
    target.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def copy_entry(entry):
        path = Path(entry.path)
        output = target / entry.name
        if entry.name.endswith((".dist-info", ".egg-info")):
            if entry.is_dir():
                shutil.copytree(path, output)
            else:
                shutil.copy2(path, output)
            hashes = {}
            for file in output.rglob("*") if output.is_dir() else [output]:
                if file.is_file():
                    hashes[str(file.relative_to(target))] = hashlib.sha256(file.read_bytes()).hexdigest()
            return {"metadata": entry.name, "file_hashes": hashes}
        output.symlink_to(path, target_is_directory=entry.is_dir())
        return None

    with os.scandir(source) as entries, ThreadPoolExecutor(max_workers=12) as pool:
        records = [record for record in pool.map(copy_entry, list(entries)) if record]
    manifest = {"source": str(source), "cache": str(target), "metadata_directories": len(records), "elapsed_seconds": time.monotonic() - started, "package_code_and_data": "symlinks to original installation; only distribution metadata is copied", "records": records}
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
