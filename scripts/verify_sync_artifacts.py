#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import sys

def main():
    parser = argparse.ArgumentParser(description="Verify sha256 sidecar files. By default excludes .stversions and .stfolder.")
    parser.add_argument("directories", nargs="*", default=["."], help="Directories to verify")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--exclude", action="append", default=[], help="Path components to exclude (repeatable)")
    parser.add_argument("--no-default-excludes", action="store_true", help="Do not apply default excludes (.stversions, .stfolder)")
    args = parser.parse_args()

    excludes = args.exclude.copy()
    if not args.no_default_excludes:
        if ".stversions" not in excludes:
            excludes.append(".stversions")
        if ".stfolder" not in excludes:
            excludes.append(".stfolder")

    def path_is_excluded(path_str):
        parts = path_str.split(os.sep)
        if os.altsep:
            new_parts = []
            for p in parts:
                new_parts.extend(p.split(os.altsep))
            parts = new_parts

        for part in parts:
            if part in excludes:
                return True
        return False

    files = set()
    for directory in args.directories:
        if os.path.isfile(directory):
            norm = os.path.normpath(directory)
            if not path_is_excluded(norm):
                files.add(norm)
        elif os.path.isdir(directory):
            for root, dirs, filenames in os.walk(directory):
                dirs[:] = [d for d in dirs if d not in excludes]

                for filename in filenames:
                    filepath = os.path.normpath(os.path.join(root, filename))
                    if not path_is_excluded(filepath):
                        files.add(filepath)

    artifacts = {f for f in files if not f.endswith(".sha256")}
    sidecars = {f for f in files if f.endswith(".sha256")}

    all_targets = artifacts.union({s[:-7] for s in sidecars})

    output_records = []
    has_failure = False

    hex_pattern = re.compile(r'^[a-fA-F0-9]{64}$')

    for target in sorted(all_targets):
        sidecar = target + ".sha256"
        has_artifact = target in artifacts
        has_sidecar = sidecar in sidecars

        status = None
        record_file = target

        if not has_artifact and has_sidecar:
            status = "orphaned sidecar"
            record_file = sidecar
            has_failure = True
        elif has_artifact and not has_sidecar:
            status = "no sidecar"
        elif has_artifact and has_sidecar:
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()

                expected_digest = first_line[:64]
                if not hex_pattern.match(expected_digest):
                    status = "malformed sidecar"
                    has_failure = True
                else:
                    h = hashlib.sha256()
                    with open(target, "rb") as f:
                        while True:
                            chunk = f.read(65536)
                            if not chunk:
                                break
                            h.update(chunk)
                    actual_digest = h.hexdigest()

                    if actual_digest.lower() == expected_digest.lower():
                        status = "verified"
                    else:
                        status = "MISMATCH"
                        has_failure = True
            except OSError:
                status = "unreadable"
                has_failure = True

        if status:
            output_records.append({"file": record_file, "status": status})

    if args.json:
        print(json.dumps(output_records, indent=2))
    else:
        for record in output_records:
            print(f"{record['status']}: {record['file']}")

    if has_failure:
        sys.exit(1)

if __name__ == "__main__":
    main()
