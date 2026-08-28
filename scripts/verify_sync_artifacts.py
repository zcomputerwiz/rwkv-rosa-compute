#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Verify sha256 sidecar files.")
    parser.add_argument("directories", nargs="*", default=["."], help="Directories to verify")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    files = set()
    for directory in args.directories:
        if os.path.isfile(directory):
            files.add(os.path.normpath(directory))
        elif os.path.isdir(directory):
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    files.add(os.path.normpath(os.path.join(root, filename)))
        else:
            # For non-existent files or directories we'll just ignore or let it be empty
            pass

    artifacts = {f for f in files if not f.endswith(".sha256")}
    sidecars = {f for f in files if f.endswith(".sha256")}

    all_targets = artifacts.union({s[:-7] for s in sidecars})

    output_records = []
    has_failure = False

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
                # Read with binary first if we want to be super explicit about CRLF handling,
                # though python's universal newlines handle it if encoding="utf-8"
                with open(sidecar, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()

                # The sidecar can contain "<digest>  <bare filename>" or "<digest> *<name>"
                # So we just extract the first 64 characters.
                expected_digest = ""
                if len(first_line) >= 64:
                    expected_digest = first_line[:64]

                h = hashlib.sha256()
                with open(target, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        h.update(chunk)
                actual_digest = h.hexdigest()

                if expected_digest and actual_digest.lower() == expected_digest.lower():
                    status = "verified"
                else:
                    status = "MISMATCH"
                    has_failure = True
            except Exception:
                status = "MISMATCH"
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
