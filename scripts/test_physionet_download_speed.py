"""
test_physionet_download_speed.py — PhysioNet download speed diagnostic.

Tests download speed for a small sample of VinDr-SpineXR files using
authenticated requests directly against the PhysioNet file server.

Requires SHA256SUMS.txt to already be present in the output directory.
Run the main download script first if it is not:
    bash scripts/download_vindr_spine.sh --username YOUR_USERNAME

Password is prompted securely and never stored or logged.
"""

import argparse
import getpass
import os
import sys
import time
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL = "https://physionet.org/files/vindr-spinexr/1.0.0/"
CHUNK_SIZE = 1_048_576  # 1 MB streaming chunks


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostic: test PhysioNet download speed for VinDr-SpineXR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/test_physionet_download_speed.py --username mmontielpz --limit 5
""",
    )
    parser.add_argument(
        "--username",
        required=True,
        help="PhysioNet username (required)",
    )
    parser.add_argument(
        "--output-dir",
        default="/media/mike/Datasets/vindr-spinexr",
        help="Directory containing downloaded data (default: /media/mike/Datasets/vindr-spinexr)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of files to test (default: 10)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Reserved for future use — parallel mode is not yet implemented "
            "in this diagnostic. Only sequential downloads are performed."
        ),
    )
    return parser.parse_args()


# ── SHA256SUMS.txt discovery ───────────────────────────────────────────────────
def find_sums_file(output_dir: Path) -> Path:
    candidates = [
        output_dir / "SHA256SUMS.txt",
        output_dir / "physionet.org" / "files" / "vindr-spinexr" / "1.0.0" / "SHA256SUMS.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    print(
        "[ERROR] SHA256SUMS.txt not found. Run the main download script first\n"
        "        to fetch it, or download it manually from PhysioNet."
    )
    sys.exit(1)


# ── Parse SHA256SUMS.txt ───────────────────────────────────────────────────────
def parse_sums_file(sums_path: Path, limit: int) -> list[str]:
    files = []
    with open(sums_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)  # split on first whitespace
            if len(parts) < 2:
                continue  # malformed line — skip silently
            file_path = parts[1].strip()
            if file_path:
                files.append(file_path)
            if len(files) >= limit:
                break
    print(f"Selected {len(files)} files for speed test.")
    return files


# ── Per-file download ─────────────────────────────────────────────────────────
def download_file(
    file_path: str,
    test_dir: Path,
    username: str,
    password: str,
) -> dict:
    """
    Download one file to test_dir, preserving subdirectory structure.
    Supports HTTP resume (Range header). Returns a result dict.
    """
    url = BASE_URL + file_path
    dest = test_dir / file_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    headers = {}
    existing_size = 0
    mode = "wb"
    if dest.exists() and dest.stat().st_size > 0:
        existing_size = dest.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    bytes_downloaded = 0
    status_code = None
    got_403 = False
    start = time.monotonic()

    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            headers=headers,
            stream=True,
            timeout=30,
        )
        status_code = response.status_code

        if status_code == 206:
            # Partial Content — resume accepted
            pass
        elif status_code == 200:
            # Full download — start fresh regardless of existing file
            mode = "wb"
            existing_size = 0
        elif status_code == 403:
            got_403 = True
            print(
                "[HTTP 403] PhysioNet is rejecting this request method or\n"
                "           throttling authenticated direct downloads."
            )
        else:
            print(f"[WARN] Unexpected status {status_code} for {file_path}")

        if status_code in (200, 206):
            with open(dest, mode) as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

    except requests.RequestException as exc:
        print(f"[ERROR] Request failed for {file_path}: {exc}")
        status_code = status_code or -1

    elapsed = max(time.monotonic() - start, 1e-6)  # avoid division by zero
    speed_mbps = (bytes_downloaded / CHUNK_SIZE) / elapsed

    print(f"  File:    {file_path}")
    print(f"  Status:  {status_code}")
    print(f"  Size:    {bytes_downloaded / CHUNK_SIZE:.2f} MB")
    print(f"  Time:    {elapsed:.2f}s")
    print(f"  Speed:   {speed_mbps:.2f} MB/s")

    return {
        "file_path": file_path,
        "status_code": status_code,
        "bytes_downloaded": bytes_downloaded,
        "elapsed": elapsed,
        "speed_mbps": speed_mbps,
        "got_403": got_403,
        "success": status_code in (200, 206),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    test_dir = output_dir / "_speed_test"

    # Secure password prompt — not stored, not logged
    password = getpass.getpass(prompt=f"Password for user '{args.username}': ")

    # Locate SHA256SUMS.txt
    sums_path = find_sums_file(output_dir)
    print(f"Using SHA256SUMS.txt: {sums_path}")

    # Parse file list
    files = parse_sums_file(sums_path, args.limit)

    print()
    print(f"=== Speed Test: {len(files)} files → {test_dir} ===")
    print()

    # Run downloads
    results = []
    for file_path in files:
        result = download_file(file_path, test_dir, args.username, password)
        results.append(result)
        print()

    # Clear password from local scope as soon as it is no longer needed
    del password

    # Summary
    n_attempted = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_failed = n_attempted - n_success
    total_bytes = sum(r["bytes_downloaded"] for r in results)
    total_mb = total_bytes / CHUNK_SIZE
    total_elapsed = sum(r["elapsed"] for r in results)
    aggregate_mbps = total_mb / max(total_elapsed, 1e-6)
    any_403 = any(r["got_403"] for r in results)

    print("=== Speed Test Summary ===")
    print(f"Files attempted:    {n_attempted}")
    print(f"Successful:         {n_success}")
    print(f"Failed:             {n_failed}")
    print(f"Total downloaded:   {total_mb:.2f} MB")
    print(f"Total time:         {total_elapsed:.2f}s")
    print(f"Aggregate speed:    {aggregate_mbps:.2f} MB/s")

    if any_403:
        print()
        print(
            "[NOTE] 403 errors suggest PhysioNet may require browser-based\n"
            "       session auth or is throttling this download method.\n"
            "       Consider using the wget script with valid credentials instead."
        )


if __name__ == "__main__":
    main()
