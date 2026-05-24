#!/usr/bin/env bash
# download_vindr_spine.sh — VinDr-SpineXR PhysioNet download helper
# Requires: credentialed PhysioNet account + signed DUA for VinDr-SpineXR
set -euo pipefail

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    # Optional exit code argument; defaults to 0 (--help path uses 0, error path uses 1)
    local exit_code="${1:-0}"
    cat <<EOF
Usage: download_vindr_spine.sh --username USERNAME [--output_dir PATH]
                                [--parallel] [--jobs N]

Options:
  --username    PhysioNet username (required)
  --output_dir  Download destination (optional, overrides .env)
  --parallel    Use aria2c parallel download mode (falls back to wget if unavailable)
  --jobs N      Number of parallel connections for aria2c (default: 4)

Prerequisites:
  - PhysioNet credentialed account
  - Signed DUA for VinDr-SpineXR
  - Access approved at: https://physionet.org/content/vindr-spinexr/1.0.0/
EOF
    exit "$exit_code"
}

# ── Parse args ────────────────────────────────────────────────────────────────
USERNAME=""
OUTPUT_DIR_ARG=""
PARALLEL=false
JOBS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            ;;
        --username)
            USERNAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR_ARG="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ -z "$USERNAME" ]]; then
    echo "[ERROR] --username is required."
    echo ""
    usage 1
fi

# ── Resolve output directory ──────────────────────────────────────────────────
# Priority: 1) --output_dir arg  2) .env DATASET_PATH  3) hardcoded fallback
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [[ -n "$OUTPUT_DIR_ARG" ]]; then
    OUTPUT_DIR="$OUTPUT_DIR_ARG"
else
    if [[ -f "$ENV_FILE" ]]; then
        # shellcheck source=/dev/null
        source "$ENV_FILE"
    fi
    DATASET_PATH="${DATASET_PATH:-/media/mike/Datasets}"
    OUTPUT_DIR="${DATASET_PATH}/vindr-spinexr"
fi

echo "Resolved output directory: $OUTPUT_DIR"

# ── Prerequisites notice ──────────────────────────────────────────────────────
echo ""
echo "=== VinDr-SpineXR PhysioNet Download ==="
echo "Ensure the following before proceeding:"
echo "  1. PhysioNet credentialed account"
echo "  2. Signed DUA for VinDr-SpineXR"
echo "  3. Access approved at https://physionet.org/content/vindr-spinexr/1.0.0/"
echo ""
echo "Username:    $USERNAME"
echo "Output dir:  $OUTPUT_DIR"
echo ""
echo "You will be prompted for your PhysioNet password."
echo "Password is NOT stored."

# ── Disk space check ──────────────────────────────────────────────────────────
echo ""
echo "=== Available disk space ==="
df -h "$(dirname "$OUTPUT_DIR")"
echo ""

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Download ──────────────────────────────────────────────────────────────────
cd "$OUTPUT_DIR"

_wget_sequential() {
    wget -r -N -c -np \
        --user "$USERNAME" \
        --ask-password \
        --show-progress \
        --progress=bar:force:noscroll \
        https://physionet.org/files/vindr-spinexr/1.0.0/
}

if [[ "$PARALLEL" == false ]]; then
    # ── Sequential wget (default) ─────────────────────────────────────────────
    _wget_sequential

else
    # ── Parallel aria2c mode ──────────────────────────────────────────────────

    # STEP 1 — Check aria2c is installed
    if ! command -v aria2c &> /dev/null; then
        echo "[WARN] aria2c not found. Install with: sudo apt install aria2"
        echo "[INFO] Falling back to wget sequential download."
        _wget_sequential
    else
        # STEP 2 — Ensure SHA256SUMS.txt is available
        SUMS_FILE=$(find "$OUTPUT_DIR" -name "SHA256SUMS.txt" | head -n 1)
        if [[ -n "$SUMS_FILE" ]]; then
            echo "[INFO] Reusing existing SHA256SUMS.txt: $SUMS_FILE"
        else
            echo "[INFO] Downloading SHA256SUMS.txt ..."
            wget -N -c \
                --user "$USERNAME" \
                --ask-password \
                --show-progress \
                --progress=bar:force:noscroll \
                -P "$OUTPUT_DIR" \
                https://physionet.org/files/vindr-spinexr/1.0.0/SHA256SUMS.txt
            SUMS_FILE="$OUTPUT_DIR/SHA256SUMS.txt"
        fi

        # STEP 3 — Build aria2 input file from SHA256SUMS.txt
        # Format: <hash>  <path/to/file>
        # aria2 input format:
        #   https://…/<path>
        #     out=<path>
        ARIA2_INPUT="$OUTPUT_DIR/urls_aria2.txt"
        : > "$ARIA2_INPUT"   # truncate / create
        while IFS= read -r line; do
            # Skip blank lines and comment lines
            [[ -z "$line" || "$line" == \#* ]] && continue
            FILE_PATH=$(echo "$line" | awk '{print $2}')
            [[ -z "$FILE_PATH" ]] && continue
            printf 'https://physionet.org/files/vindr-spinexr/1.0.0/%s\n' "$FILE_PATH" >> "$ARIA2_INPUT"
            printf '  out=%s\n' "$FILE_PATH" >> "$ARIA2_INPUT"
        done < "$SUMS_FILE"

        URL_COUNT=$(grep -c '^https://physionet.org/files/vindr-spinexr/1.0.0/' "$ARIA2_INPUT" || true)
        echo "Built aria2 URL list: $URL_COUNT files"

        # STEP 4 — Run aria2c
        aria2c \
            --http-user="$USERNAME" \
            --ask-password=true \
            -x "$JOBS" \
            -s "$JOBS" \
            -j "$JOBS" \
            --continue=true \
            --auto-file-renaming=false \
            --allow-overwrite=false \
            -d . \
            -i "$ARIA2_INPUT"

        # STEP 5 — Checksum note
        echo ""
        echo "Checksum file available: $OUTPUT_DIR/SHA256SUMS.txt"
        echo "Full checksum validation can be added later."
    fi
fi

# ── Post-download dataset root detection ──────────────────────────────────────
# wget -r creates nested dirs (e.g. physionet.org/files/vindr-spinexr/1.0.0/)
# Find the shallowest directory that actually contains data files.
DETECTED_ROOT=$(find "$OUTPUT_DIR" -type f \( -name "*.dicom" -o -name "*.dcm" -o -name "*.csv" \) \
    -printf '%h\n' | sort | uniq | head -n 1)

if [[ -z "$DETECTED_ROOT" ]]; then
    echo "[WARN] Could not detect dataset root. Defaulting to $OUTPUT_DIR"
    DETECTED_ROOT="$OUTPUT_DIR"
fi

echo ""
echo "Output dir:            $OUTPUT_DIR"
echo "Detected dataset root: $DETECTED_ROOT"

# ── Post-download validation ──────────────────────────────────────────────────
echo ""
echo "=== File counts ==="
DICOM_COUNT=$(find "$DETECTED_ROOT" -name "*.dicom" | wc -l)
DCM_COUNT=$(find "$DETECTED_ROOT" -name "*.dcm"   | wc -l)
CSV_COUNT=$(find "$DETECTED_ROOT" -name "*.csv"   | wc -l)

echo "  DICOM files (.dicom): $DICOM_COUNT"
echo "  DCM files   (.dcm):   $DCM_COUNT"
echo "  CSV files   (.csv):   $CSV_COUNT"

echo ""
echo "=== Folder structure (2 levels) ==="
find "$DETECTED_ROOT" -maxdepth 2 -type d | sort

# ── Next steps ────────────────────────────────────────────────────────────────
echo ""
echo "=== Download complete. Next steps: ==="
echo ""
echo "Validate dataset:"
echo "  python scripts/check_dataset.py --data_dir $DETECTED_ROOT"
echo ""
echo "Start EDA environment:"
echo "  docker compose --env-file .env -f infra/docker/docker-compose.yml up"
echo ""
echo "Inside the notebook, dataset is available at:"
echo "  /data/vindr-spinexr"
