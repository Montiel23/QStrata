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

Options:
  --username    PhysioNet username (required)
  --output_dir  Download destination (optional, overrides .env)

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

wget -r -N -c -np \
    --user "$USERNAME" \
    --ask-password \
    --show-progress \
    --progress=bar:force:noscroll \
    https://physionet.org/files/vindr-spinexr/1.0.0/

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
