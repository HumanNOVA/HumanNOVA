#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_PATH="${1:-$SCRIPT_DIR/mpips_smplify_public_v2.zip}"
HUMANNOVA_ARCHIVE_PATH="$SCRIPT_DIR/essentials_humannova.tar.gz"
EXTRACT_DIR="$SCRIPT_DIR/smplify_public"
SOURCE_RELATIVE_PATH="code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
SOURCE_PATH="$EXTRACT_DIR/$SOURCE_RELATIVE_PATH"
TARGET_PATH="$SCRIPT_DIR/checkpoint/SMPL_NEUTRAL.pkl"

# The original source files: https://drive.google.com/file/d/1rGUTBTddfzlhky2hZq8vER7e6m-ahUjo/view?usp=sharing
gdown 1rGUTBTddfzlhky2hZq8vER7e6m-ahUjo
tar -xzvf essentials_humannova.tar.gz

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip not found in PATH"
  exit 1
fi

if [ ! -f "$ZIP_PATH" ]; then
  echo "SMPL archive not found: $ZIP_PATH"
  exit 1
fi

rm -rf "$EXTRACT_DIR"
unzip -q "$ZIP_PATH" -d "$SCRIPT_DIR"

if [ ! -f "$SOURCE_PATH" ]; then
  echo "Expected SMPL model not found after extraction: $SOURCE_PATH"
  exit 1
fi

cp "$SOURCE_PATH" "$TARGET_PATH"
rm -rf "$EXTRACT_DIR"

echo "Extracted SMPL model to: $TARGET_PATH"

if [ -f "$HUMANNOVA_ARCHIVE_PATH" ]; then
  tar -xzf "$HUMANNOVA_ARCHIVE_PATH" -C "$SCRIPT_DIR"
  echo "Extracted HumanNOVA essentials to: $SCRIPT_DIR"
fi
