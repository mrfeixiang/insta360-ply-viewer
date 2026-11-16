#!/bin/bash
# Batch process Insta360 X5 .insv files to web-ready PLY
# Selected 7 diverse videos from different dates/locations

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
SCENES_DIR="$SCRIPT_DIR/scenes"
CAMERA="/media/syslab/Insta360 X5/DCIM/Camera01"

mkdir -p "$DATA_DIR" "$SCENES_DIR"

# Selected .insv files (diverse dates, manageable sizes)
FILES=(
    "VID_20261116_201732_00_131.insv"   # Today - already processed
    "VID_20261113_132445_00_118.insv"   # Nov 13 - 332M
    "VID_20261004_093449_00_093.insv"   # Oct 4  - 979M
    "VID_20261008_172201_00_112.insv"   # Oct 8  - 570M
    "VID_20260713_163557_00_086.insv"   # Jul 13 - 216M
    "VID_20260426_190831_00_066.insv"   # Apr 26 - 865M
    "VID_20260426_163312_00_057.insv"   # Apr 26 - 675M
)

NAMES=(
    "scene_today"
    "scene_nov13"
    "scene_oct4a"
    "scene_oct8"
    "scene_jul13"
    "scene_apr26a"
    "scene_apr26b"
)

for i in "${!FILES[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing ${i+1}/${#FILES[@]}: ${FILES[$i]}"
    echo "========================================"

    SRC="$CAMERA/${FILES[$i]}"
    NAME="${NAMES[$i]}"

    if [ ! -f "$SRC" ]; then
        echo "SKIP: $SRC not found"
        continue
    fi

    # Skip if web PLY already exists
    if [ -f "$SCENES_DIR/${NAME}.ply" ]; then
        echo "SKIP: $SCENES_DIR/${NAME}.ply already exists"
        continue
    fi

    # Copy to data dir
    if [ ! -f "$DATA_DIR/$(basename "$SRC")" ]; then
        echo "Copying from camera..."
        cp "$SRC" "$DATA_DIR/"
    fi

    # Process to PLY (web-optimized ~4MB each)
    python3 "$SCRIPT_DIR/make_ply_insv.py" \
        "$DATA_DIR/$(basename "$SRC")" \
        --output "$SCENES_DIR/${NAME}_full.ply" \
        --high --web

    # Move web version to scenes
    if [ -f "$SCENES_DIR/${NAME}_full_web.ply" ]; then
        mv "$SCENES_DIR/${NAME}_full_web.ply" "$SCENES_DIR/${NAME}.ply"
    fi

    echo "Done: ${NAME}"
done

echo ""
echo "========================================"
echo "All scenes processed!"
echo "========================================"
ls -lh "$SCENES_DIR/"*.ply 2>/dev/null
