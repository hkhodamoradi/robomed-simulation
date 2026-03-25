#!/usr/bin/env bash
set -e

ROOT="$(pwd)"

echo "Creating folder structure under: $ROOT"

mkdir -p "$ROOT/controllers/unified_med_robot/behaviors"
mkdir -p "$ROOT/controllers/unified_med_robot/perception"
mkdir -p "$ROOT/controllers/unified_med_robot/utils"
mkdir -p "$ROOT/controllers/unified_med_robot/runtime/logs"
mkdir -p "$ROOT/controllers/unified_med_robot/runtime/snapshots"
mkdir -p "$ROOT/controllers/unified_med_robot/runtime/exports"

mkdir -p "$ROOT/worlds"
mkdir -p "$ROOT/assets/monitor_images"
mkdir -p "$ROOT/assets/bed_labels"
mkdir -p "$ROOT/docs"
mkdir -p "$ROOT/videos/demo"
mkdir -p "$ROOT/agent"

echo "Done."