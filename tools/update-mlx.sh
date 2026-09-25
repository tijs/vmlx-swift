#!/bin/bash
# Regenerate only derived sources from the pinned core and C ABI.
# No CMake model/core build, submodule checkout, or unrelated build deletion.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORE_DIR="$PWD/Source/Cmlx/mlx"
GENERATED_DIR="$PWD/Source/Cmlx/mlx-generated"
# NAX header discovery needs Metal 4 and SDK 26.2+. This does not change
# the app's deployment target; runtime device/version gates still apply.
xcrun -sdk macosx metal --version
TASK_GENERATED=$(mktemp -d "${TMPDIR:-/tmp}/mlx-generated.XXXXXX")
trap 'rm -rf "$TASK_GENERATED"' EXIT
CXX_COMPILER=$(xcrun -find clang++)
JIT_SOURCES=$(perl -0777 -ne 'while (/\bmake_jit_source\(\s*([A-Za-z0-9_\/]+)/g) { print "$1\n" }' "$CORE_DIR/mlx/backend/metal/CMakeLists.txt")
[ -n "$JIT_SOURCES" ]
for source in $JIT_SOURCES; do
    bash "$CORE_DIR/mlx/backend/metal/make_compiled_preamble.sh" \
        "$TASK_GENERATED" "$CXX_COMPILER" "$CORE_DIR" "$source" \
        "-std=metal4.0 -mmacosx-version-min=26.2"
    [ -s "$TASK_GENERATED/${source##*/}.cpp" ]
done
bash "$CORE_DIR/mlx/backend/cpu/make_compiled_preamble.sh" \
    "$TASK_GENERATED/compiled_preamble.cpp" "$CXX_COMPILER" "$CORE_DIR" TRUE "$(uname -m)" ""
[ -s "$TASK_GENERATED/compiled_preamble.cpp" ]
# Copy only generated targets. Preserve any hand-maintained bridge files.
mkdir -p "$GENERATED_DIR" Source/Cmlx/include/mlx/c
for source in "$TASK_GENERATED"/*.cpp; do
    destination="$GENERATED_DIR/${source##*/}"
    cmp -s "$source" "$destination" || cp "$source" "$destination"
done
for source in Source/Cmlx/mlx-c/mlx/c/*.h; do
    destination="Source/Cmlx/include/mlx/c/${source##*/}"
    cmp -s "$source" "$destination" || cp "$source" "$destination"
done
# AOT gemv was replaced by the JIT target in 0.32. Remove only this obsolete
# generated shader, otherwise it can conflict with the new dot kernels.
rm -f "$GENERATED_DIR/metal/gemv.metal"
rm -f "$GENERATED_DIR/metal/fence.metal"
./tools/fix-metal-includes.sh
./tools/update-mlx-xcodeproj.sh
swift tools/update-xcode-membership.swift
printf 'Generated Metal JIT targets: %s\n' "$(printf '%s\n' "$JIT_SOURCES" | wc -l | tr -d ' ')"
