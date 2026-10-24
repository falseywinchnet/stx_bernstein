#!/bin/sh
set -u
build="$1"
record="$2"
mkdir -p "$record"
if ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$build" --output-on-failure > "$record/diagnostics.txt" 2>&1; then
    printf 'PASS\n' | tee "$record/status.txt"
else
    printf 'STOP\n' | tee "$record/status.txt"
    exit 1
fi
