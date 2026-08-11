#!/usr/bin/env bash
set -euo pipefail
d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; c="${CXX:-g++}"; v="$($c --version|head -1)"
common=(-std=c++17 -O3 -g -fno-unroll-loops -march=native)
if [[ "$v" == *GCC* || "$v" == *g++* ]];then extra=(-fno-if-conversion -fno-if-conversion2 -fno-tree-vectorize);else extra=(-fno-vectorize -fno-slp-vectorize);echo "warning: final Linux experiment requires GCC; local syntax build only" >&2;fi
"$c" "${common[@]}" "${extra[@]}" "$d/maxpool_benchmark.cpp" -o "$d/maxpool_benchmark"
echo "built $d/maxpool_benchmark; run verify_disassembly.py before PMU measurement"
