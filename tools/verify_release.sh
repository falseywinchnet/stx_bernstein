#!/bin/sh
set -eu
cmake --build .build/release -j 4
ctest --test-dir .build/release --output-on-failure
cmake --install .build/release --prefix "$PWD/.build/prefix"
cmake -S tests/installed -B .build/consumer -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@22/bin/clang++ -DCMAKE_PREFIX_PATH="$PWD/.build/prefix"
cmake --build .build/consumer -j 4
ctest --test-dir .build/consumer --output-on-failure
