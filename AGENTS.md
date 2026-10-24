# stx_bernstein

Preserve the explicit C API ownership, dimensions, fixed bounds and allocation-free
prepared-fit contract. No auto, lambdas, arrow member access or fast-math in
first-party C++. Preserve numerical operation order unless the change explicitly
measures its effects. Run style/lifetime tools over include, src and tests.

Build locally with the README commands. For the owner's M4 build host, run from
this repository using the configured m4build helper:

    /Users/ultimussecundai/.local/bin/m4build -- /opt/homebrew/bin/cmake -S . -B .build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@22/bin/clang++
    /Users/ultimussecundai/.local/bin/m4build -- /opt/homebrew/bin/cmake --build .build/release -j 4
    /Users/ultimussecundai/.local/bin/m4build -- /opt/homebrew/bin/ctest --test-dir .build/release --output-on-failure

Keep .build/sanitize beside release, configured with STX_BERNSTEIN_SANITIZE=ON.
Use tools/check_sanitized.sh BUILD RECORD for focused instrumented checks;
it reports only PASS or STOP. Leave detailed reports for the owner's specialist
on STOP. Do not launch a separate randomized review without a request.
