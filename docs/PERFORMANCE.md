# Measurements and limits

Measured October 23, 2026. LLVM 22.1.8, release optimization, FP contraction off.
The latest implementation is compared with its preceding installed version.
Complete CPU jobs include copying input, fitting, integer reconstruction,
RGBA packing, worker scheduling and publication. They exclude file I/O, graphics
capture, readback, upload, blending and video-frame scheduling.

Apple A18 Pro, macOS 26.7.1; three alternating batches per implementation,
two warmups and seven measured jobs per batch, 21 samples per result:

| Image | Previous median | Current median | Current CPU jobs/s |
| --- | ---: | ---: | ---: |
| Combined 1024×512 | 70.002 ms | 65.653 ms | 15.23 |
| Combined 640×320 | 24.819 ms | 23.136 ms | 43.22 |
| Smooth field 512×512 | 88.679 ms | 64.585 ms | 15.48 |

All three compared outputs were byte-identical. The 640×320 case is a fixed
bilinear resize. Raw sample times and input hashes are in `worker-a18.json`.
M4 Mini measurements in `worker-m4.json` show 88.389→84.942 ms,
31.732→29.710 ms and 97.920→76.336 ms respectively.

At 1024×512, owned solver payload is 4,337,232 bytes, excluding caller/worker
buffers, graphics storage, allocator overhead and bounded stack scratch.
Deferred curvature adds 184,880 bytes versus the preceding implementation.

A separate ten-image fit comparison retained all controls and quality metrics.
Some small fixtures regressed in timing. Host scheduling varies; these results
are not a universal speedup, a live-video frame-rate guarantee, or evidence that
the original 10× research target has been reached. The private research-image
corpus and OBS harness are not distributed in this core package; the checked-in
numbers are historical evidence, not a self-contained reproducibility claim.

The extracted package includes deterministic API, bounds, workspace reuse and
signed-render tests. The originating implementation also passed focused
ASan/UBSan checks and actual OBS Metal/OpenGL integration checks. Runtime
verification so far is on macOS arm64; x86-64/SSE2 assembly was inspected but
not executed. Other platforms have not been validated.

Publication-package verification on the M4 Mini with LLVM 22.1.8: release tests,
installed CMake consumer, and the extracted library's focused ASan/UBSan suite
all passed. The instrumented runner exposed PASS only. Style and lifetime
spelling checks reported no findings.

## Current 289-moment update

See [MOMENT_REDUCTION.md](MOMENT_REDUCTION.md) for the current implementation,
new paired measurements and explicit numerical differences. The previous tables
above remain historical evidence. The 289-moment package passed its release
suite, installed-consumer test and focused ASan/UBSan check (PASS). Actual OBS
Metal and OpenGL tests passed in the independent plugin project.
