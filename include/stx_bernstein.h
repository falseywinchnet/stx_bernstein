#ifndef STX_BERNSTEIN_H
#define STX_BERNSTEIN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct stx_bernstein_workspace stx_bernstein_workspace;
typedef struct stx_bernstein_stats {
    size_t workspace_bytes;
    size_t accepted_steps;
    size_t bound_solves;
    size_t fallback_directions;
    double initial_cost;
    double final_cost;
} stx_bernstein_stats;

/* Experimental C API. Degree eight, all pixels, RGB8 input and 243 interleaved
   Q8 controls in [0,65280]. Dimensions: 1..2048, at most 524288 pixels.
   Create owns all working storage, including linear-system scratch; null means
   invalid dimensions or allocation failure. Destroy accepts null. No mutable global state.
   Fit reuses the prepared workspace with no allocations in repeated processing.
   One caller at a time per workspace. Input and output are disjoint; all caller
   buffers remain live through the call and no address is retained.
   Fit returns 0 success, 1 invalid arguments, 3 numerical failure. Output and
   statistics are unchanged on failure. Sixteen updates/channel, eight line
   trials/update; no image sampling. The solver permits 243 coefficient-space
   face transitions/direction and stops early when the convex objective-gap
   estimate is at most max(1e-8, 1e-10 * cost). Floating-point reduction order is
   explicit; results are not promised to be identical across compilers or
   architectures.
   workspace_bytes is owned object plus vector capacities, excluding allocator
   overhead and bounded call-stack locals (< 8 KiB). */
stx_bernstein_workspace* stx_bernstein_create(size_t width, size_t height);
void stx_bernstein_destroy(stx_bernstein_workspace* workspace);
int stx_bernstein_fit(stx_bernstein_workspace* workspace,
    const uint8_t* rgb, size_t rgb_size, uint16_t* controls,
    size_t control_count, stx_bernstein_stats* stats);
/* Signed residual extension: tightly packed int16 RGB values in [-255,255],
   with 243 int32 Q8 controls in [-65280,65280]. Same ownership, iteration and
   failure contracts. Render writes signed int16 samples using half-up rounding
   (ties toward positive infinity). Counts are elements, not bytes. */
int stx_bernstein_fit_signed(stx_bernstein_workspace* workspace,
    const int16_t* residual, size_t residual_count, int32_t* controls,
    size_t control_count, stx_bernstein_stats* stats);
int stx_bernstein_render_signed(stx_bernstein_workspace* workspace,
    const int32_t* controls, size_t control_count,
    int16_t* output, size_t output_count);
#ifdef __cplusplus
}
#endif
#endif
