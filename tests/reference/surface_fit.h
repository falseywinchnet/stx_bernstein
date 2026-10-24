#ifndef STX_CNG_SURFACE_FIT_H
#define STX_CNG_SURFACE_FIT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct stx_surface_fit_stats {
    size_t admitted_pixels;
    size_t initial_mismatches;
    size_t final_mismatches;
    size_t workspace_bytes;
    size_t iterations;
} stx_surface_fit_stats;

/* Experimental degree-eight RGB refinement, not a public CNG format API.
   Input is tightly packed RGB8. Initial/output controls are 9*9*3 Q8 values,
   interleaved by channel, each at most 65280. Extents are explicit element
   counts. Dimensions are 1..2048 with at most 524288 pixels. All buffers are
   disjoint and live through the call; no pointer is retained. Storage is owned
   by this call, allocated before iteration, and released on return.
   Return 0 on success, 1 on invalid arguments, 2 on allocation failure, 3 on
   other internal failure. Output and statistics are unchanged on failure.
   No exception crosses this boundary. Workspace bytes counts owned payload,
   including the fixed workspace object, excluding allocator/runtime overhead
   and bounded procedure-local scratch (approximately another 60 KiB). */
int stx_surface_fit_refine(const uint8_t* rgb, size_t rgb_size,
    size_t width, size_t height, const uint16_t* initial, size_t initial_count,
    uint16_t* output, size_t output_count, stx_surface_fit_stats* stats);

/* Cold sampled initializer: bounded least-squares, Huber and interval stages.
   Same image/control bounds and failure contract as refinement. The optimizer
   is bounded Newton, not the research script's SciPy L-BFGS-B implementation. */
int stx_surface_fit_initialize(const uint8_t* rgb, size_t rgb_size,
    size_t width, size_t height, uint16_t* output, size_t output_count);

/* Integer decoder surface; writes width*height*3 RGB8 samples. */
int stx_surface_fit_render(const uint16_t* controls, size_t control_count,
    size_t width, size_t height, uint8_t* rgb, size_t rgb_size);

#ifdef __cplusplus
}
#endif
#endif
