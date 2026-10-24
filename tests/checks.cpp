#include "stx_bernstein.h"
#include "reference/surface_fit.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    stx_bernstein_destroy(nullptr);
    if (stx_bernstein_create(0, 1) != nullptr || stx_bernstein_create(2049, 1) != nullptr ||
        stx_bernstein_create(2048, 2048) != nullptr) {
        return 1;
    }
    constexpr std::array<std::size_t, 4> widths{1, 1, 13, 31};
    constexpr std::array<std::size_t, 4> heights{1, 19, 1, 17};
    for (std::size_t fixture = 0; fixture < widths.size(); ++fixture) {
        const std::size_t width = widths[fixture];
        const std::size_t height = heights[fixture];
        stx_bernstein_workspace* work = stx_bernstein_create(width, height);
        if (work == nullptr) {
            return 2;
        }
        std::vector<std::uint8_t> rgb(width * height * 3);
        std::vector<std::uint8_t> rendered(rgb.size());
        for (std::size_t index = 0; index < width * height; ++index) {
            rgb[index * 3] = 0;
            rgb[index * 3 + 1] = 113;
            rgb[index * 3 + 2] = 255;
        }
        std::array<std::uint16_t, 243> controls{};
        controls.fill(42);
        stx_bernstein_stats stats{};
        stats.workspace_bytes = 123;
        int status = stx_bernstein_fit(work, rgb.data(), rgb.size() - 1, controls.data(), controls.size(), &stats);
        if (status != 1 || stats.workspace_bytes != 123 || controls[0] != 42) {
            stx_bernstein_destroy(work);
            return 3;
        }
        status = stx_bernstein_fit(work, rgb.data(), rgb.size(), controls.data(), controls.size(), &stats);
        if (status != 0 || !std::isfinite(stats.final_cost) || stats.final_cost > stats.initial_cost) {
            stx_bernstein_destroy(work);
            return 4;
        }
        status = stx_surface_fit_render(controls.data(), controls.size(), width, height, rendered.data(), rendered.size());
        if (status != 0 || rendered != rgb) {
            stx_bernstein_destroy(work);
            return 5;
        }
        const std::array<std::uint16_t, 243> first = controls;
        status = stx_bernstein_fit(work, rgb.data(), rgb.size(), controls.data(), controls.size(), &stats);
        stx_bernstein_destroy(work);
        if (status != 0 || first != controls) {
            return 6;
        }
    }
    // Exercise the maximum admitted pixel count and active-bound fitting,
    // then reuse that workspace for a different image.
    constexpr std::size_t width = 1024;
    constexpr std::size_t height = 512;
    stx_bernstein_workspace* large = stx_bernstein_create(width, height);
    if (large == nullptr) {
        return 7;
    }
    std::vector<std::uint8_t> image(width * height * 3);
    std::array<std::uint16_t, 243> output{};
    for (std::size_t pass = 0; pass < 2; ++pass) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t pixel = y * width + x;
                image[pixel * 3] = x < y + pass * 173 ? 0 : 255;
                image[pixel * 3 + 1] = static_cast<std::uint8_t>((x + y + pass * 39) & 255);
                image[pixel * 3 + 2] = static_cast<std::uint8_t>((x * y + pass * 71) & 255);
            }
        }
        stx_bernstein_stats stats{};
        const int status = stx_bernstein_fit(large, image.data(), image.size(), output.data(), output.size(), &stats);
        if (status != 0 || !std::isfinite(stats.final_cost) || stats.final_cost > stats.initial_cost || stats.accepted_steps == 0) {
            stx_bernstein_destroy(large);
            return 8;
        }
        for (const std::uint16_t value : output) {
            if (value > 65280) {
                stx_bernstein_destroy(large);
                return 9;
            }
        }
    }
    stx_bernstein_destroy(large);
    stx_bernstein_workspace* signed_work = stx_bernstein_create(13, 7);
    if (signed_work == nullptr) {
        return 10;
    }
    std::vector<std::int16_t> signed_input(13 * 7 * 3);
    std::vector<std::int16_t> signed_output(signed_input.size());
    for (std::size_t pixel = 0; pixel < 13 * 7; ++pixel) {
        signed_input[pixel * 3] = -255;
        signed_input[pixel * 3 + 1] = -113;
        signed_input[pixel * 3 + 2] = 255;
    }
    std::array<std::int32_t, 243> signed_controls{};
    stx_bernstein_stats signed_stats{};
    int signed_status = stx_bernstein_fit_signed(signed_work, signed_input.data(), signed_input.size(),
        signed_controls.data(), signed_controls.size(), &signed_stats);
    if (signed_status != 0) {
        stx_bernstein_destroy(signed_work);
        return 11;
    }
    signed_status = stx_bernstein_render_signed(signed_work, signed_controls.data(), signed_controls.size(),
        signed_output.data(), signed_output.size());
    if (signed_status != 0 || signed_output != signed_input) {
        stx_bernstein_destroy(signed_work);
        return 12;
    }
    signed_input[0] = -256;
    signed_controls.fill(42);
    signed_status = stx_bernstein_fit_signed(signed_work, signed_input.data(), signed_input.size(),
        signed_controls.data(), signed_controls.size(), &signed_stats);
    if (signed_status != 1 || signed_controls[0] != 42) {
        stx_bernstein_destroy(signed_work);
        return 13;
    }
    signed_controls.fill(-384); // -1.5 rounds toward positive infinity to -1.
    signed_status = stx_bernstein_render_signed(signed_work, signed_controls.data(), signed_controls.size(),
        signed_output.data(), signed_output.size());
    stx_bernstein_destroy(signed_work);
    if (signed_status != 0) {
        return 14;
    }
    for (const std::int16_t value : signed_output) {
        if (value != -1) {
            return 15;
        }
    }
    std::puts("PASS");
    return 0;
}
