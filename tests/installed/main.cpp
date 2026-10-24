#include <stx_bernstein.h>
#include <array>
#include <cstdint>

int main() {
    stx_bernstein_workspace* workspace = stx_bernstein_create(1, 1);
    if (workspace == nullptr) return 1;
    const std::array<std::uint8_t, 3> pixel{37, 113, 210};
    std::array<std::uint16_t, 243> controls{};
    stx_bernstein_stats stats{};
    const int status = stx_bernstein_fit(workspace, pixel.data(), pixel.size(),
        controls.data(), controls.size(), &stats);
    stx_bernstein_destroy(workspace);
    if (status != 0) return 2;
    return stats.final_cost == 0.0 ? 0 : 3;
}
