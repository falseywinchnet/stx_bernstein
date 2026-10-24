#include "surface_fit.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace {
constexpr std::size_t side = 9;
constexpr std::size_t controls_count = side * side;
constexpr std::size_t pair_count = 45;
constexpr std::uint64_t weight = 1048576;
constexpr std::uint64_t denominator = 256 * weight * weight;

struct Axis {
    std::vector<std::uint32_t> integer;
    std::vector<double> values;
    std::vector<double> pairs;
};

void prepare_axis(std::size_t length, Axis& axis) {
    axis.integer.resize(length * side);
    axis.values.resize(length * side);
    axis.pairs.resize(length * pair_count);
    const std::size_t extent = std::max(length - 1, std::size_t{1});
    for (std::size_t position = 0; position < length; ++position) {
        std::array<std::uint64_t, side> current{};
        current[0] = weight;
        for (std::size_t degree = 1; degree < side; ++degree) {
            std::array<std::uint64_t, side> next{};
            for (std::size_t index = 0; index < degree; ++index) {
                const std::uint64_t right = current[index] * position / extent;
                next[index] += current[index] - right;
                next[index + 1] += right;
            }
            current = next;
        }
        for (std::size_t index = 0; index < side; ++index) {
            axis.integer[position * side + index] = static_cast<std::uint32_t>(current[index]);
            axis.values[position * side + index] = static_cast<double>(current[index]) / static_cast<double>(weight);
        }
        std::size_t pair = 0;
        for (std::size_t first = 0; first < side; ++first) {
            for (std::size_t second = first; second < side; ++second) {
                axis.pairs[position * pair_count + pair] = axis.values[position * side + first] * axis.values[position * side + second];
                ++pair;
            }
        }
    }
}

struct Workspace {
    std::size_t width;
    std::size_t height;
    Axis x;
    Axis y;
    std::vector<double> horizontal;
    std::vector<std::uint64_t> integer_horizontal;
    std::vector<double> prediction;
    std::vector<std::uint8_t> admitted;
    std::vector<std::uint8_t> outlier;
    std::vector<double> row_gradient;
    std::vector<double> row_pairs;
    std::array<double, controls_count> gradient{};
    std::array<double, controls_count> majorizer{};
    std::array<double, controls_count * controls_count> hessian{};
    std::array<double, pair_count * pair_count> moments{};
    std::array<std::size_t, controls_count> pair_lookup{};

    Workspace(std::size_t input_width, std::size_t input_height)
        : width(input_width), height(input_height), x{}, y{},
          horizontal(side * width), integer_horizontal(side * width),
          prediction(width * height), admitted(width * height, 1),
          outlier(width * height), row_gradient(height * side),
          row_pairs(height * pair_count) {
        prepare_axis(width, x);
        prepare_axis(height, y);
        std::size_t pair = 0;
        for (std::size_t first = 0; first < side; ++first) {
            for (std::size_t second = first; second < side; ++second) {
                pair_lookup[first * side + second] = pair;
                pair_lookup[second * side + first] = pair;
                ++pair;
            }
        }
    }

    std::size_t payload_bytes() const {
        std::size_t bytes = sizeof(Workspace);
        bytes += (x.integer.capacity() + y.integer.capacity()) * sizeof(std::uint32_t);
        bytes += (x.values.capacity() + y.values.capacity() + x.pairs.capacity() + y.pairs.capacity()) * sizeof(double);
        bytes += (horizontal.capacity() + prediction.capacity() + row_gradient.capacity() + row_pairs.capacity()) * sizeof(double);
        bytes += integer_horizontal.capacity() * sizeof(std::uint64_t);
        bytes += admitted.capacity() + outlier.capacity();
        return bytes;
    }
};

void render(Workspace& work, const std::array<double, controls_count>& controls) {
    std::fill(work.horizontal.begin(), work.horizontal.end(), 0.0);
    for (std::size_t row = 0; row < side; ++row) {
        for (std::size_t column = 0; column < side; ++column) {
            const double coefficient = controls[row * side + column];
            for (std::size_t x = 0; x < work.width; ++x) {
                work.horizontal[row * work.width + x] += coefficient * work.x.values[x * side + column];
            }
        }
    }
    std::fill(work.prediction.begin(), work.prediction.end(), 0.0);
    for (std::size_t y = 0; y < work.height; ++y) {
        for (std::size_t row = 0; row < side; ++row) {
            const double coefficient = work.y.values[y * side + row];
            for (std::size_t x = 0; x < work.width; ++x) {
                work.prediction[y * work.width + x] += coefficient * work.horizontal[row * work.width + x];
            }
        }
    }
}

// Exact integer contraction shares the decoder's mass-preserving basis and
// single half-up rounding. 65280*2^40 fits in uint64_t.
std::size_t check_integer(Workspace& work, const std::uint8_t* rgb,
    const std::array<std::uint16_t, controls_count>& controls,
    std::size_t channel, bool collect_outliers) {
    std::fill(work.integer_horizontal.begin(), work.integer_horizontal.end(), std::uint64_t{0});
    for (std::size_t row = 0; row < side; ++row) {
        for (std::size_t column = 0; column < side; ++column) {
            const std::uint64_t coefficient = controls[row * side + column];
            for (std::size_t x = 0; x < work.width; ++x) {
                work.integer_horizontal[row * work.width + x] += coefficient * work.x.integer[x * side + column];
            }
        }
    }
    std::size_t count = 0;
    for (std::size_t y = 0; y < work.height; ++y) {
        for (std::size_t x = 0; x < work.width; ++x) {
            std::uint64_t numerator = 0;
            for (std::size_t row = 0; row < side; ++row) {
                numerator += work.y.integer[y * side + row] * work.integer_horizontal[row * work.width + x];
            }
            const int predicted = static_cast<int>((numerator + denominator / 2) / denominator);
            const std::size_t index = y * work.width + x;
            const int difference = predicted - static_cast<int>(rgb[index * 3 + channel]);
            if (work.admitted[index] != 0 && difference != 0) {
                ++count;
            }
            if (collect_outliers && std::abs(difference) > 1) {
                work.outlier[index] = 1;
            }
        }
    }
    return count;
}

void prepare_admission(Workspace& work, const std::uint8_t* rgb, const std::uint16_t* initial) {
    for (std::size_t channel = 0; channel < 3; ++channel) {
        std::array<std::uint16_t, controls_count> controls{};
        for (std::size_t index = 0; index < controls_count; ++index) {
            controls[index] = initial[index * 3 + channel];
        }
        check_integer(work, rgb, controls, channel, true);
    }
    for (std::size_t y = 0; y < work.height; ++y) {
        for (std::size_t x = 0; x < work.width; ++x) {
            if (work.outlier[y * work.width + x] == 0) {
                continue;
            }
            const std::size_t begin_y = y == 0 ? 0 : y - 1;
            const std::size_t begin_x = x == 0 ? 0 : x - 1;
            const std::size_t end_y = std::min(work.height, y + 2);
            const std::size_t end_x = std::min(work.width, x + 2);
            for (std::size_t target_y = begin_y; target_y < end_y; ++target_y) {
                for (std::size_t target_x = begin_x; target_x < end_x; ++target_x) {
                    work.admitted[target_y * work.width + target_x] = 0;
                }
            }
        }
    }
    std::fill(work.row_gradient.begin(), work.row_gradient.end(), 0.0);
    for (std::size_t y = 0; y < work.height; ++y) {
        for (std::size_t x = 0; x < work.width; ++x) {
            if (work.admitted[y * work.width + x] == 0) {
                continue;
            }
            for (std::size_t column = 0; column < side; ++column) {
                work.row_gradient[y * side + column] += work.x.values[x * side + column];
            }
        }
        for (std::size_t row = 0; row < side; ++row) {
            for (std::size_t column = 0; column < side; ++column) {
                work.majorizer[row * side + column] += work.y.values[y * side + row] * work.row_gradient[y * side + column];
            }
        }
    }
}

enum class Loss { interval, squared, huber };

template<bool collect_system>
double evaluate(Workspace& work, const std::uint8_t* rgb, std::size_t channel,
    const std::array<double, controls_count>& controls, Loss loss = Loss::interval,
    double radius = 0.499) {
    render(work, controls);
    if constexpr (collect_system) {
        std::fill(work.row_gradient.begin(), work.row_gradient.end(), 0.0);
        std::fill(work.row_pairs.begin(), work.row_pairs.end(), 0.0);
        work.gradient.fill(0.0);
        work.moments.fill(0.0);
    }
    double cost = 0.0;
    for (std::size_t y = 0; y < work.height; ++y) {
        for (std::size_t x = 0; x < work.width; ++x) {
            const std::size_t index = y * work.width + x;
            if (work.admitted[index] == 0) {
                continue;
            }
            const double error = work.prediction[index] - static_cast<double>(rgb[index * 3 + channel]);
            double residual = error;
            double contribution = error * error;
            bool curved = true;
            double curvature = 1.0;
            switch (loss) {
            case Loss::squared: break;
            case Loss::huber:
                residual = std::clamp(error, -2.0, 2.0);
                contribution = std::abs(error) <= 2.0 ? error * error : 4.0 * std::abs(error) - 4.0;
                curvature = std::abs(error) <= 2.0 ? 1.0 : 2.0 / std::abs(error);
                break;
            default:
                if (std::abs(error) <= radius) {
                    continue;
                }
                residual = error - std::copysign(radius, error);
                contribution = residual * residual;
                break;
            }
            cost += contribution;
            if constexpr (collect_system) {
                for (std::size_t column = 0; column < side; ++column) {
                    work.row_gradient[y * side + column] += residual * work.x.values[x * side + column];
                }
                if (curved) {
                    for (std::size_t pair = 0; pair < pair_count; ++pair) {
                        work.row_pairs[y * pair_count + pair] += curvature * work.x.pairs[x * pair_count + pair];
                    }
                }
            }
        }
    }
    if constexpr (collect_system) {
        for (std::size_t y = 0; y < work.height; ++y) {
            for (std::size_t row = 0; row < side; ++row) {
                const double coefficient = work.y.values[y * side + row];
                for (std::size_t column = 0; column < side; ++column) {
                    work.gradient[row * side + column] += coefficient * work.row_gradient[y * side + column];
                }
            }
            for (std::size_t py = 0; py < pair_count; ++py) {
                const double coefficient = work.y.pairs[y * pair_count + py];
                for (std::size_t px = 0; px < pair_count; ++px) {
                    work.moments[py * pair_count + px] += coefficient * work.row_pairs[y * pair_count + px];
                }
            }
        }
        for (std::size_t first = 0; first < controls_count; ++first) {
            for (std::size_t second = 0; second < controls_count; ++second) {
                const std::size_t py = work.pair_lookup[(first / side) * side + second / side];
                const std::size_t px = work.pair_lookup[(first % side) * side + second % side];
                work.hessian[first * controls_count + second] = work.moments[py * pair_count + px];
            }
        }
    }
    cost *= 0.5;
    return cost;
}

bool solve_direction(const Workspace& work, std::array<double, controls_count>& direction) {
    std::array<double, controls_count * controls_count> matrix{};
    std::array<double, controls_count> scale{};
    for (std::size_t row = 0; row < controls_count; ++row) {
        scale[row] = std::sqrt(std::max(work.hessian[row * controls_count + row], 1e-20));
        direction[row] = -work.gradient[row] / scale[row];
    }
    for (std::size_t row = 0; row < controls_count; ++row) {
        for (std::size_t column = 0; column < controls_count; ++column) {
            matrix[row * controls_count + column] = work.hessian[row * controls_count + column] / scale[row] / scale[column];
        }
        matrix[row * controls_count + row] += 1e-9;
    }
    // Partial-pivot Gaussian elimination. No LAPACK/runtime dependency.
    for (std::size_t pivot = 0; pivot < controls_count; ++pivot) {
        std::size_t selected = pivot;
        for (std::size_t row = pivot + 1; row < controls_count; ++row) {
            if (std::abs(matrix[row * controls_count + pivot]) > std::abs(matrix[selected * controls_count + pivot])) {
                selected = row;
            }
        }
        const double magnitude = std::abs(matrix[selected * controls_count + pivot]);
        if (!(magnitude > 1e-30) || !std::isfinite(magnitude)) {
            return false;
        }
        if (selected != pivot) {
            for (std::size_t column = pivot; column < controls_count; ++column) {
                std::swap(matrix[selected * controls_count + column], matrix[pivot * controls_count + column]);
            }
            std::swap(direction[selected], direction[pivot]);
        }
        for (std::size_t row = pivot + 1; row < controls_count; ++row) {
            const double factor = matrix[row * controls_count + pivot] / matrix[pivot * controls_count + pivot];
            for (std::size_t column = pivot + 1; column < controls_count; ++column) {
                matrix[row * controls_count + column] -= factor * matrix[pivot * controls_count + column];
            }
            direction[row] -= factor * direction[pivot];
        }
    }
    for (std::size_t end = controls_count; end > 0; --end) {
        const std::size_t row = end - 1;
        for (std::size_t column = row + 1; column < controls_count; ++column) {
            direction[row] -= matrix[row * controls_count + column] * direction[column];
        }
        direction[row] /= matrix[row * controls_count + row];
    }
    for (std::size_t row = 0; row < controls_count; ++row) {
        direction[row] /= scale[row];
        if (!std::isfinite(direction[row])) {
            return false;
        }
    }
    return true;
}

std::uint16_t quantize(double value) {
    const double scaled = value * 256.0;
    const double floor_value = std::floor(scaled);
    std::uint32_t integer = static_cast<std::uint32_t>(floor_value);
    const double fraction = scaled - floor_value;
    if (fraction > 0.5 || (fraction == 0.5 && (integer & 1u) != 0)) {
        ++integer;
    }
    return static_cast<std::uint16_t>(integer);
}

void continuous_fit(Workspace& work, const std::uint8_t* rgb, std::size_t channel,
    Loss loss, std::size_t iterations, std::array<double, controls_count>& controls) {
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        const double cost = evaluate<true>(work, rgb, channel, controls, loss, 0.49);
        if (cost == 0.0) { break; }
        for (std::size_t index = 0; index < controls_count; ++index) {
            const bool lower = controls[index] <= 0.0 && work.gradient[index] > 0.0;
            const bool upper = controls[index] >= 255.0 && work.gradient[index] < 0.0;
            if (lower || upper) {
                for (std::size_t other = 0; other < controls_count; ++other) {
                    work.hessian[index * controls_count + other] = 0.0;
                    work.hessian[other * controls_count + index] = 0.0;
                }
                work.hessian[index * controls_count + index] = 1.0;
                work.gradient[index] = 0.0;
            }
        }
        std::array<double, controls_count> direction{};
        const bool solved = solve_direction(work, direction);
        if (!solved) { break; }
        bool accepted = false;
        double factor = 1.0;
        for (std::size_t trial = 0; trial < 16; ++trial) {
            std::array<double, controls_count> candidate{};
            for (std::size_t index = 0; index < controls_count; ++index) {
                candidate[index] = std::clamp(controls[index] + factor * direction[index], 0.0, 255.0);
            }
            const double next_cost = evaluate<false>(work, rgb, channel, candidate, loss, 0.49);
            if (next_cost < cost) {
                controls = candidate;
                accepted = true;
                break;
            }
            factor *= 0.5;
        }
        if (!accepted) { break; }
    }
}

std::vector<std::size_t> sample_positions(std::size_t length, std::size_t stride) {
    std::vector<std::size_t> positions;
    positions.reserve(length / stride + 2);
    for (std::size_t index = 0; index < length; index += stride) { positions.push_back(index); }
    if (positions.back() != length - 1) { positions.push_back(length - 1); }
    return positions;
}

void select_axis(const Axis& full, const std::vector<std::size_t>& positions, Axis& selected) {
    for (std::size_t index = 0; index < positions.size(); ++index) {
        for (std::size_t column = 0; column < side; ++column) {
            selected.integer[index * side + column] = full.integer[positions[index] * side + column];
            selected.values[index * side + column] = full.values[positions[index] * side + column];
        }
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            selected.pairs[index * pair_count + pair] = full.pairs[positions[index] * pair_count + pair];
        }
    }
}

void refine_channel(Workspace& work, const std::uint8_t* rgb,
    const std::uint16_t* initial, std::size_t channel,
    std::array<std::uint16_t, controls_count * 3>& output, stx_surface_fit_stats& stats) {
    std::array<double, controls_count> controls{};
    std::array<std::uint16_t, controls_count> best{};
    for (std::size_t index = 0; index < controls_count; ++index) {
        best[index] = initial[index * 3 + channel];
        controls[index] = static_cast<double>(best[index]) / 256.0;
    }
    std::size_t best_count = check_integer(work, rgb, best, channel, false);
    stats.initial_mismatches += best_count;
    for (std::size_t iteration = 0; iteration < 8; ++iteration) {
        ++stats.iterations;
        const double cost = evaluate<true>(work, rgb, channel, controls);
        std::array<double, controls_count> direction{};
        const bool solved = solve_direction(work, direction);
        bool accepted = false;
        std::array<double, controls_count> candidate{};
        if (solved) {
            double factor = 1.0;
            for (std::size_t trial = 0; trial < 9; ++trial) {
                for (std::size_t index = 0; index < controls_count; ++index) {
                    candidate[index] = std::clamp(controls[index] + factor * direction[index], 0.0, 255.0);
                }
                const double candidate_cost = evaluate<false>(work, rgb, channel, candidate);
                if (candidate_cost < cost) {
                    controls = candidate;
                    accepted = true;
                    break;
                }
                factor *= 0.5;
            }
        }
        if (!accepted) {
            for (std::size_t index = 0; index < controls_count; ++index) {
                const double step = work.gradient[index] / std::max(work.majorizer[index], 1e-20);
                candidate[index] = std::clamp(controls[index] - step, 0.0, 255.0);
            }
            const double candidate_cost = evaluate<false>(work, rgb, channel, candidate);
            if (candidate_cost < cost) {
                controls = candidate;
                accepted = true;
            }
        }
        std::array<std::uint16_t, controls_count> quantized{};
        for (std::size_t index = 0; index < controls_count; ++index) {
            quantized[index] = quantize(controls[index]);
        }
        const std::size_t count = check_integer(work, rgb, quantized, channel, false);
        if (count < best_count) {
            best_count = count;
            best = quantized;
        }
        if (!accepted || cost == 0.0) {
            break;
        }
    }
    stats.final_mismatches += best_count;
    for (std::size_t index = 0; index < controls_count; ++index) {
        output[index * 3 + channel] = best[index];
    }
}

}

extern "C" int stx_surface_fit_initialize(const std::uint8_t* rgb, std::size_t rgb_size,
    std::size_t width, std::size_t height, std::uint16_t* output, std::size_t output_count) {
    if (rgb == nullptr || output == nullptr || width == 0 || height == 0 ||
        width > 2048 || height > 2048 || width * height > 524288 ||
        rgb_size != width * height * 3 || output_count != controls_count * 3) { return 1; }
    try {
        const double ratio = static_cast<double>(width * height) / 8192.0;
        const std::size_t stride = std::max(std::size_t{1}, static_cast<std::size_t>(std::ceil(std::sqrt(ratio))));
        const std::vector<std::size_t> xs = sample_positions(width, stride);
        const std::vector<std::size_t> ys = sample_positions(height, stride);
        Axis full_x{};
        Axis full_y{};
        prepare_axis(width, full_x);
        prepare_axis(height, full_y);
        Workspace work(xs.size(), ys.size());
        select_axis(full_x, xs, work.x);
        select_axis(full_y, ys, work.y);
        std::vector<std::uint8_t> samples(xs.size() * ys.size() * 3);
        for (std::size_t y = 0; y < ys.size(); ++y) {
            for (std::size_t x = 0; x < xs.size(); ++x) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    samples[(y * xs.size() + x) * 3 + channel] = rgb[(ys[y] * width + xs[x]) * 3 + channel];
                }
            }
        }
        std::array<std::uint16_t, controls_count * 3> completed{};
        for (std::size_t channel = 0; channel < 3; ++channel) {
            std::fill(work.admitted.begin(), work.admitted.end(), std::uint8_t{1});
            std::array<double, controls_count> controls{};
            continuous_fit(work, samples.data(), channel, Loss::squared, 1, controls);
            continuous_fit(work, samples.data(), channel, Loss::huber, 64, controls);
            render(work, controls);
            for (std::size_t index = 0; index < work.admitted.size(); ++index) {
                const double error = work.prediction[index] - samples[index * 3 + channel];
                work.admitted[index] = std::abs(error) <= 2.0 ? 1 : 0;
            }
            continuous_fit(work, samples.data(), channel, Loss::interval, 24, controls);
            for (std::size_t index = 0; index < controls_count; ++index) {
                completed[index * 3 + channel] = quantize(controls[index]);
            }
        }
        std::copy(completed.begin(), completed.end(), output);
        return 0;
    } catch (const std::bad_alloc&) { return 2; }
    catch (...) { return 3; }
}

extern "C" int stx_surface_fit_render(const std::uint16_t* controls, std::size_t control_count,
    std::size_t width, std::size_t height, std::uint8_t* rgb, std::size_t rgb_size) {
    if (controls == nullptr || rgb == nullptr || width == 0 || height == 0 ||
        width > 2048 || height > 2048 || width * height > 524288 ||
        control_count != controls_count * 3 || rgb_size != width * height * 3) { return 1; }
    for (std::size_t index = 0; index < control_count; ++index) {
        if (controls[index] > 65280) { return 1; }
    }
    try {
        Axis x_axis{};
        Axis y_axis{};
        prepare_axis(width, x_axis);
        prepare_axis(height, y_axis);
        std::vector<std::uint32_t> horizontal_low(width * side * 3);
        std::vector<std::uint32_t> horizontal_high(width * side * 3);
        std::vector<std::uint64_t> low(width * 3);
        std::vector<std::uint32_t> high(width * 3);
        for (std::size_t row = 0; row < side; ++row) {
            for (std::size_t x = 0; x < width; ++x) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    std::uint64_t value = 0;
                    for (std::size_t column = 0; column < side; ++column) {
                        value += static_cast<std::uint64_t>(controls[(row * side + column) * 3 + channel]) * x_axis.integer[x * side + column];
                    }
                    horizontal_low[(row * width + x) * 3 + channel] = static_cast<std::uint32_t>(value);
                    horizontal_high[(row * width + x) * 3 + channel] = static_cast<std::uint32_t>(value >> 32);
                }
            }
        }
        for (std::size_t y = 0; y < height; ++y) {
            std::fill(low.begin(), low.end(), std::uint64_t{0});
            std::fill(high.begin(), high.end(), std::uint32_t{0});
            // Both factors in each product are 32-bit. This exposes ordinary
            // widening multiplication to common-width compiler vectorization.
            // The high limb is <=15; positive y weights sum to 2^20, so its
            // complete accumulation fits in 24 bits. Recombination is exact.
            for (std::size_t row = 0; row < side; ++row) {
                const std::uint32_t coefficient = y_axis.integer[y * side + row];
                for (std::size_t index = 0; index < width * 3; ++index) {
                    low[index] += static_cast<std::uint64_t>(coefficient) * horizontal_low[row * width * 3 + index];
                    high[index] += coefficient * horizontal_high[row * width * 3 + index];
                }
            }
            for (std::size_t index = 0; index < width * 3; ++index) {
                const std::uint64_t value = low[index] + (static_cast<std::uint64_t>(high[index]) << 32);
                rgb[y * width * 3 + index] = static_cast<std::uint8_t>((value + denominator / 2) / denominator);
            }
        }
        return 0;
    } catch (const std::bad_alloc&) { return 2; }
    catch (...) { return 3; }
}

extern "C" int stx_surface_fit_refine(const std::uint8_t* rgb, std::size_t rgb_size,
    std::size_t width, std::size_t height, const std::uint16_t* initial, std::size_t initial_count,
    std::uint16_t* output, std::size_t output_count, stx_surface_fit_stats* stats) {
    if (rgb == nullptr || initial == nullptr || output == nullptr || stats == nullptr ||
        width == 0 || height == 0 || width > 2048 || height > 2048 ||
        width * height > 524288 || rgb_size != width * height * 3 ||
        initial_count != controls_count * 3 || output_count != controls_count * 3) {
        return 1;
    }
    for (std::size_t index = 0; index < initial_count; ++index) {
        if (initial[index] > 65280) {
            return 1;
        }
    }
    try {
        Workspace work(width, height);
        prepare_admission(work, rgb, initial);
        stx_surface_fit_stats result{};
        result.workspace_bytes = work.payload_bytes();
        for (const std::uint8_t admitted : work.admitted) {
            result.admitted_pixels += admitted;
        }
        std::array<std::uint16_t, controls_count * 3> completed{};
        for (std::size_t channel = 0; channel < 3; ++channel) {
            refine_channel(work, rgb, initial, channel, completed, result);
        }
        std::copy(completed.begin(), completed.end(), output);
        *stats = result;
        return 0;
    } catch (const std::bad_alloc&) {
        return 2;
    } catch (...) {
        return 3;
    }
}
