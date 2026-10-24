#include "stx_bernstein.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
constexpr std::size_t side = 9;
constexpr std::size_t count = 81;
constexpr std::size_t pairs = 17;
constexpr std::array<double, 9> choose8{1, 8, 28, 56, 70, 56, 28, 8, 1};
constexpr std::array<double, 17> choose16{1, 16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1};
constexpr std::uint64_t mass = 1048576;
using Controls = std::array<double, count>;
using Matrix = std::array<double, count * count>;

struct Axis {
    // Coefficient-major values expose contiguous independent pixel lanes.
    std::vector<double> values;
    std::vector<double> products;
    std::array<double, count> gram{};
    std::array<double, pairs> totals{};

    explicit Axis(std::size_t length)
        : values(length * side), products(length * pairs) {
        const std::size_t extent = std::max(length - 1, std::size_t{1});
        for (std::size_t position = 0; position < length; ++position) {
            std::array<std::uint64_t, side> current{};
            current[0] = mass;
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
                values[index * length + position] = static_cast<double>(current[index]) / static_cast<double>(mass);
            }
        }
        // Exact-polynomial degree-16 moments model curvature only. The Q20
        // values above still define the objective, gradient and reconstruction.
        for (std::size_t position = 0; position < length; ++position) {
            const double t = static_cast<double>(position) / static_cast<double>(extent);
            std::array<double, pairs> basis{};
            basis[0] = 1.0;
            for (std::size_t degree = 1; degree < pairs; ++degree) {
                std::array<double, pairs> next{};
                for (std::size_t index = 0; index < degree; ++index) {
                    next[index] += (1.0 - t) * basis[index];
                    next[index + 1] += t * basis[index];
                }
                basis = next;
            }
            for (std::size_t index = 0; index < pairs; ++index) {
                products[position * pairs + index] = basis[index];
                totals[index] += basis[index];
            }
        }
        for (std::size_t first = 0; first < side; ++first) {
            for (std::size_t second = first; second < side; ++second) {
                double sum = 0.0;
                for (std::size_t position = 0; position < length; ++position) {
                    sum += values[first * length + position] * values[second * length + position];
                }
                gram[first * side + second] = sum;
                gram[second * side + first] = sum;
            }
        }
    }
};
}

struct stx_bernstein_workspace {
    std::size_t width;
    std::size_t height;
    std::size_t pixels;
    Axis x;
    Axis y;
    std::vector<std::int16_t> planar;
    std::vector<double> horizontal;
    std::vector<std::int64_t> integer_horizontal;
    std::vector<double> residual;
    std::vector<std::size_t> selected_columns;
    std::vector<double> row_products;
    std::vector<std::uint8_t> row_has_curvature;
    Matrix gram{};
    Matrix curvature_gram{};
    Matrix hessian{};
    std::array<double, count> product_factor{};
    Matrix elimination{};
    Controls gradient{};
    Controls right_hand_side{};
    Controls solution{};
    std::array<std::size_t, count> free_indices{};
    std::array<std::size_t, count> pair_lookup{};
    std::array<double, pairs * pairs> moments{};
    double ridge;
    std::array<double, 8> profile{};

    stx_bernstein_workspace(std::size_t input_width, std::size_t input_height)
        : width(input_width), height(input_height), pixels(width * height),
          x(width), y(height), planar(pixels * 3), horizontal(width * side),
          integer_horizontal(width * side),
          residual(width), selected_columns(width), row_products(height * pairs),
          row_has_curvature(height), ridge(0.0) {
        for (std::size_t first = 0; first < side; ++first) {
            for (std::size_t second = 0; second < side; ++second) {
                pair_lookup[first * side + second] = first + second;
                product_factor[first * side + second] = choose8[first] * choose8[second] / choose16[first + second];
            }
        }
        double trace = 0.0;
        for (std::size_t first = 0; first < count; ++first) {
            for (std::size_t second = 0; second < count; ++second) {
                gram[first * count + second] = y.gram[(first / side) * side + second / side] * x.gram[(first % side) * side + second % side];
            }
            for (std::size_t second = 0; second < count; ++second) {
                const std::size_t py = (first / side) * side + second / side;
                const std::size_t px = (first % side) * side + second % side;
                curvature_gram[first * count + second] = y.totals[pair_lookup[py]] * x.totals[pair_lookup[px]] * product_factor[py] * product_factor[px];
            }
            trace += gram[first * count + first];
        }
        ridge = std::max(trace / static_cast<double>(count) * 1e-6, 1e-9);
        for (std::size_t index = 0; index < count; ++index) {
            gram[index * count + index] += ridge;
            curvature_gram[index * count + index] += ridge;
        }
    }

    std::size_t payload_bytes() const {
        const std::size_t axis_values = x.values.capacity() + y.values.capacity();
        const std::size_t axis_products = x.products.capacity() + y.products.capacity();
        const std::size_t working_values = horizontal.capacity() + residual.capacity() + row_products.capacity();
        const std::size_t bytes = sizeof(stx_bernstein_workspace) + row_has_curvature.capacity() + (axis_values + axis_products + working_values) * sizeof(double) + planar.capacity() * sizeof(std::int16_t) + selected_columns.capacity() * sizeof(std::size_t) + integer_horizontal.capacity() * sizeof(std::int64_t);
        return bytes;
    }
};

namespace {
// Pivoted elimination solves the packed live n-by-n submatrix. Fixed-capacity
// arrays belong to the workspace and are overwritten on every solve.
bool solve(stx_bernstein_workspace& work, std::size_t n) {
    Matrix& matrix = work.elimination;
    Controls& rhs = work.right_hand_side;
    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t selected = pivot;
        for (std::size_t row = pivot + 1; row < n; ++row) {
            if (std::abs(matrix[row * count + pivot]) > std::abs(matrix[selected * count + pivot])) {
                selected = row;
            }
        }
        const double magnitude = std::abs(matrix[selected * count + pivot]);
        if (!(magnitude > 1e-30) || !std::isfinite(magnitude)) {
            return false;
        }
        if (selected != pivot) {
            for (std::size_t column = pivot; column < n; ++column) {
                std::swap(matrix[selected * count + column], matrix[pivot * count + column]);
            }
            std::swap(rhs[selected], rhs[pivot]);
        }
        for (std::size_t row = pivot + 1; row < n; ++row) {
            const double factor = matrix[row * count + pivot] / matrix[pivot * count + pivot];
            for (std::size_t column = pivot + 1; column < n; ++column) {
                matrix[row * count + column] -= factor * matrix[pivot * count + column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    for (std::size_t end = n; end > 0; --end) {
        const std::size_t row = end - 1;
        double value = rhs[row];
        for (std::size_t column = row + 1; column < n; ++column) {
            value -= matrix[row * count + column] * work.solution[column];
        }
        work.solution[row] = value / matrix[row * count + row];
        if (!std::isfinite(work.solution[row])) {
            return false;
        }
    }
    return true;
}

void horizontal(stx_bernstein_workspace& work, const Controls& controls) {
    std::fill(work.horizontal.begin(), work.horizontal.end(), 0.0);
    for (std::size_t row = 0; row < side; ++row) {
        for (std::size_t column = 0; column < side; ++column) {
            const double coefficient = controls[row * side + column];
            for (std::size_t x = 0; x < work.width; ++x) {
                work.horizontal[row * work.width + x] += coefficient * work.x.values[column * work.width + x];
            }
        }
    }
}

template<bool collect>
double evaluate(stx_bernstein_workspace& work, std::size_t channel, const Controls& controls, double ceiling = 0.0) {
    horizontal(work, controls);
    if constexpr (collect) {
        work.gradient.fill(0.0);
    }
    double cost = 0.0;
    for (std::size_t y = 0; y < work.height; ++y) {
        // Each output lane is independent; keep the nine basis terms in
        // registers instead of writing the same row nine times.
        const double* __restrict input = work.horizontal.data();
        double* __restrict output = work.residual.data();
        std::array<double, side> weights{};
        for (std::size_t row = 0; row < side; ++row) {
            weights[row] = work.y.values[row * work.height + y];
        }
        for (std::size_t x = 0; x < work.width; ++x) {
            double value = 0.0;
            value += weights[0] * input[0 * work.width + x];
            value += weights[1] * input[1 * work.width + x];
            value += weights[2] * input[2 * work.width + x];
            value += weights[3] * input[3 * work.width + x];
            value += weights[4] * input[4 * work.width + x];
            value += weights[5] * input[5 * work.width + x];
            value += weights[6] * input[6 * work.width + x];
            value += weights[7] * input[7 * work.width + x];
            value += weights[8] * input[8 * work.width + x];
            output[x] = value;
        }
        std::size_t inactive_count = 0;
        std::array<double, 4> costs{};
        std::array<std::size_t, 4> inactive{};
        const std::int16_t* __restrict target = work.planar.data() + channel * work.pixels + y * work.width;
        double* __restrict errors = work.residual.data();
        std::size_t position = 0;
        for (; position + 4 <= work.width; position += 4) {
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const double error = errors[position + lane] - target[position + lane];
                const double residual = error - std::clamp(error, -0.499, 0.499);
                errors[position + lane] = residual;
                costs[lane] += residual * residual;
                if constexpr (collect) {
                    inactive[lane] += residual == 0.0 ? 1 : 0;
                }
            }
        }
        cost += (costs[0] + costs[1]) + (costs[2] + costs[3]);
        inactive_count = inactive[0] + inactive[1] + inactive[2] + inactive[3];
        for (; position < work.width; ++position) {
            const double error = errors[position] - target[position];
            const double residual = error - std::clamp(error, -0.499, 0.499);
            errors[position] = residual;
            cost += residual * residual;
            if constexpr (collect) {
                inactive_count += residual == 0.0 ? 1 : 0;
            }
        }
        if constexpr (!collect) {
            // All remaining squared interval distances are nonnegative.
            // A rejected trial cannot recover once its partial cost reaches
            // the incumbent. Accepted trials still evaluate every pixel.
            if (cost >= ceiling) return cost;
        }
        if constexpr (collect) {
            std::array<double, side> row_gradient{};
            std::array<double, pairs> row_pairs{};
            for (std::size_t column = 0; column < side; ++column) {
                std::array<double, 4> sums{};
                std::size_t x = 0;
                const double* __restrict basis = work.x.values.data() + column * work.width;
                const double* __restrict errors = work.residual.data();
                for (; x + 4 <= work.width; x += 4) {
                    sums[0] += errors[x] * basis[x];
                    sums[1] += errors[x + 1] * basis[x + 1];
                    sums[2] += errors[x + 2] * basis[x + 2];
                    sums[3] += errors[x + 3] * basis[x + 3];
                }
                double sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
                for (; x < work.width; ++x) {
                    sum += errors[x] * basis[x];
                }
                row_gradient[column] = sum;
            }
            // Accumulate the smaller set of active/inactive columns. Dense
            // rows otherwise multiply 17 basis moments by mostly zero masks.
            const bool subtract_active = inactive_count > work.width / 2;
            std::size_t selected_count = 0;
            for (std::size_t x = 0; x < work.width; ++x) {
                const bool inactive = work.residual[x] == 0.0;
                if (inactive != subtract_active) {
                    work.selected_columns[selected_count] = x;
                    ++selected_count;
                }
            }
            // Pixel-major products let all 17 independent moments share each
            // selected-column lookup. Each sum retains its original order.
            for (std::size_t index = 0; index < selected_count; ++index) {
                const std::size_t x = work.selected_columns[index];
                const double* __restrict products = work.x.products.data() + x * pairs;
                double* __restrict sums = row_pairs.data();
                for (std::size_t pair = 0; pair < pairs; ++pair) {
                    sums[pair] += products[pair];
                }
            }
            if (subtract_active) {
                for (std::size_t pair = 0; pair < pairs; ++pair) {
                    row_pairs[pair] = work.x.totals[pair] - row_pairs[pair];
                }
            }
            for (std::size_t row = 0; row < side; ++row) {
                for (std::size_t column = 0; column < side; ++column) {
                    work.gradient[row * side + column] += work.y.values[row * work.height + y] * row_gradient[column];
                }
            }
            work.row_has_curvature[y] = inactive_count != 0;
            std::copy(row_pairs.begin(), row_pairs.end(), work.row_products.data() + y * pairs);
        }
    }
    return cost;
}

// The gap check needs only the gradient. Defer curvature assembly until a
// further step is needed, retaining each row's small sufficient statistic.
void assemble_curvature(stx_bernstein_workspace& work) {
    work.moments.fill(0.0);
    for (std::size_t y = 0; y < work.height; ++y) {
        if (work.row_has_curvature[y] == 0) continue;
        const double* __restrict products = work.row_products.data() + y * pairs;
        for (std::size_t py = 0; py < pairs; ++py) {
            const double coefficient = work.y.products[y * pairs + py];
            double* __restrict moments = work.moments.data() + py * pairs;
            for (std::size_t px = 0; px < pairs; ++px) {
                moments[px] += coefficient * products[px];
            }
        }
    }
    {
        for (std::size_t first = 0; first < count; ++first) {
            for (std::size_t second = 0; second < count; ++second) {
                const std::size_t py = work.pair_lookup[(first / side) * side + second / side];
                const std::size_t px = work.pair_lookup[(first % side) * side + second % side];
                const double factor_y = work.product_factor[(first / side) * side + second / side];
                const double factor_x = work.product_factor[(first % side) * side + second % side];
                work.hessian[first * count + second] = work.curvature_gram[first * count + second] - work.moments[py * pairs + px] * factor_y * factor_x;
            }
        }
    }
}

bool initialize(stx_bernstein_workspace& work, std::size_t channel, double lower, double upper, Controls& controls) {
    Controls rhs{};
    double sum = 0.0;
    for (std::size_t y = 0; y < work.height; ++y) {
        std::array<double, side> row_values{};
        for (std::size_t x = 0; x < work.width; ++x) {
            const double value = work.planar[channel * work.pixels + y * work.width + x];
            sum += value;
            for (std::size_t column = 0; column < side; ++column) {
                row_values[column] += value * work.x.values[column * work.width + x];
            }
        }
        for (std::size_t row = 0; row < side; ++row) {
            for (std::size_t column = 0; column < side; ++column) {
                rhs[row * side + column] += work.y.values[row * work.height + y] * row_values[column];
            }
        }
    }
    const double mean = sum / static_cast<double>(work.pixels);
    work.elimination = work.gram;
    for (std::size_t index = 0; index < count; ++index) {
        work.right_hand_side[index] = rhs[index] + work.ridge * mean;
    }
    if (!solve(work, count)) {
        return false;
    }
    for (std::size_t index = 0; index < count; ++index) {
        controls[index] = std::clamp(work.solution[index], lower, upper);
    }
    return true;
}

// Solve the current quadratic chart with a primal feasible active set.
// Bound crossings update only 81-dimensional state, not image-sized fields.
void direction(stx_bernstein_workspace& work, const Controls& values,
    double lower, double upper, Controls& update, stx_bernstein_stats& stats) {
    Controls current = values;
    Controls gradient = work.gradient;
    std::array<bool, count> blocked{};
    double gradient_scale = 1.0;
    for (std::size_t index = 0; index < count; ++index) {
        blocked[index] = current[index] == lower || current[index] == upper;
        gradient_scale = std::max(gradient_scale, std::abs(gradient[index]));
    }
    for (std::size_t iteration = 0; iteration < 243; ++iteration) {
        std::size_t live = 0;
        for (std::size_t index = 0; index < count; ++index) {
            if (!blocked[index]) {
                work.free_indices[live] = index;
                ++live;
            }
        }
        Controls step{};
        double maximum_step = 0.0;
        if (live != 0) {
            for (std::size_t row = 0; row < live; ++row) {
                const std::size_t source_row = work.free_indices[row];
                work.right_hand_side[row] = -gradient[source_row];
                for (std::size_t column = 0; column < live; ++column) {
                    work.elimination[row * count + column] = work.hessian[source_row * count + work.free_indices[column]];
                }
            }
            ++stats.bound_solves;
            if (!solve(work, live)) {
                ++stats.fallback_directions;
                break;
            }
            for (std::size_t index = 0; index < live; ++index) {
                const std::size_t target = work.free_indices[index];
                step[target] = work.solution[index];
                maximum_step = std::max(maximum_step, std::abs(step[target]));
            }
        }
        if (maximum_step <= 1e-8) {
            // A minimizer on this face is global only if every active bound
            // has the correct multiplier sign. Release the worst violation.
            std::size_t release = count;
            double violation = gradient_scale * 1e-10;
            for (std::size_t index = 0; index < count; ++index) {
                if (blocked[index]) {
                    const double value = current[index] == lower ? -gradient[index] : gradient[index];
                    if (value > violation) {
                        violation = value;
                        release = index;
                    }
                }
            }
            if (release == count) {
                break;
            }
            blocked[release] = false;
            continue;
        }
        double scale = 1.0;
        std::size_t boundary = count;
        for (std::size_t index = 0; index < count; ++index) {
            double bound = 1.0;
            if (step[index] > 0.0) {
                bound = (upper - current[index]) / step[index];
            } else if (step[index] < 0.0) {
                bound = (lower - current[index]) / step[index];
            }
            if (bound < scale) {
                scale = bound;
                boundary = index;
            }
        }
        for (std::size_t index = 0; index < count; ++index) {
            double next = std::clamp(current[index] + scale * step[index], lower, upper);
            if (index == boundary) {
                next = step[index] > 0.0 ? upper : lower;
            }
            current[index] = next;
        }
        if (boundary != count) {
            blocked[boundary] = true;
        }
        // Re-form from the retained origin to avoid accumulated gradient drift.
        for (std::size_t row = 0; row < count; ++row) {
            double change = 0.0;
            for (std::size_t column = 0; column < count; ++column) {
                change += work.hessian[row * count + column] * (current[column] - values[column]);
            }
            gradient[row] = work.gradient[row] + change;
        }
    }
    for (std::size_t index = 0; index < count; ++index) {
        update[index] = current[index] - values[index];
    }
}

std::int32_t quantize(double value) {
    const double scaled = value * 256.0;
    const double base = std::floor(scaled);
    std::int32_t integer = static_cast<std::int32_t>(base);
    const double fraction = scaled - base;
    if (fraction > 0.5 || (fraction == 0.5 && integer % 2 != 0)) {
        ++integer;
    }
    return integer;
}
}

extern "C" stx_bernstein_workspace* stx_bernstein_create(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0 || width > 2048 || height > 2048 || width * height > 524288) {
        return nullptr;
    }
    try {
        stx_bernstein_workspace* result = new stx_bernstein_workspace(width, height);
        return result;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void stx_bernstein_destroy(stx_bernstein_workspace* workspace) {
    delete workspace;
}

namespace {
int fit_loaded(stx_bernstein_workspace& work, double lower, double upper,
    std::array<std::int32_t, count * 3>& completed, stx_bernstein_stats& stats) {
    work.profile.fill(0.0);
    using Clock = std::chrono::steady_clock;
    stx_bernstein_stats result{};
    result.workspace_bytes = work.payload_bytes();
    std::array<double, 3> channel_initial{};
    std::array<double, 3> channel_final{};
    for (std::size_t channel = 0; channel < 3; ++channel) {
        std::size_t identical = channel;
        for (std::size_t previous = 0; previous < channel; ++previous) {
            const std::int16_t* input = work.planar.data() + channel * work.pixels;
            const std::int16_t* earlier = work.planar.data() + previous * work.pixels;
            if (std::equal(input, input + work.pixels, earlier)) {
                identical = previous;
                break;
            }
        }
        if (identical != channel) {
            channel_initial[channel] = channel_initial[identical];
            channel_final[channel] = channel_final[identical];
            result.initial_cost += channel_initial[channel];
            result.final_cost += channel_final[channel];
            for (std::size_t index = 0; index < count; ++index) {
                completed[index * 3 + channel] = completed[index * 3 + identical];
            }
            continue;
        }
        Controls values{};
        const Clock::time_point init_start = Clock::now();
        if (!initialize(work, channel, lower, upper, values)) {
            return 3;
        }
        work.profile[0] += std::chrono::duration<double>(Clock::now() - init_start).count();
        double final_cost = 0.0;
        for (std::size_t iteration = 0; iteration < 16; ++iteration) {
            const Clock::time_point collect_start = Clock::now();
            const double cost = evaluate<true>(work, channel, values);
            work.profile[1] += std::chrono::duration<double>(Clock::now() - collect_start).count();
            work.profile[4] += 1.0;
            if (iteration == 0) {
                result.initial_cost += cost;
                channel_initial[channel] = cost;
            }
            final_cost = cost;
            // Convexity gives a global objective-gap bound by minimizing the
            // tangent over the control box. Our gradient is half the true
            // gradient of the reported sum of squared interval distances.
            double objective_gap = 0.0;
            for (std::size_t index = 0; index < count; ++index) {
                const double bound = work.gradient[index] >= 0.0 ? lower : upper;
                objective_gap += 2.0 * work.gradient[index] * (values[index] - bound);
            }
            work.profile[6] = std::max(work.profile[6], objective_gap / std::max(cost, 1.0));
            if (objective_gap <= std::max(1e-8, cost * 1e-10)) {
                work.profile[7] += 1.0;
                break;
            }
            const Clock::time_point curvature_start = Clock::now();
            assemble_curvature(work);
            work.profile[1] += std::chrono::duration<double>(Clock::now() - curvature_start).count();
            Controls update{};
            const Clock::time_point solve_start = Clock::now();
            direction(work, values, lower, upper, update, result);
            work.profile[2] += std::chrono::duration<double>(Clock::now() - solve_start).count();
            bool accepted = false;
            double scale = 1.0;
            for (std::size_t trial = 0; trial < 8; ++trial) {
                Controls candidate{};
                for (std::size_t index = 0; index < count; ++index) {
                    candidate[index] = std::clamp(values[index] + scale * update[index], lower, upper);
                }
                const Clock::time_point trial_start = Clock::now();
                const double candidate_cost = evaluate<false>(work, channel, candidate, cost);
                work.profile[3] += std::chrono::duration<double>(Clock::now() - trial_start).count();
                work.profile[5] += 1.0;
                if (candidate_cost < cost) {
                    values = candidate;
                    final_cost = candidate_cost;
                    ++result.accepted_steps;
                    accepted = true;
                    break;
                }
                scale *= 0.5;
            }
            if (!accepted) {
                break;
            }
        }
        result.final_cost += final_cost;
        channel_final[channel] = final_cost;
        for (std::size_t index = 0; index < count; ++index) {
            completed[index * 3 + channel] = quantize(values[index]);
        }
    }
    stats = result;
    return 0;
}
}

extern "C" int stx_bernstein_fit(stx_bernstein_workspace* workspace,
    const std::uint8_t* rgb, std::size_t rgb_size, std::uint16_t* controls,
    std::size_t control_count, stx_bernstein_stats* stats) {
    if (workspace == nullptr || rgb == nullptr || controls == nullptr || stats == nullptr) {
        return 1;
    }
    stx_bernstein_workspace& work = *workspace;
    if (rgb_size != work.pixels * 3 || control_count != count * 3) {
        return 1;
    }
    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t index = 0; index < work.pixels; ++index) {
            work.planar[channel * work.pixels + index] = static_cast<std::int16_t>(rgb[index * 3 + channel]);
        }
    }
    std::array<std::int32_t, count * 3> completed{};
    stx_bernstein_stats result{};
    const int status = fit_loaded(work, 0.0, 255.0, completed, result);
    if (status != 0) {
        return status;
    }
    for (std::size_t index = 0; index < completed.size(); ++index) {
        controls[index] = static_cast<std::uint16_t>(completed[index]);
    }
    *stats = result;
    return 0;
}

extern "C" int stx_bernstein_fit_signed(stx_bernstein_workspace* workspace,
    const std::int16_t* residual, std::size_t residual_count, std::int32_t* controls,
    std::size_t control_count, stx_bernstein_stats* stats) {
    if (workspace == nullptr || residual == nullptr || controls == nullptr || stats == nullptr) {
        return 1;
    }
    stx_bernstein_workspace& work = *workspace;
    if (residual_count != work.pixels * 3 || control_count != count * 3) {
        return 1;
    }
    for (std::size_t index = 0; index < residual_count; ++index) {
        if (residual[index] < -255 || residual[index] > 255) {
            return 1;
        }
    }
    for (std::size_t channel = 0; channel < 3; ++channel) {
        for (std::size_t index = 0; index < work.pixels; ++index) {
            work.planar[channel * work.pixels + index] = residual[index * 3 + channel];
        }
    }
    std::array<std::int32_t, count * 3> completed{};
    stx_bernstein_stats result{};
    const int status = fit_loaded(work, -255.0, 255.0, completed, result);
    if (status != 0) {
        return status;
    }
    std::copy(completed.begin(), completed.end(), controls);
    *stats = result;
    return 0;
}

extern "C" int stx_bernstein_render_signed(stx_bernstein_workspace* workspace,
    const std::int32_t* controls, std::size_t control_count,
    std::int16_t* output, std::size_t output_count) {
    if (workspace == nullptr || controls == nullptr || output == nullptr) {
        return 1;
    }
    stx_bernstein_workspace& work = *workspace;
    if (control_count != count * 3 || output_count != work.pixels * 3) {
        return 1;
    }
    for (std::size_t index = 0; index < control_count; ++index) {
        if (controls[index] < -65280 || controls[index] > 65280) {
            return 1;
        }
    }
    constexpr std::int64_t denominator = 281474976710656;
    for (std::size_t channel = 0; channel < 3; ++channel) {
        std::fill(work.integer_horizontal.begin(), work.integer_horizontal.end(), std::int64_t{0});
        for (std::size_t row = 0; row < side; ++row) {
            for (std::size_t column = 0; column < side; ++column) {
                const std::int64_t coefficient = controls[(row * side + column) * 3 + channel];
                for (std::size_t x = 0; x < work.width; ++x) {
                    const std::int64_t weight = static_cast<std::int64_t>(work.x.values[column * work.width + x] * static_cast<double>(mass));
                    work.integer_horizontal[row * work.width + x] += coefficient * weight;
                }
            }
        }
        const std::size_t width = work.width;
        const std::size_t height = work.height;
        const std::int64_t* __restrict rows = work.integer_horizontal.data();
        std::int16_t* __restrict destination = output;
        for (std::size_t y = 0; y < height; ++y) {
            std::array<std::int64_t, side> weights{};
            for (std::size_t row = 0; row < side; ++row) {
                weights[row] = static_cast<std::int64_t>(work.y.values[row * height + y] * static_cast<double>(mass));
            }
            for (std::size_t x = 0; x < width; ++x) {
                std::int64_t numerator = weights[0] * rows[x];
                numerator += weights[1] * rows[width + x];
                numerator += weights[2] * rows[2 * width + x];
                numerator += weights[3] * rows[3 * width + x];
                numerator += weights[4] * rows[4 * width + x];
                numerator += weights[5] * rows[5 * width + x];
                numerator += weights[6] * rows[6 * width + x];
                numerator += weights[7] * rows[7 * width + x];
                numerator += weights[8] * rows[8 * width + x];
                // C++ division truncates toward zero. Correct it to floor so
                // signed half-up rounding matches the integer Python oracle.
                const std::int64_t shifted = numerator + denominator / 2;
                std::int64_t value = shifted / denominator;
                if (shifted < 0 && shifted % denominator != 0) {
                    --value;
                }
                destination[(y * width + x) * 3 + channel] = static_cast<std::int16_t>(value);
            }
        }
    }
    return 0;
}

extern "C" void stx_bernstein_profile(const stx_bernstein_workspace* workspace, double* output) {
    std::copy((*workspace).profile.begin(), (*workspace).profile.end(), output);
}
