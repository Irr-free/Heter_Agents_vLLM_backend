#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <omp.h>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double sequential_read_bandwidth_gib_s() {
  constexpr std::size_t bytes = 256ULL * 1024ULL * 1024ULL;
  constexpr int repeats = 8;
  const std::size_t count = bytes / sizeof(float);
  std::vector<float> data(count, 1.0f);

  volatile double sink = 0.0;
  const auto start = Clock::now();
  for (int r = 0; r < repeats; ++r) {
    double local = 0.0;
#pragma omp parallel for reduction(+ : local) schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(count); ++i) {
      local += data[i];
    }
    sink += local;
  }
  const double elapsed_s = seconds_since(start);
  const double total_gib =
      static_cast<double>(bytes) * repeats / (1024.0 * 1024.0 * 1024.0);
  if (sink == 0.0) {
    std::cerr << "unreachable\n";
  }
  return total_gib / elapsed_s;
}

double pointer_chase_ns_per_access() {
  constexpr std::size_t count = 8ULL * 1024ULL * 1024ULL;
  constexpr int rounds = 4;
  std::vector<std::uint32_t> permutation(count);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::mt19937 rng(0);
  std::shuffle(permutation.begin(), permutation.end(), rng);

  std::vector<std::uint32_t> next(count);
  for (std::size_t i = 0; i < count; ++i) {
    next[permutation[i]] = permutation[(i + 1) % count];
  }

  volatile std::uint32_t idx = permutation[0];
  const auto start = Clock::now();
  for (std::size_t i = 0; i < count * rounds; ++i) {
    idx = next[idx];
  }
  const double elapsed_s = seconds_since(start);
  if (idx == count + 1) {
    std::cerr << "unreachable\n";
  }
  return elapsed_s * 1e9 / static_cast<double>(count * rounds);
}

double omp_barrier_us() {
  constexpr int iterations = 20000;
  double elapsed_s = 0.0;
#pragma omp parallel
  {
#pragma omp barrier
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
#pragma omp barrier
    }
#pragma omp master
    elapsed_s = seconds_since(start);
  }
  return elapsed_s * 1e6 / iterations;
}

double contended_atomic_fetch_add_ns() {
  const int threads = omp_get_max_threads();
  const std::int64_t ops_per_thread = 200000;
  std::atomic<std::int64_t> counter{0};
  const auto start = Clock::now();
#pragma omp parallel
  {
    for (std::int64_t i = 0; i < ops_per_thread; ++i) {
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  }
  const double elapsed_s = seconds_since(start);
  const auto total_ops = static_cast<double>(threads) * ops_per_thread;
  return elapsed_s * 1e9 / total_ops;
}

}  // namespace

int main() {
  const int threads = omp_get_max_threads();
  const double read_gib_s = sequential_read_bandwidth_gib_s();
  const double random_ns = pointer_chase_ns_per_access();
  const double barrier_us = omp_barrier_us();
  const double atomic_ns = contended_atomic_fetch_add_ns();

  std::cout << "{";
  std::cout << "\"omp_threads\":" << threads;
  std::cout << ",\"sequential_read_gib_s\":" << read_gib_s;
  std::cout << ",\"random_pointer_chase_ns\":" << random_ns;
  std::cout << ",\"omp_barrier_us\":" << barrier_us;
  std::cout << ",\"contended_atomic_fetch_add_ns\":" << atomic_ns;
  std::cout << "}\n";
  return 0;
}
