#include "gguf_q4k.hpp"
#include "independent_q4k.hpp"
#include "input_pattern.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
volatile float sink = 0.0f;
}

int main(int argc, char** argv) {
    try {
        std::string path, name, output_path, pattern = "cos", kernel = "packed-dot";
        int batch = 50, threads = 4, iterations = 30;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + option);
            const std::string value = argv[++i];
            if (option == "--gguf") path = value;
            else if (option == "--tensor") name = value;
            else if (option == "--output") output_path = value;
            else if (option == "--input") pattern = value;
            else if (option == "--kernel") kernel = value;
            else if (option == "--batch") batch = std::stoi(value);
            else if (option == "--threads") threads = std::stoi(value);
            else if (option == "--iterations") iterations = std::stoi(value);
            else throw std::runtime_error("unknown option " + option);
        }
        if (path.empty() || name.empty() || batch < 1 || threads < 1 || iterations < 1 ||
            (kernel != "raw-dot" && kernel != "packed-dot"))
            throw std::runtime_error("need --gguf and --tensor; positive batch/threads/iterations; "
                                     "--kernel raw-dot or packed-dot");
        auto tensor = independent_q4k::read_gguf_q4k(path, name);
        const auto prepare_start = Clock::now();
        independent_q4k::Matrix matrix(tensor.bytes.data(), tensor.bytes.size(),
                                       tensor.cols, tensor.rows, kernel == "packed-dot");
        const double prepare_us = std::chrono::duration<double, std::micro>(
            Clock::now() - prepare_start).count();
        std::vector<float> input(size_t(tensor.cols) * batch);
        std::vector<float> result(size_t(tensor.rows) * batch);
        independent_q4k::fill_input(input.data(), input.size(), pattern);
        const bool raw = kernel == "raw-dot";
        for (int i = 0; i < 5; ++i) matrix.run(input.data(), batch, result.data(), threads, raw);
        std::vector<double> times;
        times.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            const auto start = Clock::now();
            matrix.run(input.data(), batch, result.data(), threads, raw);
            const auto end = Clock::now();
            sink = result[0];
            times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }
        std::sort(times.begin(), times.end());
        if (!output_path.empty()) {
            std::ofstream output(output_path, std::ios::binary);
            if (!output) throw std::runtime_error("cannot open output file");
            output.write(reinterpret_cast<const char*>(result.data()),
                         static_cast<std::streamsize>(result.size() * sizeof(float)));
            if (!output) throw std::runtime_error("cannot write output file");
        }
        const auto argmax = std::max_element(result.begin(), result.end()) - result.begin();
        std::printf("backend=independent-q4k kernel=%s input=%s K=%lld N=%lld M=%d "
                    "weights=%.1f MiB extra=%.1f MiB prepare=%.1f us threads=%d\n",
                    kernel.c_str(), pattern.c_str(), static_cast<long long>(tensor.cols),
                    static_cast<long long>(tensor.rows), batch,
                    tensor.bytes.size() / 1048576.0,
                    (matrix.packed_bytes() + matrix.metadata_bytes()) / 1048576.0,
                    prepare_us, threads);
        std::printf("median=%.1f us argmax=%lld first=%.8g\n", times[times.size() / 2],
                    static_cast<long long>(argmax), result[0]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "standalone_q4k: %s\n", e.what());
        return 1;
    }
}
