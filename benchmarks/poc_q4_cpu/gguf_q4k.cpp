#include "gguf_q4k.hpp"
#include "independent_q4k.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace independent_q4k {
namespace {

uint64_t read_unsigned(std::ifstream& file, int bytes) {
    uint64_t value = 0;
    for (int i = 0; i < bytes; ++i) {
        const int c = file.get();
        if (c == EOF) throw std::runtime_error("truncated GGUF metadata");
        value |= uint64_t(uint8_t(c)) << (8 * i);
    }
    return value;
}

void skip(std::ifstream& file, uint64_t bytes) {
    if (bytes > uint64_t(std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("GGUF metadata length overflow");
    file.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!file) throw std::runtime_error("truncated GGUF metadata");
}

std::string read_string(std::ifstream& file) {
    const uint64_t length = read_unsigned(file, 8);
    if (length > (1u << 24)) throw std::runtime_error("GGUF string too long");
    std::string value(static_cast<size_t>(length), '\0');
    file.read(value.data(), static_cast<std::streamsize>(length));
    if (!file) throw std::runtime_error("truncated GGUF string");
    return value;
}

void skip_string(std::ifstream& file) {
    skip(file, read_unsigned(file, 8));
}

int scalar_bytes(uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}

void skip_value(std::ifstream& file, uint32_t type) {
    if (type == 8) { skip_string(file); return; }
    if (type == 9) {
        const uint32_t item_type = static_cast<uint32_t>(read_unsigned(file, 4));
        const uint64_t n = read_unsigned(file, 8);
        if (item_type == 8) {
            if (n > 10000000) throw std::runtime_error("GGUF string array too large");
            for (uint64_t i = 0; i < n; ++i) skip_string(file);
            return;
        }
        const int width = scalar_bytes(item_type);
        if (!width || n > UINT64_MAX / uint64_t(width))
            throw std::runtime_error("unsupported GGUF array type or length");
        skip(file, n * uint64_t(width));
        return;
    }
    const int width = scalar_bytes(type);
    if (!width) throw std::runtime_error("unsupported GGUF metadata type");
    skip(file, width);
}

} // namespace

TensorBytes read_gguf_q4k(const std::string& path, const std::string& name) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open GGUF: " + path);
    char magic[4];
    file.read(magic, 4);
    if (!file || std::string(magic, 4) != "GGUF")
        throw std::runtime_error("invalid GGUF magic");
    if (read_unsigned(file, 4) != 3) throw std::runtime_error("GGUF v3 required");
    const uint64_t n_tensors = read_unsigned(file, 8);
    const uint64_t n_kv = read_unsigned(file, 8);
    if (n_tensors > 1000000 || n_kv > 1000000)
        throw std::runtime_error("GGUF metadata count too large");

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = read_string(file);
        const uint32_t type = static_cast<uint32_t>(read_unsigned(file, 4));
        if (key == "general.alignment") {
            if (type != 4) throw std::runtime_error("invalid GGUF alignment type");
            alignment = read_unsigned(file, 4);
        } else skip_value(file, type);
    }
    if (!alignment || (alignment & (alignment - 1)) || alignment > 4096)
        throw std::runtime_error("invalid GGUF alignment");

    int64_t cols = 0, rows = 0;
    uint64_t target_offset = 0;
    bool found = false;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        const std::string tensor_name = read_string(file);
        const uint32_t dims = static_cast<uint32_t>(read_unsigned(file, 4));
        if (dims < 1 || dims > 4) throw std::runtime_error("invalid GGUF tensor rank");
        int64_t shape[4] = {1, 1, 1, 1};
        for (uint32_t j = 0; j < dims; ++j)
            shape[j] = static_cast<int64_t>(read_unsigned(file, 8));
        const uint32_t type = static_cast<uint32_t>(read_unsigned(file, 4));
        const uint64_t offset = read_unsigned(file, 8);
        if (tensor_name == name) {
            if (found) throw std::runtime_error("duplicate GGUF tensor name");
            if (dims != 2 || type != 12 || shape[0] <= 0 || shape[1] <= 0 ||
                shape[0] % block_width)
                throw std::runtime_error("target tensor is not a 2-D Q4_K matrix");
            cols = shape[0]; rows = shape[1]; target_offset = offset; found = true;
        }
    }
    if (!found) throw std::runtime_error("Q4_K tensor not found: " + name);

    const uint64_t metadata_end = static_cast<uint64_t>(file.tellg());
    const uint64_t data_start = (metadata_end + alignment - 1) & ~(alignment - 1);
    const uint64_t blocks = uint64_t(cols / block_width) * uint64_t(rows);
    if (blocks > SIZE_MAX / weight_block_bytes)
        throw std::runtime_error("GGUF tensor byte count overflow");
    const uint64_t bytes = blocks * weight_block_bytes;
    if (target_offset > UINT64_MAX - data_start ||
        bytes > UINT64_MAX - data_start - target_offset)
        throw std::runtime_error("GGUF tensor offset overflow");
    file.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(file.tellg());
    const uint64_t absolute_offset = data_start + target_offset;
    if (absolute_offset + bytes > file_size)
        throw std::runtime_error("truncated GGUF tensor data");
    TensorBytes result;
    result.cols = cols; result.rows = rows;
    result.bytes.resize(static_cast<size_t>(bytes));
    file.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
    file.read(reinterpret_cast<char*>(result.bytes.data()),
              static_cast<std::streamsize>(bytes));
    if (!file) throw std::runtime_error("failed to read GGUF tensor data");
    return result;
}

} // namespace independent_q4k
