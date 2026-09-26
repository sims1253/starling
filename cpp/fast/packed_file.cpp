// packed_file.cpp — see packed_file.hpp.

#include "packed_file.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

namespace starling::fast {

namespace {

// Explicit little-endian reads (the format is LE whatever the host is).
uint32_t rd_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
uint64_t rd_u64(const uint8_t* p) {
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}

// NUL-terminated string inside a fixed-width header field.
bool rd_field(const uint8_t* p, size_t width, std::string& out) {
    const void* nul = std::memchr(p, 0, width);
    if (!nul) return false;
    out.assign((const char*)p, (size_t)((const uint8_t*)nul - p));
    return true;
}

} // namespace

std::unique_ptr<PackedWeights> PackedWeights::load(const std::string& path, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "packed file: cannot open " + path;
        return nullptr;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    if (buf.size() < 4 + 4 + 65 + 16 + 4) {
        err = "packed file: truncated header";
        return nullptr;
    }
    if (std::memcmp(buf.data(), "SFPK", 4) != 0) {
        err = "packed file: bad magic";
        return nullptr;
    }
    size_t off = 4;
    const uint32_t ver = rd_u32(buf.data() + off);
    off += 4;
    if (ver != 1) {
        err = "packed file: unsupported version " + std::to_string(ver);
        return nullptr;
    }
    auto out = std::make_unique<PackedWeights>();
    if (!rd_field(buf.data() + off, 65, out->source_) ||
        !rd_field(buf.data() + off + 65, 16, out->rounding_)) {
        err = "packed file: unterminated header string";
        return nullptr;
    }
    off += 65 + 16;
    const uint32_t n_tensors = rd_u32(buf.data() + off);
    off += 4;
    // Invariant: off <= buf.size(); `have` compares without overflow.
    auto have = [&](uint64_t n) { return n <= (uint64_t)(buf.size() - off); };
    for (uint32_t t = 0; t < n_tensors; ++t) {
        if (!have(4)) { err = "packed file: truncated (tensor header)"; return nullptr; }
        const uint32_t name_len = rd_u32(buf.data() + off);
        off += 4;
        if (!have((uint64_t)name_len + 4)) { err = "packed file: truncated (name)"; return nullptr; }
        PackedTensor pt;
        pt.name.assign((const char*)buf.data() + off, name_len);
        off += name_len;
        const uint32_t spec_len = rd_u32(buf.data() + off);
        off += 4;
        if (!have((uint64_t)spec_len + 8)) { err = "packed file: truncated (spec)"; return nullptr; }
        std::string spec((const char*)buf.data() + off, spec_len);
        off += spec_len;
        if (!layout_from_string(spec, &pt.desc, &err)) {
            err = "packed file: " + pt.name + ": " + err;
            return nullptr;
        }
        pt.N = rd_u32(buf.data() + off);
        pt.K = rd_u32(buf.data() + off + 4);
        off += 8;
        if (pt.N == 0 || pt.K == 0 || pt.K % pt.desc.group != 0) {
            err = "packed file: " + pt.name + ": bad shape " + std::to_string(pt.N) + "x" +
                  std::to_string(pt.K) + " for " + spec;
            return nullptr;
        }
        struct { uint64_t code, scale, super; } sz;
        if (!have(24)) { err = "packed file: truncated (sizes)"; return nullptr; }
        sz.code = rd_u64(buf.data() + off);
        sz.scale = rd_u64(buf.data() + off + 8);
        sz.super = rd_u64(buf.data() + off + 16);
        off += 24;
        const uint64_t want_code = layout_code_bytes(pt.desc, pt.K) * pt.N;
        const uint64_t want_scale = layout_scale_bytes(pt.desc, pt.K) * pt.N;
        const uint64_t want_super =
            layout_super_bytes(pt.desc) ? (uint64_t)pt.N * 4 : 0;
        if (sz.code != want_code || sz.scale != want_scale ||
            sz.super != (pt.desc.scale_dtype == ScaleDtype::U8Super ? want_super : 0)) {
            err = "packed file: " + pt.name + ": blob sizes " + std::to_string(sz.code) + "/" +
                  std::to_string(sz.scale) + "/" + std::to_string(sz.super) +
                  " disagree with the descriptor (want " + std::to_string(want_code) + "/" +
                  std::to_string(want_scale) + "/" + std::to_string(want_super) + ")";
            return nullptr;
        }
        auto padded = [](uint64_t bytes) { return (bytes + 3) & ~(uint64_t)3; };
        auto take = [&](uint64_t bytes) {
            std::vector<uint8_t> v(buf.begin() + off, buf.begin() + off + bytes);
            off += (size_t)padded(bytes);
            return v;
        };
        // Every blob is written padded to 4 bytes (starling_layout_quant wblob).
        if (!have(padded(sz.code) + padded(sz.scale) + padded(sz.super))) {
            err = "packed file: truncated (blobs of " + pt.name + ")";
            return nullptr;
        }
        pt.codes = take(sz.code);
        pt.scales = take(sz.scale);
        if (sz.super) {
            const std::vector<uint8_t> raw = take(sz.super);
            pt.super.resize(pt.N);
            for (uint32_t r = 0; r < pt.N; ++r)
                pt.super[r] = (uint16_t)(raw[4 * r] | (raw[4 * r + 1] << 8));
        }
        out->tensors_.emplace(pt.name, std::move(pt));
    }
    return out;
}

bool PackedWeights::matrix(const std::string& name, HostMatrix& out, std::string& err) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        err = "packed file: no tensor " + name;
        return false;
    }
    const PackedTensor& pt = it->second;
    // Kernel support today: exactly the two legacy shapes. New descriptors
    // arrive with their #317 kernel variants; until then the engine must
    // refuse rather than misread the bytes.
    if (!pt.desc.is_legacy_w4() && !pt.desc.is_legacy_w8()) {
        err = "packed file: " + name + " uses layout " + layout_to_string(pt.desc) +
              ", which has no kernel support in this build";
        return false;
    }
    HostMatrix m;
    m.layout = pt.desc;
    m.N = pt.N;
    m.K = pt.K;
    m.fmt = pt.desc.is_legacy_w4() ? GpuFmt::W4 : GpuFmt::W8;
    // codes/scales are byte-counted (not word-padded): size up and zero
    // the tail so the copy never reads past what the file stored.
    m.q.resize((pt.codes.size() + 3) / 4);
    std::memcpy(m.q.data(), pt.codes.data(), pt.codes.size());
    if (pt.codes.size() % 4)
        std::memset((uint8_t*)m.q.data() + pt.codes.size(), 0, 4 - pt.codes.size() % 4);
    m.s.resize((pt.scales.size() + 3) / 4);
    std::memcpy(m.s.data(), pt.scales.data(), pt.scales.size());
    if (pt.scales.size() % 4)
        std::memset((uint8_t*)m.s.data() + pt.scales.size(), 0, 4 - pt.scales.size() % 4);
    // Per-row super scales (u8super layouts; unreachable until a kernel
    // accepts one, but carried so the descriptor and data never disagree).
    for (uint16_t h : pt.super) m.x.push_back(h);
    m.lossless = false;  // quantized from source, not repacked from GGUF
    out = m;
    return true;
}

uint64_t PackedWeights::bytes() const {
    uint64_t b = 0;
    for (const auto& kv : tensors_)
        b += kv.second.codes.size() + kv.second.scales.size() +
             (uint64_t)kv.second.super.size() * 2;
    return b;
}

} // namespace starling::fast
