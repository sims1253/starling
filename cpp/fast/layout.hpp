// layout.hpp — parameterized description of a packed fast-engine weight
// layout (#319).
//
// The W4/W8 layouts in weights.hpp were chosen so GGUF files repack
// losslessly; this header is the descriptor side of designing the layout
// together with the decode kernel (#317). One LayoutDesc is consumed by
//
//   * the offline quantizer (tools/starling_layout_quant.cpp), which packs
//     MOSS source weights straight into the described layout with
//     round-to-nearest plus an optional imatrix-weighted scale search, and
//   * the CPU reference dequant (layout_dequant below), which mirrors the
//     shader arithmetic bit for bit (f16 scales are converted to f32 before
//     use, exactly like unpackHalf2x16 in the kernels), and
//   * the packed-file loader (packed_file.hpp), which turns a stored
//     descriptor back into the HostMatrix the engine uploads.
//
// Canonical spec grammar (layout_to_string / layout_from_string):
//
//   w4g32asym    4-bit, f16 (scale, offset) per 32 — exactly today's GPU W4
//                (the Q4_0/Q4_1/Q4_K shape)
//   w8g16sym     8-bit, f16 scale per 16 — exactly today's GPU W8 (the
//                Q8_0/Q6_K shape; symmetric: w = s*q)
//   w4g64sym     4-bit symmetric, w = s*(q-8), one f16 scale per 64
//   w4g128symu8s 4-bit symmetric, u8 group scales under an f16 per-row
//                super-scale, group 128
//   w8g32sym     8-bit symmetric, one f16 scale per 32
//
// One group size concept: `group` weights along K share one scale (and one
// offset, when asymmetric). Storage per row of K weights (K % group == 0):
//
//   codes  4-bit: one nibble per weight, sequential inside each 32-code word
//          (order=1 permutes nibbles inside the word so the low nibbles of
//          the four bytes are codes 0..3 and the high nibbles codes 4..7 —
//          a byte-oriented unpack then yields K-consecutive values with no
//          activation reordering). 8-bit: one int8 per weight, sequential.
//   scales scale_dtype F16: one f16 scale (symmetric) or one f16 pair
//          (scale, offset) — asymmetric — per group, packed two halves per
//          32-bit word in K order. w4g32asym and w8g16sym therefore produce
//          byte-identical rows to today's pack_gpu_matrix output.
//          scale_dtype U8Super (symmetric only): one u8 per group (the
//          effective scale is super*u8, computed in f32 exactly in that
//          order) plus one f16 super-scale per row.
//
// The packed row order matches HostMatrix: codes -> HostMatrix::q,
// scale words -> HostMatrix::s, per-row super scales -> HostMatrix::x, so
// permute_rows / concat_rows / row chunking keep working on any layout.

#pragma once

#include <cstdint>
#include <string>

namespace starling::fast {

enum class ScaleDtype : uint32_t { F16 = 0, U8Super = 1 };

struct LayoutDesc {
    uint32_t bits = 4;       // code width: 4 or 8
    uint32_t group = 32;     // weights per scale group along K (multiple of 16)
    bool symmetric = false;  // asym: w = s*q + o, codes 0..2^bits-1
                             // sym: w = s*(q - 8) for 4-bit, w = s*q for 8-bit
    ScaleDtype scale_dtype = ScaleDtype::F16;
    uint32_t order = 0;      // 0: sequential codes; 1: nibbles pre-permuted
                             // for byte-wise unpack (4-bit only)
    bool store_pair = false; // sym + F16: store the legacy (s, o=-8s) pair per
                             // group instead of one lean scale — byte-compatible
                             // with today's W4 kernel (spec suffix `-a`)

    bool valid() const {
        if (bits != 4 && bits != 8) return false;
        if (group == 0 || group % 16 != 0 || group > 1024) return false;
        if (order > 1 || (order == 1 && bits != 4)) return false;
        if (scale_dtype == ScaleDtype::U8Super && !symmetric) return false;  // no offset storage
        if (scale_dtype == ScaleDtype::U8Super && bits != 4) return false;   // sym4 codes only
        if (store_pair && (!symmetric || bits != 4 || scale_dtype != ScaleDtype::F16))
            return false;   // the legacy (s, -8s) W4 pair; meaningless at 8 bits
        return true;
    }

    // Exactly today's GPU W4 / W8 bytes?
    bool is_legacy_w4() const {
        return bits == 4 && group == 32 && scale_dtype == ScaleDtype::F16 && order == 0 &&
               (!symmetric || store_pair);
    }
    bool is_legacy_w8() const {
        return bits == 8 && group == 16 && symmetric && scale_dtype == ScaleDtype::F16 &&
               order == 0 && !store_pair;
    }
};

std::string layout_to_string(const LayoutDesc& d);
// Parses a canonical spec; `err` receives a reason on failure.
bool layout_from_string(const std::string& spec, LayoutDesc* out, std::string* err);
// 64-bit FNV-1a of the canonical string — cache/manifest key.
uint64_t layout_hash(const LayoutDesc& d);

// Per-row packed sizes (bytes) for K weights.
uint64_t layout_code_bytes(const LayoutDesc& d, uint32_t K);
uint64_t layout_scale_bytes(const LayoutDesc& d, uint32_t K);
uint64_t layout_super_bytes(const LayoutDesc& d);  // U8Super only, else 0

// Codes for one row: K code values (4-bit: 0..15; 8-bit: int8), stored per
// the physical order. Single place that knows the order, so quantizer,
// dequant and tests agree.
void layout_encode_row(const LayoutDesc& d, const int16_t* q, uint32_t K, uint8_t* codes);
int16_t layout_decode_code(const LayoutDesc& d, const uint8_t* codes, uint32_t k);

// f16 scale/offset storage for one row (F16 scale dtype only): group i's
// halves land at the i-th entry position in K order, two halves per word.
void layout_store_scales(const LayoutDesc& d, const float* s, const float* o, uint32_t K,
                         uint8_t* scale_bytes);
// Effective scale of group g and offset (asym), converting through f16
// exactly like the shader's unpackHalf2x16.
float layout_scale_at(const LayoutDesc& d, const uint8_t* scale_bytes, const uint16_t* super,
                      const uint8_t* u8s, uint32_t g);
float layout_offset_at(const LayoutDesc& d, const uint8_t* scale_bytes, uint32_t g);

// Reference dequant of element k of one row. Mirrors the shader arithmetic:
// scales/offsets/super are converted to f32 first, then
//   asym:  s * (float)q + o   (separate mul and add, never fused)
//   sym4:  s * ((float)q - 8)
//   sym8:  s * (float)q
//   u8s:   t = super * (float)u8 ; w = t * ((float)q - 8)
float layout_dequant(const LayoutDesc& d, const uint8_t* codes, const uint8_t* scale_bytes,
                     const uint16_t* super, const uint8_t* u8s, uint32_t k);

// Quantize one row of K source weights into the physical buffers.
// `im` (importance per input channel, K values) or nullptr. The scale (and
// offset) search minimizes the importance-weighted squared error; candidate
// scales are rounded to f16 before codes are chosen so the dequant the GPU
// computes is the one that was optimized. Returns the plain relative RMS
// error of the row (||w-deq|| / ||w||), for reporting, or -1 when K is not
// a multiple of d.group (nothing is written then).
double layout_quant_row(const LayoutDesc& d, const float* w, uint32_t K, const float* im,
                        uint8_t* codes, uint8_t* scale_bytes, uint16_t* super, uint8_t* u8s);

} // namespace starling::fast
