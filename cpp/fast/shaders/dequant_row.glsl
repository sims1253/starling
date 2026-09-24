// dequant_row.glsl — scalar dequantization of element k of row n of a
// W4 / W8 / F16 matrix (see gemm.comp for the layouts) that may be split into
// up to four row chunks (large embedding tables exceed a single storage
// buffer range on some devices). Chunk c holds rows [c*CHUNK_ROWS, ...).
// Expects bindings 1..4 (wq0..wq3) and 5..8 (ws0..ws3).

layout(std430, binding = 1) readonly buffer WQ0 { uint wq0[]; };
layout(std430, binding = 2) readonly buffer WQ1 { uint wq1[]; };
layout(std430, binding = 3) readonly buffer WQ2 { uint wq2[]; };
layout(std430, binding = 4) readonly buffer WQ3 { uint wq3[]; };
layout(std430, binding = 5) readonly buffer WS0 { uint ws0[]; };
layout(std430, binding = 6) readonly buffer WS1 { uint ws1[]; };
layout(std430, binding = 7) readonly buffer WS2 { uint ws2[]; };
layout(std430, binding = 8) readonly buffer WS3 { uint ws3[]; };

uint rd_q(uint c, uint i) { return c == 0u ? wq0[i] : c == 1u ? wq1[i] : c == 2u ? wq2[i] : wq3[i]; }
uint rd_s(uint c, uint i) { return c == 0u ? ws0[i] : c == 1u ? ws1[i] : c == 2u ? ws2[i] : ws3[i]; }

float dequant(uint row, uint k, uint K) {
    const uint c = row / CHUNK_ROWS, n = row % CHUNK_ROWS;
    const uint G = K / 32u, g = n * G + k / 32u, j = k % 32u;
#if defined(W_W4)
    const vec2 so = unpackHalf2x16(rd_s(c, g));
    const uint q = (rd_q(c, g * 4u + j / 8u) >> (4u * (j % 8u))) & 15u;
    return so.x * float(q) + so.y;
#elif defined(W_W8)
    const float s = unpackHalf2x16(rd_s(c, g))[j / 16u];
    return s * float(bitfieldExtract(int(rd_q(c, g * 8u + j / 4u)), int(8u * (j % 4u)), 8));
#else
    const uint e = n * K + k;
    return unpackHalf2x16(rd_q(c, e >> 1))[e & 1u];
#endif
}
