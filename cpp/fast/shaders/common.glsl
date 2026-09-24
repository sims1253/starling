// common.glsl — helpers shared by the fast-engine kernels.
//
// Portability: 16-bit values are stored packed two per 32-bit word and read
// with unpackHalf2x16 (core GLSL), so no kernel needs the 16-bit-storage
// feature; int8 weights are unpacked from words with bitfieldExtract.

#ifndef STARLING_FAST_COMMON_GLSL
#define STARLING_FAST_COMMON_GLSL

#extension GL_EXT_control_flow_attributes : require

float sigmoid_f(float x) { return 1.0 / (1.0 + exp(-x)); }
float silu_f(float x) { return x / (1.0 + exp(-x)); }

// erf with |error| < 1.2e-7 (Numerical Recipes erfc Chebyshev fit), enough
// for torch's exact (approximate="none") GELU at f32.
float erf_f(float x) {
    float z = abs(x);
    float t = 1.0 / (1.0 + 0.5 * z);
    float r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
              t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
              t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    float e = 1.0 - r;
    return x >= 0.0 ? e : -e;
}
float gelu_erf_f(float x) { return 0.5 * x * (1.0 + erf_f(x * 0.70710678118654752)); }

// ACT codes shared by every kernel with a fused activation.
#define ACT_NONE 0u
#define ACT_SILU 1u
#define ACT_RELU 2u
#define ACT_GELU 3u

float apply_act(uint act, float v) {
    if (act == ACT_SILU) return silu_f(v);
    if (act == ACT_RELU) return max(v, 0.0);
    if (act == ACT_GELU) return gelu_erf_f(v);
    return v;
}

#endif
