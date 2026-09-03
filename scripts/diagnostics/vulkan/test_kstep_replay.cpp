// K-step replay probe — minimal deterministic repro of the ggml-vulkan
// intra-graph stale-read scheduling bug behind the parakeet TDT
// early-termination on Vulkan (2026-09-03).
//
// The engine symptom (full trace: scripts/diagnostics/vulkan/README.md #5):
// the parakeet K-step multistep graph (one captured cgraph replayed per K
// decode steps) computes step-chain state garbage on every replay AFTER the
// first — ring frames jump to f32-bit-pattern values (e.g. 1073741824.0 =
// float(0x40000000)) while the same buffers read back CORRECT from the host,
// and i32 gather results come back holding f32 BITS (0x3F800000 where i32 1
// belongs). Both multistep sub-paths (device-resident add_graph_root
// write-back and the host round-trip baseline) are affected; a
// single-replay graph and the per-step serial paths are exact.
//
// This probe shrinks the trigger to a ~20-node graph WITHOUT any replay: on
// Vulkan the ARGMAX over a [64]-wide row reads ZEROS (returns 0) even though
// its input tensor's contents are verifiably correct by the end of the
// compute (captured and compared against the CPU reference) — i.e. an op
// executed against pre-write buffer state. Isolated argmax (leaf input, same
// width, with or without a view) is correct, so the violation depends on the
// surrounding graph shape — the same class as the engine's replay-boundary
// corruption, and the starting point for the ggml-vulkan scheduling fix
// (batch/semaphore choreography in ggml_vk_build_graph /
// ggml_vk_compute_forward).
//
// Build (repo root; zsh: use ${=INC} etc.):
//   g++ -O1 -std=c++20 $INC -o /tmp/test_kstep_replay \
//     scripts/diagnostics/vulkan/test_kstep_replay.cpp $LIBS $RPATH
// Run:  /tmp/test_kstep_replay                     — Vulkan0: tok=00000000
//       KSTEP_REPLAY_DEVICE=cpu /tmp/test_kstep_replay — reference: tok=0000003f
// Toggles: KSTEP_REPLAY_OPS=comma list of clamp,i32idx,getrows,cont (every
// combination still fails on Vulkan); KSTEP_REPLAY_SYNC=1 uses the sync
// compute + sync readback path (also fails).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"  // cgraph internals (n_nodes/n_leafs)

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static uint32_t bits_of(float v) { uint32_t b; memcpy(&b, &v, 4); return b; }

int main() {
    const char* dev_name = std::getenv("KSTEP_REPLAY_DEVICE");
    if (!dev_name) dev_name = "Vulkan0";
    ggml_backend_t bk = nullptr;
    if (std::string(dev_name) == "cpu") {
        bk = ggml_backend_cpu_init();
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (std::string(ggml_backend_dev_name(d)) == dev_name) {
                bk = ggml_backend_dev_init(d, nullptr);
                break;
            }
        }
    }
    if (!bk) { printf("no device %s\n", dev_name); return 1; }
    printf("device: %s\n", ggml_backend_name(bk));

    const bool sync_mode = std::getenv("KSTEP_REPLAY_SYNC") != nullptr;
    const std::string dis = std::getenv("KSTEP_REPLAY_OPS") ? std::getenv("KSTEP_REPLAY_OPS") : "";
    auto off = [&](const char* n) { return dis.find(n) != std::string::npos; };
    const int K = 2, Hp = 64, T = 24;

    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context* ctx = ggml_init(ip);

    std::vector<ggml_tensor*> inputs, captures;
    std::vector<const void*> host_backings;
    std::vector<std::vector<float>> cap_data;
    cap_data.reserve(16);  // cap_ptrs holds pointers into cap_data — no realloc
    std::vector<std::vector<float>*> cap_ptrs;
    auto reg = [&](ggml_tensor* t, const void* host) {
        ggml_set_input(t);
        inputs.push_back(t);
        host_backings.push_back(host);
    };
    auto cap = [&](ggml_tensor* t) {
        ggml_set_output(t);
        captures.push_back(t);
        cap_data.emplace_back((size_t)ggml_nelements(t), 0.0f);
        cap_ptrs.push_back(&cap_data.back());
    };

    // Host input data: the enc_proj-like table has a unique max per row at a
    // known position, so CPU argmax = 63 and any stale/zero read yields 0.
    std::vector<float> encp((size_t)Hp * T);
    for (size_t i = 0; i < encp.size(); ++i) encp[i] = 0.05f * (float)(((i * 13) % 199) - 99) / 99.0f;
    const int32_t frame_seed[1] = { 0 };

    ggml_tensor* t_encp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Hp, T);
    reg(t_encp, encp.data());
    ggml_tensor* frame_i32 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    reg(frame_i32, frame_seed);

    std::vector<ggml_tensor*> tok_nodes, frame_nodes;
    ggml_tensor* dbg_i32 = nullptr;
    for (int j = 0; j < K; ++j) {
        // The engine's per-step frame chain: cast i32->f32, in-place clamp,
        // cast f32->i32, gather a row, argmax it.
        ggml_tensor* frame_f = ggml_cast(ctx, frame_i32, GGML_TYPE_F32);
        ggml_tensor* frame_src = frame_f;
        if (!off("clamp")) frame_src = ggml_clamp(ctx, frame_f, 0.0f, (float)(T - 1));
        ggml_tensor* idx = frame_i32;
        if (!off("i32idx")) idx = ggml_cast(ctx, frame_src, GGML_TYPE_I32);
        ggml_tensor* ep = off("getrows")
            ? ggml_view_1d(ctx, t_encp, Hp, 0)
            : ggml_get_rows(ctx, t_encp, idx);
        if (!off("cont")) ep = ggml_cont_1d(ctx, ep, Hp);
        ggml_tensor* tok = ggml_argmax(ctx, ep);
        ggml_tensor* frame_next = ggml_add(ctx, frame_f, ggml_cast(ctx, tok, GGML_TYPE_F32));

        // Intermediates: ep ends byte-identical on both backends, tok does not.
        if (j == 0) { cap(ep); cap(tok); cap(frame_f); }
        tok_nodes.push_back(ggml_cast(ctx, tok, GGML_TYPE_F32));
        frame_nodes.push_back(frame_next);
        frame_i32 = ggml_cast(ctx, frame_next, GGML_TYPE_I32);
        dbg_i32 = idx;
    }
    ggml_tensor* ring_tok = tok_nodes[0];
    for (int j = 1; j < K; ++j) ring_tok = ggml_concat(ctx, ring_tok, tok_nodes[j], 0);
    ggml_tensor* ring_frame = frame_nodes[0];
    for (int j = 1; j < K; ++j) ring_frame = ggml_concat(ctx, ring_frame, frame_nodes[j], 0);
    cap(ring_tok);
    cap(ring_frame);
    cap(dbg_i32);
    ggml_tensor* out = ring_frame;

    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, out);
    for (ggml_tensor* c : captures) ggml_build_forward_expand(gf, c);
    printf("graph: %d nodes, %d leafs (ops off: %s)\n", gf->n_nodes, gf->n_leafs, dis.c_str());

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bk));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("alloc failed\n"); return 1; }

    int mismatches = 0;
    for (int rep = 0; rep < 2; ++rep) {
        for (size_t i = 0; i < inputs.size(); ++i)
            if (inputs[i]->data)  // unreachable inputs get no gallocr buffer
                ggml_backend_tensor_set(inputs[i], host_backings[i], 0, ggml_nbytes(inputs[i]));
        bool ok;
        if (sync_mode) {
            ok = ggml_backend_graph_compute(bk, gf) == GGML_STATUS_SUCCESS;
            if (ok)
                for (size_t i = 0; i < captures.size(); ++i)
                    ggml_backend_tensor_get(captures[i], cap_ptrs[i]->data(), 0,
                                            (size_t)ggml_nelements(captures[i]) * ggml_element_size(captures[i]));
        } else {
            ok = ggml_backend_graph_compute_async(bk, gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                for (size_t i = 0; i < captures.size(); ++i)
                    ggml_backend_tensor_get_async(bk, captures[i], cap_ptrs[i]->data(), 0,
                                                  (size_t)ggml_nelements(captures[i]) * ggml_element_size(captures[i]));
                ggml_backend_synchronize(bk);
            }
        }
        if (!ok) { printf("rep %d compute FAILED\n", rep); return 1; }
        printf("rep %d (%s): ep0=%08x tok=%08x f0=%08x ring_tok=%08x %08x ring_frame=%08x %08x dbg_i32=%08x\n",
               rep, sync_mode ? "sync" : "async",
               bits_of((*cap_ptrs[0])[0]), bits_of((*cap_ptrs[1])[0]), bits_of((*cap_ptrs[2])[0]),
               bits_of((*cap_ptrs[3])[0]), bits_of((*cap_ptrs[3])[1]),
               bits_of((*cap_ptrs[4])[0]), bits_of((*cap_ptrs[4])[1]),
               bits_of((*cap_ptrs[5])[0]));
        // CPU reference: tok = 63 (0x3f), frames 63.0/127.0, dbg_i32 = 23.
        if (bits_of((*cap_ptrs[1])[0]) != 0x3fu) ++mismatches;
    }
    printf("%s\n", mismatches ? "STALE-READ REPRODUCED (tok != 63; see README #5)" : "argmax OK (cpu-class behavior)");
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    ggml_backend_free(bk);
    return mismatches ? 1 : 0;
}
