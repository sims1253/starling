// replay_graph_support_test.cpp — unit test for the unsupported-node
// enumeration ReplayGraph::alloc_internal uses to fail loudly when an
// accelerator rejects a captured graph (issue #184).
//
// The helper under test (enumerate_unsupported_graph_nodes /
// format_unsupported_graph_node) takes a support PREDICATE, so this test
// needs no GPU backend and makes no CUDA coverage claims: the predicates here
// are fakes that reject chosen ops, and the test only verifies the
// enumeration's own contract:
//
//   (1) a fully-supported graph (always-true predicate) reports NO nodes;
//   (2) EVERY rejected node is listed (no early break — the old sched-dbg
//       loop stopped at the first hit and hid that the parakeet graph had one
//       rejected node per conformer layer);
//   (3) each entry names the right node (op, dst type, node index) with the
//       src details incl. cont/STRIDED layout flags, matching the sched-dbg
//       line format;
//   (4) nodes of several rejected op kinds are all enumerated.
//
// CPU-only. Exit 0 = pass, 1 = assertion failure.

#include "runtime/backend.hpp"

#include "ggml.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace starling::ggml;

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// A small graph exercising both rejection shapes of #184: two UNARY nodes
// over strided views (the parakeet GLU pattern) plus a UNARY over a
// contiguous leaf, mixed with supported ADD/MUL nodes.
struct TestGraph {
    ggml_context* ctx = nullptr;
    ggml_cgraph*  gf  = nullptr;
    ggml_tensor*  sig_strided[2] = {nullptr, nullptr};
    ggml_tensor*  silu_contig = nullptr;

    TestGraph() {
        struct ggml_init_params prm = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 64
                             + ggml_graph_overhead_custom(64, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(prm);
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
        ggml_tensor* y = ggml_add(ctx, a, b);                       // [8,4] cont
        // Strided [4,4] halves of y (row stride 8 floats) — the GLU pattern.
        ggml_tensor* va = ggml_view_2d(ctx, y, 4, 4, y->nb[1], 0);
        ggml_tensor* vb = ggml_view_2d(ctx, y, 4, 4, y->nb[1],
                                       (size_t)4 * y->nb[0]);
        sig_strided[0] = ggml_sigmoid(ctx, va);
        sig_strided[1] = ggml_sigmoid(ctx, vb);
        ggml_tensor* glu = ggml_mul(ctx, sig_strided[0], sig_strided[1]);
        // A UNARY over a CONTIGUOUS leaf: same op kind, different layout flag.
        silu_contig = ggml_silu(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4));
        ggml_tensor* out = ggml_add(ctx, glu, silu_contig);
        ggml_set_output(out);
        gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, out);
    }
    ~TestGraph() { if (ctx) ggml_free(ctx); }

    int n_nodes() const { return ggml_graph_n_nodes(gf); }
    int count_op(ggml_op op) const {
        int n = 0;
        for (int i = 0; i < n_nodes(); ++i)
            if (ggml_graph_node(gf, i)->op == op) ++n;
        return n;
    }
};

std::function<bool(const ggml_tensor*)> supports_only(
    std::vector<ggml_op> rejected) {
    return [rejected](const ggml_tensor* t) {
        for (ggml_op op : rejected)
            if (t->op == op) return false;
        return true;
    };
}

} // namespace

int main() {
    // (1) Fully-supported graph reports nothing.
    {
        TestGraph tg;
        auto entries = enumerate_unsupported_graph_nodes(
            tg.gf, supports_only({}));
        check(entries.empty(),
              "always-true predicate must report no unsupported nodes");
    }

    // (2)+(3) UNARY rejected: every UNARY node listed, in order, with the
    // right node indices and sched-dbg details.
    {
        TestGraph tg;
        const int n_nodes = tg.n_nodes();
        auto entries = enumerate_unsupported_graph_nodes(
            tg.gf, supports_only({GGML_OP_UNARY}));
        check((int)entries.size() == tg.count_op(GGML_OP_UNARY),
              "all UNARY nodes enumerated (no early break)");
        check(entries.size() == 3, "expected 3 UNARY nodes (2 strided + 1 contig)");
        int prev = -1;
        for (const auto& e : entries) {
            check(e.node_index > prev, "entries in increasing node order");
            prev = e.node_index;
            check(e.op == std::string("UNARY"), "entry op is UNARY");
            check(e.dst_type == std::string("f32"), "entry dst type is f32");
            check(ggml_graph_node(tg.gf, e.node_index)->op == GGML_OP_UNARY,
                  "entry index points at a UNARY node");
            check(e.srcs.rfind("src0=f32(", 0) == 0,
                  "entry srcs start with src0=f32(");
        }
        // The two strided-view sigmoids and the contiguous silu must each be
        // present, with the layout flag the sched-dbg line prints.
        int strided = 0, contig = 0;
        for (const auto& e : entries) {
            if (e.srcs.find("src0=f32(VIEW,STRIDED)") != std::string::npos) ++strided;
            if (e.srcs.find("src0=f32(NONE,cont)") != std::string::npos) ++contig;
        }
        check(strided == 2, "2 UNARY nodes over strided views flagged STRIDED");
        check(contig == 1, "1 UNARY node over contiguous leaf flagged cont");

        // Full formatted lines match the sched-dbg format exactly.
        const std::string expect_strided =
            "node " + std::to_string(entries[0].node_index) + "/" +
            std::to_string(n_nodes) + ": op=UNARY dst=f32 src0=f32(VIEW,STRIDED)";
        check(format_unsupported_graph_node(entries[0], n_nodes) == expect_strided,
              "formatted strided line matches sched-dbg format");
        const auto* contig_entry = &entries[0];
        for (const auto& e : entries)
            if (e.srcs.find(",cont)") != std::string::npos) contig_entry = &e;
        const std::string expect_contig =
            "node " + std::to_string(contig_entry->node_index) + "/" +
            std::to_string(n_nodes) + ": op=UNARY dst=f32 src0=f32(NONE,cont)";
        check(format_unsupported_graph_node(*contig_entry, n_nodes) == expect_contig,
              "formatted contiguous line matches sched-dbg format");
    }

    // (4) Several rejected op kinds are all enumerated together.
    {
        TestGraph tg;
        auto entries = enumerate_unsupported_graph_nodes(
            tg.gf, supports_only({GGML_OP_UNARY, GGML_OP_MUL}));
        const int expected = tg.count_op(GGML_OP_UNARY) + tg.count_op(GGML_OP_MUL);
        check((int)entries.size() == expected,
              "UNARY and MUL rejections both enumerated");
        int unaries = 0, muls = 0;
        for (const auto& e : entries) {
            if (e.op == "UNARY") ++unaries;
            if (e.op == "MUL") ++muls;
        }
        check(unaries == tg.count_op(GGML_OP_UNARY) && muls >= 1,
              "mixed rejection lists nodes of every rejected kind");
    }

    if (g_failures) {
        std::printf("replay_graph_support_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("replay_graph_support_test: all checks passed\n");
    return 0;
}
