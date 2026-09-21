// replay_graph_support_test.cpp — unit tests for the unsupported-node
// machinery ReplayGraph::alloc_internal uses to fail loudly when an
// accelerator rejects a captured graph (issue #184).
//
// The helpers under test (enumerate_unsupported_graph_nodes /
// format_unsupported_graph_node / check_no_unsupported_graph_nodes) take a
// support PREDICATE, so these tests need no GPU backend and make no CUDA
// coverage claims: the predicates here are fakes that reject chosen ops, and
// the tests verify the machinery's own contract:
//
//   (1) a fully-supported graph (always-true predicate) reports NO nodes;
//   (2) EVERY rejected node is listed (no early break — the old sched-dbg
//       loop stopped at the first hit and hid that the parakeet graph had one
//       rejected node per conformer layer);
//   (3) each entry names the right node (op, dst type, node index) with the
//       src details incl. cont/STRIDED layout flags, matching the sched-dbg
//       line format;
//   (4) nodes of several rejected op kinds are all enumerated;
//   (5) the build-time guard THROWS std::runtime_error whose message names
//       the device, the rejected/total counts, the sched-dbg node lines and
//       the actionable tail (#184 + STARLING_SCHED_DEBUG) — the error that
//       reaches g_last_error and the HTTP 500 body instead of ggml's
//       first-upload assert;
//   (6) the guard is a no-op (no throw) for a fully-supported graph;
//   (7) the embedded node lines cap at 16 with an "... and N more" line so
//       the message fits capi's 2048-byte error buffer;
//   (8) STARLING_SCHED_DEBUG=1 echoes every rejected node to stderr (checked
//       through a fork with redirected stderr; POSIX only).
//
// CPU-only. Exit 0 = pass, 1 = assertion failure.

#include "runtime/backend.hpp"

#include "ggml.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace starling::ggml;

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

int count_occurrences(const std::string& haystack, const std::string& needle) {
    int n = 0;
    for (size_t pos = haystack.find(needle); pos != std::string::npos;
         pos = haystack.find(needle, pos + needle.size()))
        ++n;
    return n;
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

// A graph that is a chain of `n_unary` sigmoids over one contiguous leaf —
// n_unary UNARY nodes over CONTIGUOUS sources, for exercising the guard's
// message cap (kMaxEmbeddedLines) with a chosen rejection count.
struct UnaryChainGraph {
    ggml_context* ctx = nullptr;
    ggml_cgraph*  gf  = nullptr;

    explicit UnaryChainGraph(int n_unary) {
        struct ggml_init_params prm = {
            /*.mem_size   =*/ ggml_tensor_overhead() * size_t(n_unary + 4)
                             + ggml_graph_overhead_custom(n_unary + 4, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(prm);
        ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
        for (int i = 0; i < n_unary; ++i) t = ggml_sigmoid(ctx, t);
        ggml_set_output(t);
        gf = ggml_new_graph_custom(ctx, n_unary + 4, false);
        ggml_build_forward_expand(gf, t);
    }
    ~UnaryChainGraph() { if (ctx) ggml_free(ctx); }
};

#ifndef _WIN32
// Run the guard in a forked child with stderr redirected into a pipe and
// STARLING_SCHED_DEBUG forced on/off in the child's environment; return the
// child's captured stderr. The child appends a GUARD_THREW marker line to
// stderr from the catch block, so a silently non-throwing guard is
// distinguishable from a broken stderr echo.
std::string guard_stderr_with_debug(bool debug_on) {
    int fds[2];
    if (pipe(fds) != 0) {
        check(false, "guard_stderr_with_debug: pipe() failed");
        return {};
    }
    pid_t pid = fork();
    if (pid < 0) {
        check(false, "guard_stderr_with_debug: fork() failed");
        close(fds[0]);
        close(fds[1]);
        return {};
    }
    if (pid == 0) {
        // Child: stderr -> pipe, control the env, run, exit.
        close(fds[0]);
        dup2(fds[1], 2);
        close(fds[1]);
        if (debug_on) setenv("STARLING_SCHED_DEBUG", "1", 1);
        else unsetenv("STARLING_SCHED_DEBUG");
        UnaryChainGraph tg(3);
        bool threw = false;
        try {
            check_no_unsupported_graph_nodes(
                tg.gf, supports_only({GGML_OP_UNARY}), "CUDA0-fake");
        } catch (const std::runtime_error&) { threw = true; /* expected */ }
        if (threw) std::fputs("GUARD_THREW\n", stderr);
        std::fflush(nullptr);
        _exit(0);
    }
    close(fds[1]);
    std::string captured;
    char buf[512];
    for (;;) {
        ssize_t n = read(fds[0], buf, sizeof buf);
        if (n > 0) { captured.append(buf, (size_t)n); continue; }
        if (n < 0 && errno == EINTR) continue; // e.g. SIGCHLD mid-read
        break; // EOF or unrecoverable error
    }
    close(fds[0]);
    int status = 0;
    pid_t waited;
    do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
    check(waited == pid, "guard_stderr_with_debug: waitpid() failed");
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "sched-dbg child exited cleanly");
    check(captured.find("GUARD_THREW\n") != std::string::npos,
          "guard threw in the child (a silent non-throw is a regression, "
          "not an echo problem)");
    return captured;
}
#endif // !_WIN32

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

    // (5) The build-time guard throws the typed engine error (the one that
    // reaches g_last_error / HTTP 500 instead of ggml's first-upload assert).
    {
        TestGraph tg;
        const int n_nodes = tg.n_nodes();
        bool threw = false;
        std::string msg;
        try {
            check_no_unsupported_graph_nodes(
                tg.gf, supports_only({GGML_OP_UNARY}), "CUDA0-fake");
        } catch (const std::runtime_error& e) {
            threw = true;
            msg = e.what();
        }
        check(threw, "guard throws std::runtime_error for rejected nodes");
        check(msg.find("ReplayGraph allocation failed") == 0,
              "message leads with the allocation-failure tag");
        check(msg.find("device 'CUDA0-fake'") != std::string::npos,
              "message names the rejecting device");
        check(msg.find("rejected 3 of " + std::to_string(n_nodes)) !=
                  std::string::npos,
              "message reports rejected/total node counts");
        check(count_occurrences(msg, ": op=UNARY dst=f32") == 3,
              "message embeds all 3 rejected sched-dbg node lines");
        check(msg.find("src0=f32(VIEW,STRIDED)") != std::string::npos,
              "embedded lines carry the STRIDED layout detail");
        check(msg.find("more rejected node(s)") == std::string::npos,
              "no cap line when the count fits the embedded limit");
        check(msg.find("issues/184") != std::string::npos &&
                  msg.find("STARLING_SCHED_DEBUG") != std::string::npos,
              "actionable tail points at #184 and STARLING_SCHED_DEBUG");
        check(msg.size() < 2048,
              "message fits capi's 2048-byte g_last_error buffer");
    }

    // (6) The guard is a no-op for a fully-supported graph.
    {
        TestGraph tg;
        bool threw = false;
        try {
            check_no_unsupported_graph_nodes(
                tg.gf, supports_only({}), "CUDA0-fake");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(!threw, "guard does not throw for a fully-supported graph");
    }

    // (7) More than 16 rejected nodes: the embedded lines cap at 16 and the
    // count continues in an "... and N more" line (the message must keep
    // fitting g_last_error's 2048 bytes with the actionable tail intact).
    {
        constexpr int kChain = 20;
        UnaryChainGraph tg(kChain);
        bool threw = false;
        std::string msg;
        try {
            check_no_unsupported_graph_nodes(
                tg.gf, supports_only({GGML_OP_UNARY}), "CUDA0-fake");
        } catch (const std::runtime_error& e) {
            threw = true;
            msg = e.what();
        }
        check(threw, "chain guard throws");
        check(msg.find("rejected " + std::to_string(kChain) + " of ") !=
                  std::string::npos,
              "chain message reports all 20 rejections");
        check(count_occurrences(msg, ": op=UNARY dst=f32") == 16,
              "exactly 16 node lines embedded (kMaxEmbeddedLines)");
        check(msg.find("... and 4 more rejected node(s)") != std::string::npos,
              "cap line reports the remaining 4 rejections");
        check(msg.find("issues/184") != std::string::npos,
              "actionable tail survives the cap");
        check(msg.size() < 2048, "capped message fits the 2048-byte buffer");
    }

#ifndef _WIN32
    // (8) STARLING_SCHED_DEBUG=1 echoes EVERY rejected node to stderr — the
    // only full enumeration when the message cap kicks in.
    {
        const std::string on = guard_stderr_with_debug(true);
        check(count_occurrences(on, "[sched-dbg] unsupported node ") == 3,
              "STARLING_SCHED_DEBUG=1 echoes all 3 rejected nodes");
        const std::string off = guard_stderr_with_debug(false);
        check(off.find("[sched-dbg]") == std::string::npos,
              "no sched-dbg echo without STARLING_SCHED_DEBUG");
    }
#endif

    if (g_failures) {
        std::printf("replay_graph_support_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("replay_graph_support_test: all checks passed\n");
    return 0;
}
