// tdt_graph_budget_test.cpp — unit test for the S02 byte-aware TDT K-step
// graph cache (ByteBudgetLruCache, runtime/lru_cache.hpp): the Parakeet TDT
// (T,K) graph map was an unbounded unordered_map (cpp/parakeet/
// tdt_multistep.cpp), retaining one captured ReplayGraph + private gallocr
// device buffer per distinct encoder length until model destruction,
// bypassing STARLING_REPLAY_CACHE_SIZE. This is the same VRAM-exhaustion
// class the encoder LRU fixed (replay_cache_lru_test.cpp), keyed on bytes
// instead of entries because graph cost varies by orders of magnitude with
// (T,K).
//
// The cache is a generic template, so the budget/LRU/pin/accounting contract
// is tested in two layers:
//
// Part A (pure logic, no model, no backend — tiny fake shapes):
//   (1) byte accounting: bytes_in_use() equals the sum of resident entries'
//       tracked bytes after every operation;
//   (2) budget enforced under many distinct (T,K) shapes: steady state
//       saturates at the budget, never exceeds it at rest;
//   (3) eviction is LRU by LAST USE (a re-acquired old key survives; an
//       untouched older key goes first);
//   (4) pin/lease: a pinned (in-use) entry is never evicted — its Value*
//       stays stable and intact while enough newer entries are inserted to
//       force repeated evictions, even when it is itself the LRU entry;
//   (5) oversized entry (bigger than the whole budget): admitted while
//       pinned (everything else evicted), trimmed at release — the budget
//       invariant is bytes <= budget at REST, overshoot only while leased;
//   (6) construction failure: an init that throws rolls the entry back (no
//       bytes retained, no stale LRU slot) and later acquires still work;
//   (7) no-growth steady state at fixed shapes: repeating the same shape set
//       builds each graph exactly once and holds bytes constant.
//
// Part A (pure logic, no model, no backend — tiny fake shapes), R20 review
// additions:
//   (12) env override parsing (env_budget_bytes, the
//       STARLING_TDT_GRAPH_BUDGET_MB path): whole-value validation — garbage,
//       trailing junk, empty, zero, and negative are rejected LOUDLY (stderr
//       diagnostic + the default budget); a value whose << 20 would wrap
//       clamps loudly to the largest representable budget;
//   (13) a zero byte budget is rejected at construction
//       (std::invalid_argument), not silently clamped to 1;
//   (14) mass eviction picks the LRU-ordered UNPINNED prefix in one pass,
//       skipping interleaved pins (regression net for the single-pass trim).
//
// Part B (real ReplayGraph entries on the CPU backend — the
// device_cache_clear_test pattern; the TDT decode path itself is GPU-gated,
// but the graph/accounting primitives it uses are backend-agnostic):
//   (8) ReplayGraph::device_alloc_bytes() — the accounting source — reports
//       a nonzero buffer size covering at least the graph's input tensors;
//   (9) the budget holds across MANY distinct (T,K) keys wrapping REAL
//       captured graphs, with bytes_in_use() bounded by the budget at rest;
//  (10) a leased graph's pointers stay stable while other entries are
//       evicted, and it still replays CORRECTLY afterwards (each entry's
//       private gallocr means one entry's destruction never corrupts another
//       entry's device buffers);
//  (11) steady state at fixed shapes: repeats are pure cache hits (no new
//       builds, constant byte total).
//
// GPU/model-gated validation (hundreds of real transcript lengths through
// the parakeet TDT path on CUDA, VRAM plateau, hit-rate reporting, sanitizer
// runs of the full model) is out of scope here and recorded as a validation
// gap on the S02 card.
//
// CPU-only. Exit 0 = pass, 1 = failure.

#include "runtime/backend.hpp"
#include "runtime/lru_cache.hpp"
#include "trace_test_support.hpp"   // SETENV/UNSETENV + capture_stderr

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
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

// ---------------------------------------------------------------------------
// Part A: pure cache logic with fake entries.
// ---------------------------------------------------------------------------
struct FakeEntry {
    int payload = 0;      // written once at build; verified intact later
    size_t bytes = 0;     // tracked bytes (mirrored into the cache)
};

int g_fake_builds = 0;

// Leased acquire (RAII): the entry stays PINNED for the handle's lifetime,
// so a test cannot dangle its entry pointer by construction — the raw
// pointer the old acquire_and_release helper returned belonged to an entry
// it had already made evictable. Destroy the handle (end its scope) to
// release the lease; at-rest assertions after the scope see the post-trim
// state, exactly as the old immediate release did.
class FakeLease {
public:
    FakeLease(ByteBudgetLruCache<int, FakeEntry>& c, int key,
              size_t bytes, int payload)
        : cache_(&c), key_(key) {
        entry_ = c.get_or_init_pinned(key, [bytes, payload](FakeEntry& v) {
            v.payload = payload;
            v.bytes = bytes;
            ++g_fake_builds;
            return bytes;
        });
    }
    ~FakeLease() { cache_->release(key_); }
    FakeLease(const FakeLease&) = delete;
    FakeLease& operator=(const FakeLease&) = delete;
    FakeEntry* get() const { return entry_; }

private:
    ByteBudgetLruCache<int, FakeEntry>* cache_;
    int key_;
    FakeEntry* entry_ = nullptr;
};

void test_logic_budget_lru_accounting() {
    std::printf("[A] budget / LRU order / byte accounting\n");
    // 100-byte entries, budget 1000 -> 10 resident at rest.
    ByteBudgetLruCache<int, FakeEntry> cache(1000);
    for (int i = 1; i <= 12; ++i) {
        FakeLease leased(cache, i, 100, 1000 + i);   // released at iteration end
    }

    check(cache.size() == 10, "A2: steady-state entry count saturates at 10 (got " +
          std::to_string(cache.size()) + ")");
    check(cache.bytes_in_use() == 1000, "A1: bytes_in_use == sum of entries (1000, got " +
          std::to_string(cache.bytes_in_use()) + ")");
    check(cache.bytes_in_use() <= cache.byte_budget(), "A2: budget never exceeded at rest");
    // LRU: keys 1..2 (oldest use) evicted; 3..12 resident.
    check(cache.get(1) == nullptr, "A3: oldest key evicted (1)");
    check(cache.get(2) == nullptr, "A3: second-oldest key evicted (2)");
    check(cache.get(12) != nullptr, "A3: newest key resident (12)");
    check(g_fake_builds == 12, "A2: exactly 12 builds for 12 distinct shapes (got " +
          std::to_string(g_fake_builds) + ")");
    // Accounting survives the plain get() touch above (no byte change).
    check(cache.bytes_in_use() == 1000, "A1: bytes unchanged by plain get() touch");

    // LRU by last use, not insertion: touch key 3 (currently oldest resident),
    // then one more distinct insert must evict key 4 (now the LRU), not 3.
    FakeEntry* e3 = cache.get(3);
    check(e3 != nullptr && e3->payload == 1003, "A3: key 3 hit with intact payload");
    {
        FakeLease leased(cache, 13, 100, 1013);
    }
    check(cache.get(3) != nullptr, "A3: re-used key survives (LRU is by last use)");
    check(cache.get(4) == nullptr, "A3: untouched older key evicted instead (4)");
    check(cache.size() == 10 && cache.bytes_in_use() == 1000,
          "A1/A2: size+bytes stable after churn");

    // No-growth steady state at fixed shapes: re-running the same keys must be
    // pure hits — no new builds, byte total constant.
    const int builds_before = g_fake_builds;
    for (int round = 0; round < 10; ++round)
        for (int k = 3; k <= 13; ++k) {
            if (k == 4) continue;  // evicted above; rebuilding it would be a miss
            FakeLease leased(cache, k, 100, 1000 + k);
        }
    check(g_fake_builds == builds_before, "A7: fixed-shape steady state builds nothing");
    check(cache.size() == 10 && cache.bytes_in_use() == 1000,
          "A7: fixed-shape steady state holds size+bytes constant");
}

void test_logic_pin_stability_and_oversize() {
    std::printf("[A] pin/lease stability + oversized entry\n");
    ByteBudgetLruCache<int, FakeEntry> cache(1000);

    // Lease a 600-byte entry and KEEP it pinned while newer entries churn.
    FakeEntry* leased = cache.get_or_init_pinned(100, [](FakeEntry& v) {
        v.payload = 0xC0FFEE;
        v.bytes = 600;
        ++g_fake_builds;
        return (size_t)600;
    });
    check(leased != nullptr && cache.pinned_size() == 1, "A4: entry leased (pinned)");

    // 5 x 300-byte inserts under a 1000 budget with 600 pinned: every insert
    // past the second must evict an UNPINNED entry (the leased key IS the LRU
    // entry and would be the victim without pin protection).
    for (int i = 1; i <= 5; ++i) {
        FakeLease churn(cache, i, 300, 2000 + i);
    }

    check(cache.pinned_size() == 1, "A4: lease still held after churn");
    check(cache.get(100) == leased, "A4: leased entry's address is stable across evictions");
    check(leased->payload == 0xC0FFEE, "A4: leased entry's contents intact");
    check(cache.bytes_in_use() <= cache.byte_budget(),
          "A4: bytes within budget while leased entry (600) fits it");
    cache.release(100);
    check(cache.pinned_size() == 0, "A4: lease released");

    // Oversized: 250 bytes against a 100 budget. Admitted while pinned (all
    // unpinned entries evicted), trimmed at release.
    ByteBudgetLruCache<int, FakeEntry> small(100);
    {
        FakeLease filler(small, 1, 50, 1);   // resident filler once released
    }
    FakeEntry* big = small.get_or_init_pinned(2, [](FakeEntry& v) {
        v.bytes = 250;
        v.payload = 7;
        return (size_t)250;
    });
    check(big != nullptr, "A5: oversized entry admitted while pinned");
    check(small.size() == 1 && small.bytes_in_use() == 250,
          "A5: everything else evicted; overshoot only while leased");
    small.release(2);
    check(small.size() == 0 && small.bytes_in_use() == 0,
          "A5: oversized entry trimmed at release (budget holds at rest)");
}

void test_logic_build_failure() {
    std::printf("[A] construction failure rollback\n");
    ByteBudgetLruCache<int, FakeEntry> cache(1000);
    {
        FakeLease seeded(cache, 1, 100, 11);
    }
    const size_t bytes_before = cache.bytes_in_use();
    const size_t size_before = cache.size();

    bool threw = false;
    try {
        cache.get_or_init_pinned(2, [](FakeEntry&) -> size_t {
            throw std::runtime_error("simulated graph build failure");
        });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "A6: init failure propagates");
    check(cache.size() == size_before && cache.bytes_in_use() == bytes_before,
          "A6: failed entry rolled back (no bytes, no stale slot)");
    check(cache.get(2) == nullptr, "A6: failed key not resident");
    check(cache.pinned_size() == 0, "A6: failed entry left no pin");

    // The cache must still work afterwards.
    FakeLease e(cache, 2, 100, 22);
    check(e.get() != nullptr && cache.bytes_in_use() == 200,
          "A6: later acquires work and account correctly");
    for (int i = 3; i < 6; ++i) {
        FakeLease refill(cache, i, 100, i);
    }
    check(cache.size() == 5 && cache.bytes_in_use() == 500,
          "A6: LRU/list consistent after a failed insert");
}

// (12) The env-override path behind tdt_graph_byte_budget()
// (STARLING_TDT_GRAPH_BUDGET_MB): strict whole-value validation with LOUD
// failure (stderr diagnostic — captured here) and a loud clamp on overflow.
// The old atol parse truncated ("64x" -> 64) and wrapped silently.
void test_env_budget_parsing() {
    std::printf("[A] env override parsing (env_budget_bytes)\n");
    const char* var = "STARLING_TEST_GRAPH_BUDGET_MB";
    const size_t dflt = size_t(128) << 20;

    struct Case { const char* value; size_t expect; bool loud; };
    const Case cases[] = {
        { "64",  size_t(64) << 20, false },  // valid override
        { " 32", size_t(32) << 20, false },  // leading whitespace: strtoll skips it
        { "1",   size_t(1) << 20,  false },  // minimum accepted value
        { "",    dflt,             true  },  // set-but-empty
        { "foo", dflt,             true  },  // pure garbage
        { "64x", dflt,             true  },  // trailing garbage (atol read 64)
        { "0",   dflt,             true  },  // zero: loud reject (was: silent default)
        { "-5",  dflt,             true  },  // negative: loud reject
    };
    for (const Case& c : cases) {
        SETENV(var, c.value);
        size_t got = 0;
        const std::string err = capture_stderr(
            [&] { got = env_budget_bytes(var, dflt); });
        check(got == c.expect,
              "ENV: '" + std::string(c.value) + "' -> " +
                  std::to_string(c.expect) + " bytes (got " +
                  std::to_string(got) + ")");
        const bool loud = err.find("starling:") != std::string::npos;
        check(loud == c.loud,
              "ENV: '" + std::string(c.value) + "' diagnostic " +
                  (c.loud ? "emitted" : "silent") + " (stderr: '" + err + "')");
    }
    UNSETENV(var);
    size_t got = 0;
    const std::string err = capture_stderr(
        [&] { got = env_budget_bytes(var, dflt); });
    check(got == dflt && err.empty(), "ENV: unset -> default, silently");

    // Overflow: exactly the largest representable MiB (SIZE_MAX >> 20) is
    // accepted as-is; one more — and an out-of-long-long value — clamp
    // loudly to the same budget instead of wrapping the shift.
    const size_t max_mib = SIZE_MAX >> 20;
    const size_t max_budget = max_mib << 20;
    SETENV(var, std::to_string(max_mib).c_str());
    got = 0;
    capture_stderr([&] { got = env_budget_bytes(var, dflt); });
    check(got == max_budget, "ENV: boundary max_mib accepted unclamped");
    for (const char* v : { std::to_string(max_mib + 1).c_str(),
                           "99999999999999999999" }) {
        SETENV(var, v);
        got = 0;
        const std::string clamp_err = capture_stderr(
            [&] { got = env_budget_bytes(var, dflt); });
        check(got == max_budget &&
                  clamp_err.find("clamping") != std::string::npos,
              "ENV: overflow '" + std::string(v) + "' clamps loudly to max");
    }
    UNSETENV(var);
}

// (13) A zero byte budget is a construction error, not a silent 1-byte clamp.
void test_zero_budget_rejected() {
    std::printf("[A] zero byte budget rejected at construction\n");
    bool threw = false;
    try {
        ByteBudgetLruCache<int, FakeEntry> zero(0);
        (void)zero;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "CTOR: ByteBudgetLruCache(0) throws std::invalid_argument");
}

// (14) Mass eviction: one insert that must evict MANY entries at once, with
// pinned (leased) entries older than every victim. The single-pass trim must
// produce exactly the same victim set as the per-victim rescan did: the
// LRU-ordered unpinned prefix whose bytes bring the total within budget,
// pins skipped in place.
void test_logic_mass_eviction_skips_pins() {
    std::printf("[A] mass eviction: victim set with interleaved pins\n");
    ByteBudgetLruCache<int, FakeEntry> cache(1000);

    // Two OLD entries leased for the whole test: they sit at the LRU end and
    // must be skipped by every eviction pass.
    FakeEntry* p1 = cache.get_or_init_pinned(1, [](FakeEntry& v) {
        v.bytes = 100; v.payload = 9001; return (size_t)100;
    });
    FakeEntry* p2 = cache.get_or_init_pinned(2, [](FakeEntry& v) {
        v.bytes = 100; v.payload = 9002; return (size_t)100;
    });
    // Six unpinned fillers between the pins and the incoming entry.
    for (int i = 3; i <= 8; ++i) {
        FakeLease filler(cache, i, 100, i);
    }
    check(cache.size() == 8 && cache.bytes_in_use() == 800,
          "MASS: pre-churn state (8 entries, 800 bytes)");

    // A 700-byte insert (1500 total vs a 1000 budget) must mass-evict the
    // unpinned LRU prefix 3..7 (500 bytes) to land exactly at budget, leaving
    // the pins and the newest filler alone.
    FakeEntry* big = cache.get_or_init_pinned(9, [](FakeEntry& v) {
        v.bytes = 700; v.payload = 9009; return (size_t)700;
    });
    check(big != nullptr && cache.pinned_size() == 3,
          "MASS: oversized insert admitted while pinned (3 leases live)");
    cache.release(9);
    check(cache.pinned_size() == 2, "MASS: insert lease released");

    for (int k = 3; k <= 7; ++k)
        check(cache.get(k) == nullptr,
              "MASS: unpinned LRU prefix evicted (" + std::to_string(k) + ")");
    check(cache.size() == 4 && cache.bytes_in_use() == 1000,
          "MASS: lands at exactly the budget with pins + newest filler + insert");
    check(cache.get(8) != nullptr && cache.get(9) != nullptr,
          "MASS: newest filler and the inserted entry stay resident");
    check(cache.get(1) == p1 && p1->payload == 9001,
          "MASS: oldest pin survived with contents intact");
    check(cache.get(2) == p2 && p2->payload == 9002,
          "MASS: second pin survived with contents intact");
    cache.release(1);
    cache.release(2);
    check(cache.pinned_size() == 0 &&
              cache.bytes_in_use() <= cache.byte_budget(),
          "MASS: pins released; budget holds at rest");
}

// ---------------------------------------------------------------------------
// Part B: real ReplayGraph entries on the CPU backend.
// ---------------------------------------------------------------------------
struct ShapeKey {
    int T, K;
    bool operator==(const ShapeKey& o) const { return T == o.T && K == o.K; }
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey& k) const noexcept {
        return (size_t)k.T * 257u + (size_t)k.K;
    }
};

constexpr size_t kGraphVec = 256;   // f32 elements per input vector

// One entry = one tiny captured graph of the shape the TDT cache stores:
// ReplayGraph + stable host backing, tracked bytes = device_alloc_bytes().
struct GraphEntry {
    std::unique_ptr<ReplayGraph> rg;
    std::vector<float> host_x;    // [kGraphVec] f32 input (stable address)
    float scale = 0.0f, bias = 0.0f;
    int id = 0;
};

int g_graph_builds = 0;

// y[i] = x[i] * scale + bias  (elementwise; scale/bias are graph inputs from
// the entry's own stable storage, exactly like the TDT constant tables).
void build_shape_entry(Backend& backend, GraphEntry& ge, int id,
                       float scale, float bias) {
    ge.id = id;
    ge.scale = scale;
    ge.bias = bias;
    ge.host_x.assign(kGraphVec, 0.0f);
    ge.rg = std::make_unique<ReplayGraph>(backend,
        [&ge](ggml_context* ctx) -> ggml_tensor* {
            int64_t nv[1] = {(int64_t)kGraphVec};
            int64_t n1[1] = {1};
            ggml_tensor* x = graph_input_tensor(ctx, GGML_TYPE_F32, 1, nv,
                                                ge.host_x.data(),
                                                kGraphVec * sizeof(float));
            ggml_tensor* s = graph_input_tensor(ctx, GGML_TYPE_F32, 1, n1,
                                                &ge.scale, sizeof(float));
            ggml_tensor* b = graph_input_tensor(ctx, GGML_TYPE_F32, 1, n1,
                                                &ge.bias, sizeof(float));
            ggml_tensor* sx = ggml_mul(ctx, ggml_repeat(ctx, s, x), x);
            return ggml_add(ctx, sx, ggml_repeat(ctx, b, x));
        });
    ++g_graph_builds;
}

// Replay `ge` with x[i] = base + i and check y[i] == scale*x + bias exactly.
bool replay_and_check(GraphEntry& ge, float base) {
    for (size_t i = 0; i < kGraphVec; ++i) ge.host_x[i] = base + (float)i;
    ge.rg->set_input(0, ge.host_x.data(), ge.host_x.size() * sizeof(float));
    ge.rg->set_input(1, &ge.scale, sizeof(float));
    ge.rg->set_input(2, &ge.bias, sizeof(float));
    std::vector<float> out;
    if (!ge.rg->compute(out) || out.size() != kGraphVec) return false;
    for (size_t i = 0; i < kGraphVec; ++i)
        if (out[i] != ge.scale * (base + (float)i) + ge.bias) return false;
    return true;
}

void test_real_graphs() {
    std::printf("[B] real ReplayGraph entries on CPU backend\n");
    Backend backend(2);
    std::printf("[B] backend=%s\n", backend.device_name());

    // Probe one graph to size the budget deterministically (~4 entries fit).
    size_t entry_bytes = 0;
    {
        GraphEntry probe;
        build_shape_entry(backend, probe, -99, 1.0f, 0.0f);
        entry_bytes = probe.rg->device_alloc_bytes();
        check(entry_bytes > 0, "B8: device_alloc_bytes() > 0 for a real graph");
        check(entry_bytes >= kGraphVec * sizeof(float),
              "B8: tracked bytes cover at least the input vector");
        check(replay_and_check(probe, 1.0f),
              "B8: probe graph computes x*scale+bias exactly");
    }
    if (entry_bytes == 0) {
        // Guard the budget-sizing step at init: B8 above already recorded
        // the failure, and a zero budget is a construction error — bail out
        // of Part B instead of failing every later assertion.
        return;
    }
    const size_t budget = entry_bytes * 4 + entry_bytes / 2;   // ~4.5 entries
    std::printf("[B] entry_bytes=%zu budget=%zu\n", entry_bytes, budget);

    ByteBudgetLruCache<ShapeKey, GraphEntry, ShapeKeyHash> cache(budget);

    // (9) Many distinct (T,K) shapes wrapping REAL captured graphs: the budget
    // holds at rest after every acquire/release cycle.
    constexpr int kShapes = 40;
    bool budget_ok = true;
    for (int i = 0; i < kShapes; ++i) {
        ShapeKey key{ 100 + i, (i % 2) ? 16 : 96 };
        cache.get_or_init_pinned(key, [&](GraphEntry& ge) {
            build_shape_entry(backend, ge, i, 1.0f + (float)i, 0.5f * (float)i);
            return ge.rg->device_alloc_bytes();
        });
        cache.release(key);
        if (cache.bytes_in_use() > cache.byte_budget()) budget_ok = false;
    }
    check(budget_ok, "B9: byte budget holds at rest across 40 distinct (T,K) shapes");
    check(cache.size() >= 1 && cache.size() < (size_t)kShapes,
          "B9: entry count bounded well below the distinct-shape count (got " +
          std::to_string(cache.size()) + ")");
    check(g_graph_builds == kShapes + 1, "B9: one build per distinct shape (probe incl.)");

    // (10) Pointer + replay stability of a LEASED graph across evictions.
    ShapeKey held{ 500, 16 };
    GraphEntry* leased = cache.get_or_init_pinned(held, [&](GraphEntry& ge) {
        build_shape_entry(backend, ge, -1, 3.0f, 7.0f);
        return ge.rg->device_alloc_bytes();
    });
    ReplayGraph* leased_rg = leased->rg.get();
    const float* leased_x = leased->host_x.data();
    check(replay_and_check(*leased, 1.0f), "B10: leased graph replays correctly");
    // Churn enough distinct shapes to force repeated evictions of entries
    // newer AND older than the leased one.
    for (int i = 0; i < kShapes; ++i) {
        ShapeKey key{ 1000 + i, (i % 3) + 1 };
        cache.get_or_init_pinned(key, [&](GraphEntry& ge) {
            build_shape_entry(backend, ge, i, 1.0f, 0.0f);
            return ge.rg->device_alloc_bytes();
        });
        cache.release(key);
    }
    check(cache.pinned_size() == 1, "B10: lease survived the churn");
    check(leased->rg.get() == leased_rg,
          "B10: leased ReplayGraph pointer stable across evictions");
    check(leased->host_x.data() == leased_x,
          "B10: leased host backing pointer stable across evictions");
    check(cache.bytes_in_use() <=
              std::max(cache.byte_budget(), leased->rg->device_alloc_bytes()),
          "B10: bytes <= max(budget, pinned bytes) while leased");
    check(replay_and_check(*leased, 50.0f),
          "B10: leased graph still computes correct results after other "
          "entries' graphs were freed (private gallocr isolation)");
    cache.release(held);
    check(cache.bytes_in_use() <= cache.byte_budget(),
          "B10: budget restored at rest after release");

    // (11) Steady state at fixed shapes: the two most recently inserted churn
    // keys are guaranteed resident (budget fits ~4); repeating them must be
    // pure hits — no rebuilds, constant size+bytes.
    const int builds_before = g_graph_builds;
    const size_t bytes_before = cache.bytes_in_use();
    const size_t size_before = cache.size();
    const ShapeKey rep1{ 1000 + kShapes - 2, (kShapes - 2) % 3 + 1 };
    const ShapeKey rep2{ 1000 + kShapes - 1, (kShapes - 1) % 3 + 1 };
    // Guard B11's steady-state assumption BEFORE relying on it: the budget
    // (~4.5 entries) must be holding the two most recent churn keys plus the
    // re-released `held` key resident after the churn. If that ever fails,
    // the repeat loop below would legitimately REBUILD (a miss, not a
    // steady-state violation) and the no-build assertion would misreport the
    // cause — this check names it.
    check(cache.get(rep1) != nullptr && cache.get(rep2) != nullptr &&
              cache.get(held) != nullptr,
          "B11 precondition: repeat keys resident after churn");
    for (int round = 0; round < 10; ++round) {
        for (const ShapeKey& k : { rep1, rep2, held }) {
            cache.get_or_init_pinned(k, [&](GraphEntry&) -> size_t {
                check(false, "B11: steady-state repeat must not rebuild");
                return 0;
            });
            cache.release(k);
        }
    }
    check(g_graph_builds == builds_before, "B11: steady state builds no new graphs");
    check(cache.bytes_in_use() == bytes_before && cache.size() == size_before,
          "B11: steady state holds bytes+size constant");
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("tdt_graph_budget_test: start\n");

    test_logic_budget_lru_accounting();
    test_logic_pin_stability_and_oversize();
    test_logic_build_failure();
    test_env_budget_parsing();
    test_zero_budget_rejected();
    test_logic_mass_eviction_skips_pins();
    test_real_graphs();

    if (g_failures) {
        std::printf("tdt_graph_budget_test: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("tdt_graph_budget_test: PASS (byte budget, LRU, pins, accounting)\n");
    return 0;
}
