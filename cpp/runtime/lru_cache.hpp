// Bounded replay caches keep entry addresses stable because captured graphs
// retain pointers into their input pools. Callers serialize cache access.

#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "trace.hpp"

namespace starling::ggml {

// Maximum retained shapes per model cache; configurable below.
constexpr size_t kDefaultReplayCacheSize = 16;

// Reduced cap for devices whose single memory window has to hold the weights
// AND every cached per-shape replay graph (integrated GPUs, small-VRAM cards).
constexpr size_t kSharedMemoryReplayCacheSize = 2;

// Device-aware fallback for replay_cache_size() when STARLING_REPLAY_CACHE_SIZE
// is unset; defined in backend.cpp from the active device's properties.
size_t device_replay_cache_default();

// Process-global cache capacity, read from STARLING_REPLAY_CACHE_SIZE (>=1) on
// each cache's first construction. Reading at first use (rather than once at
// process start) lets a test or harness set the env before loading a model.
inline size_t replay_cache_size() {
    if (const char* env = std::getenv("STARLING_REPLAY_CACHE_SIZE")) {
        long v = std::atol(env);
        if (v >= 1) return (size_t)v;
    }
    return device_replay_cache_default();
}

// Parse an environment variable as a mebibyte count into a byte budget (the
// STARLING_*_BUDGET_MB pattern). Strict-ish, strtoll grammar: optional leading
// whitespace and +/- sign, then decimal digits to the END of the value
// (leading whitespace is accepted and " 32" == "32" is deliberate — pinned by
// the test suite; a sign alone is not digits and is rejected), and the number
// must land in [1, SIZE_MAX >> 20] MiB. Anything else — garbage like "foo",
// trailing junk like "64x", an empty value, zero, or a negative — is REJECTED
// with a one-line diagnostic on stderr and `default_bytes` is used (a
// malformed knob must not silently disable or shrink the bound; atol's silent
// truncation is exactly what this replaces).
// A parseable value too large to represent (ERANGE, or > SIZE_MAX >> 20 —
// whose << 20 would wrap) is loudly CLAMPED to the largest representable
// budget, (SIZE_MAX >> 20) MiB, rather than rejected: the operator asked for
// "a lot", so they get the most the type can express, with a diagnostic.
inline size_t env_budget_bytes(const char* var, size_t default_bytes) {
    const char* e = std::getenv(var);
    if (!e) return default_bytes;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(e, &end, 10);
    const size_t max_mib = SIZE_MAX >> 20;
    if (end == e || *end != '\0' || v <= 0) {
        std::fprintf(stderr,
                     "starling: %s='%s' is invalid (expected an integer "
                     "number of MiB >= 1); using the default budget "
                     "(%zu bytes)\n",
                     var, e, default_bytes);
        return default_bytes;
    }
    if (errno == ERANGE || (unsigned long long)v > max_mib) {
        std::fprintf(stderr,
                     "starling: %s='%s' MiB exceeds the largest "
                     "representable budget; clamping to %zu MiB\n",
                     var, e, max_mib);
        return max_mib << 20;
    }
    return (size_t)v << 20;
}

// A bounded LRU map. `Value` is typically an entry struct holding a
// GraphInputPool + unique_ptr<ReplayGraph>; it must be default-constructible
// (get_or_init places a default value first, then fills it in place).
//
// `label` (optional) names the cache in the STARLING_TRACE records ("cache"
// events: hit/miss/evict + occupancy). Null (the default) = unlabeled: the
// cache stays silent, so only the caches the trace scopes over need changes.
template <typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class LruCache {
public:
    explicit LruCache(size_t capacity, const char* label = nullptr)
        : capacity_(capacity == 0 ? 1 : capacity), label_(label) {}

    size_t size() const { return map_.size(); }

    // On hit: mark MRU and return a pointer to the value (stable until the next
    // non-const operation that evicts THIS key). On miss: return nullptr (the
    // caller may then call get_or_init).
    Value* get(const Key& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;  // the get_or_init that follows reports the miss
        touch(it);
        trace_hit();
        return &it->second.second;
    }

    // On hit: mark MRU and return a pointer to the existing value (init is NOT
    // called). On miss: evict LRU until below capacity, insert a default value,
    // call `init(value)` so the caller fills the pool + builds the ReplayGraph
    // against the value's now-stable address, and return a pointer to it.
    // The returned pointer stays valid until the key is evicted.
    template <typename Init>
    Value* get_or_init(const Key& key, Init&& init) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            touch(it);
            trace_hit();
            return &it->second.second;
        }
        const size_t evicted = trim();
        lru_.push_front(key);
        auto inserted = map_.end();
        try {
            inserted = map_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(key),
                std::forward_as_tuple(lru_.begin(), Value())).first;
            init(inserted->second.second);
            trace_miss(evicted);
            return &inserted->second.second;
        } catch (...) {
            if (inserted != map_.end()) map_.erase(inserted);
            lru_.pop_front();
            throw;
        }
    }

    void clear() {
        map_.clear();
        lru_.clear();
    }

private:
    using ListIt = typename std::list<Key>::iterator;
    using MapVal = std::pair<ListIt, Value>;
    using Map = std::unordered_map<Key, MapVal, Hash, KeyEqual>;

    void touch(typename Map::iterator it) {
        lru_.splice(lru_.begin(), lru_, it->second.first);
        it->second.first = lru_.begin();
    }

    // Evict LRU entries while size >= capacity, so a following insert lands at
    // <= capacity. No-op below capacity. Returns the number of victims — the
    // trace miss record reports them.
    size_t trim() {
        size_t evicted = 0;
        while (map_.size() >= capacity_ && !lru_.empty()) {
            Key victim = lru_.back();
            lru_.pop_back();
            map_.erase(victim);  // destroys the value (and its ReplayGraph)
            ++evicted;
        }
        return evicted;
    }

    // The cache layer cannot see the device, so mem_free stays "unavailable"
    // (trace.hpp rule: never fabricate). Occupancy (size/cap) is the available
    // allocation counter this layer actually owns.
    void trace_hit() {
        if (label_) trace::cache_event(label_, "hit", 0, map_.size(), capacity_, -1);
    }
    void trace_miss(size_t evicted) {
        if (label_) trace::cache_event(label_, "miss", evicted, map_.size(), capacity_, -1);
    }

    size_t capacity_;
    const char* label_ = nullptr;
    std::list<Key> lru_;
    Map map_;
};

// A byte-aware bounded LRU with pin (lease) semantics, for caches whose
// entries hold captured graphs whose DEVICE cost — not their count — is the
// resource that must stay bounded (issue #177 / experiment S02).
//
// Semantics:
//   * Each entry carries a caller-supplied tracked-bytes figure (the `init`
//     callback returns it after filling the value; the figure must not change
//     while the entry is resident). The cache enforces a BYTE budget, not an
//     entry cap.
//   * Address stability: values live in unordered_map nodes and are never
//     moved; a returned Value* stays valid until ITS key is evicted/erased.
//   * Lease: get_pinned/get_or_init_pinned mark the entry pinned; release()
//     unpins. PINNED ENTRIES ARE NEVER EVICTED — a caller replaying a captured
//     graph keeps stable pointers for the whole lease even while other
//     entries are inserted and evicted. Eviction runs at insert time (to make
//     room) and at release time (to trim), by LRU of last use, skipping pins.
//   * Oversized entries: an entry whose bytes alone exceed the whole budget is
//     still admitted (refusing would rebuild it on every use); every unpinned
//     entry is evicted first and the overshoot is trimmed at release. The
//     invariant is bytes_in_use() <= max(byte_budget(), pinned bytes) — i.e.
//     the budget always holds AT REST, and is exceeded only by live leases.
//   * Failure: a throwing build never poisons the cache — the rollback bullet
//     above erases the half-built entry and the exception propagates. (Callers
//     that can leave a VALID-but-unusable entry must poison the value itself,
//     like the encoder's pos-projection failure.)
//   * A ZERO byte budget is rejected at construction (std::invalid_argument)
//     rather than silently clamped: the env parser above already guarantees a
//     validated budget on the production path, so zero can only be a
//     direct-construction bug — fail loudly instead of running a cache that
//     must evict everything.
//   * Zero-byte floor (R22): an entry whose `init` reports ZERO tracked bytes
//     has, by this cache's contract, an UNKNOWN device cost, not a free one —
//     on the TDT path zero means the ReplayGraph took the ggml_backend_sched
//     fallback (no private gallocr buffer to measure; see tdt_multistep.cpp's
//     accounting note). Tracking it as 0 would silently degenerate the byte
//     budget into an unbounded entry cache. Such entries are therefore charged
//     a conservative floor (`zero_byte_floor`, default
//     kZeroByteEntryFloorBytes) AND hard-capped in COUNT (`floored_cap`,
//     default kFlooredEntryCap) so an all-sched workload stays bounded no
//     matter how large the byte budget is. The first floored insert emits a
//     one-time stderr diagnostic naming the accounting that applies. Passing
//     0 for either knob opts out of that guard (documented; the defaults keep
//     every cache bounded).
//   * Rollback: if `init` throws, OR anything between the accounting commit
//     and the return throws (trim_over_budget's victim-vector allocation,
//     the trace call), the half-built entry is erased with its accounted
//     bytes, pin, floored count, and LRU node all restored — bytes_in_use()
//     is never left inflated by a failed insert. The LRU node is erased via
//     the entry's STORED list iterator, not pop_front(), so even a reentrant
//     touch from `init` (which would splice the node away from the front)
//     rolls back the right node.
//
// As above: callers serialize cache access; `label` names the cache in the
// STARLING_TRACE "cache" records (size/capacity are BYTES here, evicted is an
// entry count). get() and the lease paths trace the same hit/miss event
// kinds; lease-path events carry an extra "pin":1 marker so a leased hit is
// distinguishable from a plain-LRU hit in the records.

// Conservative charge for an entry whose `init` reports ZERO tracked bytes
// (R22; see the zero-byte floor bullet in the class comment). 8 MiB sits at
// the top of the observed per-(T,K) TDT graph range (roughly 1.5-3 MiB
// short/medium, ~4-8 MiB long), so it never UNDER-counts a sched-path graph
// — over-charging only evicts earlier, the safe direction. Compile-time
// nonzero so the default cache is bounded by construction: budget/floor is
// an upper bound on all-floored entry count.
constexpr size_t kZeroByteEntryFloorBytes = size_t(8) << 20;
// Hard COUNT cap on floored entries (R22): with an operator-sized byte budget
// (STARLING_TDT_GRAPH_BUDGET_MB clamps up to SIZE_MAX>>20 MiB) budget/floor
// alone can still admit a pathological number of zero-byte entries; 16
// matches the encoder entry-LRU's default order (kDefaultReplayCacheSize).
constexpr size_t kFlooredEntryCap = 16;
static_assert(kZeroByteEntryFloorBytes > 0, "zero-byte floor must be nonzero");
static_assert(kFlooredEntryCap >= 1, "floored entry cap must be >= 1");

namespace detail {
// Fault-injection seam for tests (R22): when non-null, invoked at the top of
// every ByteBudgetLruCache::trim_over_budget call. A hook that throws
// simulates the std::vector victim-buffer allocation failure the insert
// rollback must survive (bytes_/floored_ already committed at that point).
// MUST remain null in production.
inline void (*trim_fault_hook)() = nullptr;
}  // namespace detail

template <typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class ByteBudgetLruCache {
public:
    // `zero_byte_floor` / `floored_cap` configure the R22 zero-byte guard
    // (see the class comment); the defaults bound every cache. Pass 0 to opt
    // out of a guard — only for callers whose init provably never reports 0.
    explicit ByteBudgetLruCache(size_t byte_budget, const char* label = nullptr,
                                size_t zero_byte_floor = kZeroByteEntryFloorBytes,
                                size_t floored_cap = kFlooredEntryCap)
        : label_(label), floor_(zero_byte_floor), floored_cap_(floored_cap) {
        if (byte_budget == 0)
            throw std::invalid_argument(
                "ByteBudgetLruCache: byte budget must be at least 1 byte");
        budget_ = byte_budget;
    }

    size_t size() const { return map_.size(); }
    size_t byte_budget() const { return budget_; }
    size_t bytes_in_use() const { return bytes_; }
    size_t pinned_size() const { return pinned_; }
    // Entries currently charged the zero-byte floor (R22 observability; the
    // floored_cap bounds this count).
    size_t floored_size() const { return floored_; }

    // Plain LRU lookup (touch, no pin). Stable until a non-const operation
    // evicts THIS key. Nullptr on miss (the get_or_init that follows reports
    // the miss — same rule as LruCache::get). Traces the same "hit" event
    // kind the lease paths emit; the records stay distinguishable by the pin
    // marker (present only on lease-path events).
    Value* get(const Key& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        touch(it);
        trace("hit", 0);
        return &entry(it).value;
    }

    // Lease lookup: like get(), but the entry is pinned until release(key).
    // Nullptr on miss. Re-pinning an already-pinned entry is a no-op (single
    // lease per key; serialized callers acquire once per use).
    Value* get_pinned(const Key& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        touch(it);
        pin(it);
        trace("hit", 0, /*pins=*/true);
        return &entry(it).value;
    }

    // Lease fetch-or-build. On miss: insert a default value at a stable
    // address, PIN it (so the trim below can never drop the entry being
    // built), call `init(value)` — which fills the value and returns its
    // tracked bytes — account the bytes (a ZERO return is charged the
    // zero-byte floor; see the class comment), then evict unpinned LRU
    // entries until back under budget (and, for a floored insert, down to
    // the floored-entry cap). Any throw from here onward — init, the
    // accounting-adjacent trim, the trace — rolls the entry back completely
    // (bytes, pin, floored count, LRU node) and propagates.
    template <typename Init>
    Value* get_or_init_pinned(const Key& key, Init&& init) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            touch(it);
            pin(it);
            trace("hit", 0, /*pins=*/true);
            return &entry(it).value;
        }
        // try_emplace default-constructs the value IN PLACE (no move): the
        // entry types used here (captured-graph holders) may be move-suppressed
        // (deleted copy ctors), so the LruCache emplace pattern would not work.
        lru_.push_front(key);
        auto inserted = map_.end();
        try {
            inserted = map_.try_emplace(key).first;
            inserted->second.first = lru_.begin();
            pin(inserted);
            Entry& e = entry(inserted);
            const size_t reported = init(e.value);
            e.floored = (reported == 0 && floor_ > 0);
            e.bytes = e.floored ? floor_ : reported;
            if (e.floored) {
                ++floored_;
                warn_floor_once();
            }
            bytes_ += e.bytes;
            const size_t evicted = trim_over_budget(/*enforce_floored_cap=*/e.floored);
            trace("miss", evicted, /*pins=*/true);
            return &e.value;
        } catch (...) {
            if (inserted != map_.end()) {
                // Rollback in reverse order of the commits above. bytes_ -=
                // undoes the accounting even when the throw happened AFTER
                // bytes_ += (e.g. trim_over_budget's victim-vector
                // allocation): without it a failed insert would permanently
                // inflate bytes_in_use() and the cache would chronically
                // over-evict (R22 review finding).
                Entry& e = entry(inserted);
                if (e.pinned) --pinned_;
                bytes_ -= e.bytes;
                if (e.floored) --floored_;
                // Erase the LRU node by its STORED iterator: today init never
                // calls back into the cache so the node is still at the
                // front, but a reentrant touch() would splice it away and a
                // pop_front() would then drop the WRONG key and leave this
                // node's ListIt dangling (R22 review finding). The stored
                // iterator is authoritative in both cases.
                lru_.erase(inserted->second.first);
                map_.erase(inserted);  // destroys the half-built value
            } else {
                // try_emplace itself threw: the node pushed above is still
                // the front by construction (nothing ran that could move it).
                lru_.pop_front();
            }
            throw;
        }
    }

    // End the lease. If the cache is over budget afterwards (e.g. an
    // oversized entry just unpinned), evict unpinned LRU entries until it is
    // not. Releasing a key that is not resident/pinned is a no-op.
    void release(const Key& key) {
        auto it = map_.find(key);
        if (it == map_.end() || !entry(it).pinned) return;
        entry(it).pinned = false;
        --pinned_;
        const size_t evicted = trim_over_budget();
        if (evicted > 0) trace("evict", evicted);
    }

    void clear() {
        map_.clear();
        lru_.clear();
        bytes_ = 0;
        pinned_ = 0;
        floored_ = 0;
    }

private:
    struct Entry {
        Value value;
        size_t bytes = 0;   // tracked bytes (init's return, or the floor)
        bool pinned = false;
        bool floored = false;  // bytes is the zero-byte floor, not a measurement
    };
    using ListIt = typename std::list<Key>::iterator;
    using MapVal = std::pair<ListIt, Entry>;
    using Map = std::unordered_map<Key, MapVal, Hash, KeyEqual>;

    // The map's mapped value is MapVal (ListIt + Entry pair); the Entry lives
    // at .second.second (same shape as LruCache's MapVal).
    static Entry& entry(typename Map::iterator it) { return it->second.second; }

    void touch(typename Map::iterator it) {
        lru_.splice(lru_.begin(), lru_, it->second.first);
        it->second.first = lru_.begin();
    }
    void pin(typename Map::iterator it) {
        if (!entry(it).pinned) { entry(it).pinned = true; ++pinned_; }
    }

    // Evict unpinned LRU entries while over budget. SINGLE back-to-front
    // pass: walk LRU->MRU once, collecting unpinned victims (oldest first)
    // until the projected remaining bytes fit the budget, then erase them in
    // one sweep — one rescan per victim made mass eviction quadratic in the
    // entry count. Pinned entries are skipped in place (a lease is never a
    // victim); if the walk exhausts the list while still over budget,
    // everything left is pinned (the at-rest invariant's escape hatch for
    // oversized live leases). No-op when already within budget. Returns the
    // number of victims. Invariants kept: bytes_in_use() <= budget at rest
    // (except live leases), pinned entries never evicted, LRU victim order.
    //
    // With enforce_floored_cap (set by the insert path when the just-inserted
    // entry was floored), the pass additionally evicts until the floored
    // ENTRY count is within floored_cap_ — the hard count bound that keeps an
    // all-zero-byte (sched-path) workload bounded even when the byte budget
    // itself is huge (R22).
    size_t trim_over_budget(bool enforce_floored_cap = false) {
        // Test-only fault seam (see detail::trim_fault_hook): a throwing hook
        // lands in the caller's rollback path with the accounting committed.
        if (detail::trim_fault_hook) detail::trim_fault_hook();
        std::vector<typename Map::iterator> victims;
        size_t victim_bytes = 0;
        size_t victim_floored = 0;
        for (auto lit = lru_.rbegin(); lit != lru_.rend(); ++lit) {
            if (bytes_ - victim_bytes <= budget_ &&
                (!enforce_floored_cap || floored_ - victim_floored <= floored_cap_))
                break;   // budget (and floored cap, if enforced) met
            auto mit = map_.find(*lit);
            if (mit == map_.end() || entry(mit).pinned) continue;
            victim_bytes += entry(mit).bytes;
            if (entry(mit).floored) ++victim_floored;
            victims.push_back(mit);
        }
        // The sweep is safe: erasing an unordered_map node invalidates only
        // its own iterator, and list erase does not invalidate other list
        // iterators, so each victim's MapVal (holding its ListIt) stays
        // readable until its own erasure.
        for (auto mit : victims) {
            bytes_ -= entry(mit).bytes;
            if (entry(mit).floored) --floored_;
            lru_.erase(mit->second.first);
            map_.erase(mit);
        }
        return victims.size();
    }

    // One-time-per-cache diagnostic when the zero-byte floor is first applied
    // (R22): zero tracked bytes means the device cost is UNKNOWN (sched-path
    // graph), not free — say so, and say which accounting now applies.
    void warn_floor_once() {
        if (floor_warned_) return;
        floor_warned_ = true;
        std::fprintf(stderr,
                     "starling: %s: entry reports 0 tracked device bytes "
                     "(device accounting unavailable - typically a "
                     "ggml_backend_sched-allocated graph with no private "
                     "buffer to measure); charging a conservative %zu-byte "
                     "floor per such entry and hard-capping floored entries "
                     "at %zu, so the byte budget stays bounded\n",
                     label_ ? label_ : "byte-budget cache", floor_,
                     floored_cap_);
    }

    // Field mapping into the shared cache_event schema: size=bytes_ and
    // cap=budget_ are BYTES here (not entries — see the class comment), and
    // `evicted` is an ENTRY count. `pins` marks hit/miss events from the
    // lease paths (rendered "pin":1) so they are distinguishable from the
    // plain-LRU get()'s unmarked "hit" of the same kind.
    void trace(const char* op, size_t evicted, bool pins = false) const {
        if (label_)
            trace::cache_event(label_, op, evicted, bytes_, budget_, -1, pins);
    }

    size_t budget_;
    const char* label_ = nullptr;
    size_t bytes_ = 0;
    size_t pinned_ = 0;
    // Zero-byte accounting guard (R22): floor charged per zero-reporting
    // entry, hard count cap on floored entries, current floored count, and
    // the one-time diagnostic latch.
    size_t floor_;
    size_t floored_cap_;
    size_t floored_ = 0;
    bool floor_warned_ = false;
    std::list<Key> lru_;
    Map map_;
};

}  // namespace starling::ggml
