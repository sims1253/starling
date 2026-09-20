// Bounded replay caches keep entry addresses stable because captured graphs
// retain pointers into their input pools. Callers serialize cache access.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <unordered_map>
#include <utility>

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
//   * Failure: if `init` throws, the half-built entry is erased (no bytes
//     retained) and the exception propagates — a failed build never poisons
//     the cache. (Callers that can leave a VALID-but-unusable entry must
//     poison the value itself, like the encoder's pos-projection failure.)
//
// As above: callers serialize cache access; `label` names the cache in the
// STARLING_TRACE "cache" records (size/capacity are BYTES here, evicted is an
// entry count).
template <typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class ByteBudgetLruCache {
public:
    explicit ByteBudgetLruCache(size_t byte_budget, const char* label = nullptr)
        : budget_(byte_budget == 0 ? 1 : byte_budget), label_(label) {}

    size_t size() const { return map_.size(); }
    size_t byte_budget() const { return budget_; }
    size_t bytes_in_use() const { return bytes_; }
    size_t pinned_size() const { return pinned_; }

    // Plain LRU lookup (touch, no pin). Stable until a non-const operation
    // evicts THIS key. Nullptr on miss.
    Value* get(const Key& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        touch(it);
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
        trace("hit", 0);
        return &entry(it).value;
    }

    // Lease fetch-or-build. On miss: insert a default value at a stable
    // address, PIN it (so the trim below can never drop the entry being
    // built), call `init(value)` — which fills the value and returns its
    // tracked bytes — account the bytes, then evict unpinned LRU entries
    // until back under budget. If `init` throws, the entry is erased and the
    // exception rethrown (no bytes retained).
    template <typename Init>
    Value* get_or_init_pinned(const Key& key, Init&& init) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            touch(it);
            pin(it);
            trace("hit", 0);
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
            entry(inserted).bytes = init(entry(inserted).value);
            bytes_ += entry(inserted).bytes;
            const size_t evicted = trim_over_budget();
            trace("miss", evicted);
            return &entry(inserted).value;
        } catch (...) {
            if (inserted != map_.end()) {
                if (entry(inserted).pinned) --pinned_;
                map_.erase(inserted);  // destroys the half-built value
            }
            lru_.pop_front();
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
    }

private:
    struct Entry {
        Value value;
        size_t bytes = 0;   // tracked bytes (init's return value)
        bool pinned = false;
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

    // Evict unpinned LRU entries while over budget. Stops when only pinned
    // entries remain (the at-rest invariant's escape hatch for oversized
    // live leases). Returns the number of victims.
    size_t trim_over_budget() {
        size_t evicted = 0;
        while (bytes_ > budget_) {
            bool evicted_one = false;
            // LRU-back-to-front scan for the first unpinned entry.
            for (auto lit = lru_.rbegin(); lit != lru_.rend(); ++lit) {
                auto mit = map_.find(*lit);
                if (mit == map_.end() || entry(mit).pinned) continue;
                bytes_ -= entry(mit).bytes;
                lru_.erase(std::next(lit).base());
                map_.erase(mit);
                ++evicted;
                evicted_one = true;
                break;                       // rescan from the new LRU back
            }
            if (!evicted_one) break;         // everything left is pinned
        }
        return evicted;
    }

    void trace(const char* op, size_t evicted) const {
        if (label_)
            trace::cache_event(label_, op, evicted, bytes_, budget_, -1);
    }

    size_t budget_;
    const char* label_ = nullptr;
    size_t bytes_ = 0;
    size_t pinned_ = 0;
    std::list<Key> lru_;
    Map map_;
};

}  // namespace starling::ggml
