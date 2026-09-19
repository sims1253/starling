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

}  // namespace starling::ggml
