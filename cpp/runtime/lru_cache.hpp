// Bounded replay caches keep entry addresses stable because captured graphs
// retain pointers into their input pools. Callers serialize cache access.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <unordered_map>
#include <utility>

namespace starling::ggml {

// Maximum retained shapes per model cache; configurable below.
constexpr size_t kDefaultReplayCacheSize = 16;

// Process-global cache capacity, read from STARLING_REPLAY_CACHE_SIZE (>=1) on
// each cache's first construction. Reading at first use (rather than once at
// process start) lets a test or harness set the env before loading a model.
inline size_t replay_cache_size() {
    if (const char* env = std::getenv("STARLING_REPLAY_CACHE_SIZE")) {
        long v = std::atol(env);
        if (v >= 1) return (size_t)v;
    }
    return kDefaultReplayCacheSize;
}

// A bounded LRU map. `Value` is typically an entry struct holding a
// GraphInputPool + unique_ptr<ReplayGraph>; it must be default-constructible
// (get_or_init places a default value first, then fills it in place).
template <typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class LruCache {
public:
    explicit LruCache(size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

    size_t size() const { return map_.size(); }

    // On hit: mark MRU and return a pointer to the value (stable until the next
    // non-const operation that evicts THIS key). On miss: return nullptr (the
    // caller may then call get_or_init).
    Value* get(const Key& key) {
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        touch(it);
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
            return &it->second.second;
        }
        trim();
        lru_.push_front(key);
        auto inserted = map_.end();
        try {
            inserted = map_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(key),
                std::forward_as_tuple(lru_.begin(), Value())).first;
            init(inserted->second.second);
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
    // <= capacity. No-op below capacity.
    void trim() {
        while (map_.size() >= capacity_ && !lru_.empty()) {
            Key victim = lru_.back();
            lru_.pop_back();
            map_.erase(victim);  // destroys the value (and its ReplayGraph)
        }
    }

    size_t capacity_;
    std::list<Key> lru_;
    Map map_;
};

}  // namespace starling::ggml
