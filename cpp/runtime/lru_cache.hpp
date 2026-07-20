// lru_cache.hpp — a generic bounded LRU container for per-shape capture caches.
//
// Starling's three per-shape ReplayGraph caches (parakeet Encoder::replay_cache_,
// moss g_encoder_cache, moss g_prefill_cache) key a captured ReplayGraph on an
// audio length / mel shape and keep it alive for reuse. Real audio produces a
// near-continuous length distribution, so WITHOUT eviction every distinct shape
// permanently pins its own device buffer (the ReplayGraph's private gallocr +
// captured CUDA graph) until process exit -> unbounded VRAM growth -> OOM. That
// is the Wave H bug. This helper gives those caches a genuine LRU bound.
//
// Mechanics: a doubly-linked list of keys in MRU->LRU order, plus an
// unordered_map from key -> {list iterator, value}. touch/get/get_or_init splice
// the accessed node to the front (MRU); a miss at capacity evicts the back
// (LRU) first. All operations are O(1).
//
// Node stability is load-bearing: the value is stored BY VALUE inside the map
// node, and std::unordered_map guarantees element addresses are stable across
// other insert/erase. The three caches' ReplayGraph build lambdas capture
// pointers into the value's GraphInputPool (chunks/valid buffers); those host
// pointers are registered on the ReplayGraph and re-uploaded every replay, so
// the value (hence its pool) must not relocate for the ReplayGraph's life. Map
// node stability guarantees exactly that, until the value is evicted (which
// destroys its ReplayGraph in the same step). get_or_init therefore places the
// value in the map FIRST, then hands the caller a stable reference to fill +
// build against.
//
// Eviction safety mid-run: a value is only evicted by trim(), called inside a
// miss path BEFORE the caller uses the returned entry. The just-inserted entry
// is MRU, so it is never the trim() victim while the caller holds it; nothing
// re-enters the cache during a ReplayGraph::compute (the call pattern is fully
// synchronous: get -> set_input -> compute). Evicting a DIFFERENT (older) entry
// is harmless to the in-use one: each ReplayGraph owns its own private gallocr
// (see backend.hpp), so freeing one never touches another's device pointers.
//
// Synchronization: NONE. The cache is NOT internally locked. Starling's
// inference is process-serial (one Backend; callers serialize), so the existing
// cache sites are unsynchronized and this helper preserves that. Add a mutex at
// the call site if concurrent callers are ever introduced.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <unordered_map>
#include <utility>

namespace starling::ggml {

// Default per-cache capacity. Overridable per-process via the env var below.
// Chosen to keep realistic within-run reuse warm (the 3 synthetic tiers plus a
// handful of repeated similar-length utterances) while capping worst-case VRAM
// at ~16 private gallocrs per cache. See review.md §5 for the bound rationale.
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

    size_t capacity() const { return capacity_; }
    void set_capacity(size_t c) {
        capacity_ = (c == 0) ? 1 : c;
        trim();
    }
    size_t size() const { return map_.size(); }
    bool empty() const { return map_.empty(); }

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
        auto ins = map_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(key),
            std::forward_as_tuple(lru_.begin(), Value()));
        init(ins.first->second.second);
        return &ins.first->second.second;
    }

    // Insert/overwrite `key` -> `value`, marking MRU. Evicts LRU first if at
    // capacity (on a fresh key). Returns a reference to the stored value.
    template <typename V>
    Value& put(const Key& key, V&& value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            it->second.second = std::forward<V>(value);
            touch(it);
            return it->second.second;
        }
        trim();
        lru_.push_front(key);
        auto ins = map_.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(key),
            std::forward_as_tuple(lru_.begin(), std::forward<V>(value)));
        return ins.first->second.second;
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
