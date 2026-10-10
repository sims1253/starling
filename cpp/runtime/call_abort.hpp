// call_abort.hpp — cooperative cancel of one engine call (issue #357).
//
// A serving layer that no longer needs the result of the call it is making
// (a live preview overtaken by a commit) installs a CallAbortScope around
// the synchronous engine call. Engines poll call_abort_requested() at their
// natural checkpoints (between pipeline stages, encoder graph slices and
// decoder steps) and return a "cancelled" error when it fires, leaving no
// partial state behind. Engines without checkpoints simply finish the call;
// the caller decides what a result that completes after the abort fired is
// worth.
//
// The hook is thread-local: the C engine call is synchronous, so the
// serving thread's scope is the engine's scope (the same convention as
// trace::RequestScope). No scope installed means never abort.
#pragma once

namespace starling::ggml {

using CallAbortFn = bool (*)(void* user);

struct CallAbortHook {
    CallAbortFn fn = nullptr;
    void* user = nullptr;
    bool fired = false;  // the predicate returned true at least once
};

inline thread_local CallAbortHook* t_call_abort = nullptr;

// True when the caller asked the engine call in progress on this thread to
// stop. Cheap enough to poll once per decoder step.
inline bool call_abort_requested() {
    CallAbortHook* h = t_call_abort;
    if (!h || !h->fn) return false;
    if (h->fired) return true;
    h->fired = h->fn(h->user);
    return h->fired;
}

// Error text engines report for an aborted call.
inline constexpr const char* kCallAbortedError = "cancelled";

// RAII: install `fn(user)` as this thread's abort predicate; restores the
// previous scope on exit.
class CallAbortScope {
public:
    CallAbortScope(CallAbortFn fn, void* user) : prev_(t_call_abort) {
        hook_.fn = fn;
        hook_.user = user;
        t_call_abort = &hook_;
    }
    ~CallAbortScope() { t_call_abort = prev_; }
    CallAbortScope(const CallAbortScope&) = delete;
    CallAbortScope& operator=(const CallAbortScope&) = delete;

    // Whether the predicate fired during the scope.
    bool fired() const { return hook_.fired; }

private:
    CallAbortHook hook_;
    CallAbortHook* prev_;
};

} // namespace starling::ggml
