// trace_test_support.hpp — shared helpers for the stderr-capture tests
// (granite_stage_test.cpp and the STARLING_TRACE tests): capture_stderr and
// SETENV/UNSETENV portability macros.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#define SETENV(k, v) _putenv_s(k, v)
#define UNSETENV(k) _putenv_s(k, "")
#else
#include <fcntl.h>
#include <unistd.h>
#define SETENV(k, v) setenv(k, v, 1)
#define UNSETENV(k) unsetenv(k)
#endif

// --------------------------------------------------------------------------- //
// stderr capture + GRANITE_STAGE line parsing.
// --------------------------------------------------------------------------- //
#ifdef _WIN32
inline std::string capture_stderr(const std::function<void()>& fn) {
    fn();  // no capture on Windows; the e2e layer is skipped by the caller
    return "";
}
#else
inline std::string capture_stderr(const std::function<void()>& fn) {
    const char* tmpl = "/tmp/granite_stage_stderr_XXXXXX";
    std::vector<char> name(tmpl, tmpl + std::strlen(tmpl) + 1);
    const int fd = mkstemp(name.data());
    if (fd < 0) return "";
    std::fflush(stderr);
    const int saved = dup(2);
    if (saved < 0) { close(fd); unlink(name.data()); return ""; }
    dup2(fd, 2);
    close(fd);
    fn();
    std::fflush(stderr);
    dup2(saved, 2);
    close(saved);
    std::string out;
    if (const int rfd = open(name.data(), O_RDONLY); rfd >= 0) {
        char buf[4096];
        ssize_t k;
        while ((k = read(rfd, buf, sizeof buf)) > 0) out.append(buf, (size_t) k);
        close(rfd);
    }
    unlink(name.data());
    return out;
}
#endif

