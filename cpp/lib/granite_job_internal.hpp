#pragma once

#include "starling_ggml.h"

#include <cstdint>
#include <string>

// Internal serving seam. A job borrows the PCM buffer and model: the caller
// must keep both alive until free_granite_job. A job's create, last-chunk,
// step and free calls belong to one request thread; never access the same job
// concurrently. Each step executes one chunk under the C API runtime lock.
namespace starling::ggml::lib {
struct GraniteChunkJob;

GraniteChunkJob* create_granite_job(starling_ggml_ctx* ctx,
                                    const float* pcm, int64_t n);
// -1 = error (starling_ggml_last_error), 0 = more chunks, 1 = final text.
int step_granite_job(starling_ggml_ctx* ctx, GraniteChunkJob* job,
                     std::string* final_text);
bool granite_job_last_chunk(const GraniteChunkJob* job);
void free_granite_job(GraniteChunkJob* job);

// Engine implementations used only by the C API shell above. The shell
// supplies the runtime lock and translates errors into its context storage.
GraniteChunkJob* granite_job_create_impl(void* model, const float* pcm,
                                          int64_t n, const char** err);
int granite_job_step_impl(GraniteChunkJob* job, std::string* final_text,
                          const char** err);
bool granite_job_last_chunk_impl(const GraniteChunkJob* job);
void granite_job_free_impl(GraniteChunkJob* job);
}
