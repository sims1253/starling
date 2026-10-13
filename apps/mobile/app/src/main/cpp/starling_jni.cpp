// JNI bridge over the flat Starling C API (cpp/include/starling_ggml.h).
// The engine serializes calls internally, so this layer is deliberately
// thin: convert types, translate errors to null, and never leak native
// memory across the boundary (transcripts are copied into jstrings and
// the malloc'd buffer is released here).
#include <jni.h>

#include <string>

#include "runtime/call_abort.hpp"
#include "starling_ggml.h"

namespace {

// The speech-model kinds the app loads (ModelKind.kt). The S1 text model
// and the size aliases (ARK06, QWEN3_06: same engines, dims from the GGUF)
// are deliberately not among them.
bool isSpeechModel(jint kind) {
  switch (kind) {
    case STARLING_GGML_PARAKEET_TDT:
    case STARLING_GGML_MOSS:
    case STARLING_GGML_ARK:
    case STARLING_GGML_GRANITE:
    case STARLING_GGML_QWEN3:
      return true;
    default:
      return false;
  }
}

// Why the last load on this thread was refused before reaching the engine;
// empty otherwise. Thread-local like the engine's own load error
// (starling_ggml_last_error(NULL)), which lastError(0) reports otherwise.
thread_local std::string unsupportedKindError;

// Transcripts are malloc'd engine strings that [owned] marks for release;
// error strings are borrowed engine storage and must not be freed.
jstring toJString(JNIEnv* env, const char* value, bool owned) {
  if (value == nullptr) {
    return nullptr;
  }
  jstring result = env->NewStringUTF(value);
  if (owned) {
    starling_ggml_free_string(const_cast<char*>(value));
  }
  return result;
}

// The Java cancel predicate of one transcribeCancellable call, polled by the
// engine at its checkpoints (call_abort.hpp) on the calling thread, so the
// call's JNIEnv is valid. A predicate that throws counts as "keep going".
struct JavaCancel {
  JNIEnv* env;
  jobject cancel;
  jmethodID requested;
};

bool javaCancelRequested(void* user) {
  auto* c = static_cast<JavaCancel*>(user);
  const jboolean requested = c->env->CallBooleanMethod(c->cancel, c->requested);
  if (c->env->ExceptionCheck()) {
    c->env->ExceptionClear();
    return false;
  }
  return requested == JNI_TRUE;
}

jstring transcribe(JNIEnv* env, jlong handle, jfloatArray samples,
                   jint sample_rate) {
  if (handle == 0) {
    return nullptr;
  }
  const jsize length = env->GetArrayLength(samples);
  jfloat* body = env->GetFloatArrayElements(samples, nullptr);
  if (body == nullptr) {
    return nullptr;
  }
  char* text = starling_ggml_transcribe_pcm(
      reinterpret_cast<starling_ggml_ctx*>(handle), body, length,
      sample_rate);
  // The engine never mutates the buffer; skip the copy-back.
  env->ReleaseFloatArrayElements(samples, body, JNI_ABORT);
  return toJString(env, text, /* owned = */ true);
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_dev_starling_mobile_engine_StarlingNative_abiVersion(JNIEnv*, jclass) {
  return starling_ggml_abi_version();
}

extern "C" JNIEXPORT jlong JNICALL
Java_dev_starling_mobile_engine_StarlingNative_load(JNIEnv* env, jclass,
                                                    jstring gguf_path,
                                                    jint kind) {
  if (!isSpeechModel(kind)) {
    // Reported like an engine load error: load() returns 0 and
    // lastError(0) carries the reason.
    unsupportedKindError = "unsupported model kind " + std::to_string(kind);
    return 0;
  }
  unsupportedKindError.clear();
  const char* path = env->GetStringUTFChars(gguf_path, nullptr);
  if (path == nullptr) {
    unsupportedKindError = "out of memory reading the model path";
    return 0;
  }
  starling_ggml_ctx* ctx =
      starling_ggml_load(static_cast<starling_ggml_model>(kind), path);
  env->ReleaseStringUTFChars(gguf_path, path);
  return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_starling_mobile_engine_StarlingNative_transcribe(
    JNIEnv* env, jclass, jlong handle, jfloatArray samples, jint sample_rate) {
  return transcribe(env, handle, samples, sample_rate);
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_starling_mobile_engine_StarlingNative_transcribeCancellable(
    JNIEnv* env, jclass, jlong handle, jfloatArray samples, jint sample_rate,
    jobject cancel) {
  jclass type = env->GetObjectClass(cancel);
  jmethodID requested = env->GetMethodID(type, "requested", "()Z");
  env->DeleteLocalRef(type);
  if (requested == nullptr) {
    // Like a predicate that throws: clear the NoSuchMethodError and run
    // without checkpoints, so the call fails or succeeds on its own terms
    // (the session discards a preview that completes after it was dropped).
    env->ExceptionClear();
    return transcribe(env, handle, samples, sample_rate);
  }
  JavaCancel state{env, cancel, requested};
  starling::ggml::CallAbortScope scope(javaCancelRequested, &state);
  return transcribe(env, handle, samples, sample_rate);
}

extern "C" JNIEXPORT void JNICALL
Java_dev_starling_mobile_engine_StarlingNative_free(JNIEnv*, jclass,
                                                    jlong handle) {
  if (handle != 0) {
    starling_ggml_free(reinterpret_cast<starling_ggml_ctx*>(handle));
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_starling_mobile_engine_StarlingNative_lastError(JNIEnv* env, jclass,
                                                         jlong handle) {
  if (handle == 0 && !unsupportedKindError.empty()) {
    return toJString(env, unsupportedKindError.c_str(), /* owned = */ false);
  }
  return toJString(
      env,
      starling_ggml_last_error(
          reinterpret_cast<starling_ggml_ctx*>(handle)),
      /* owned = */ false);
}
