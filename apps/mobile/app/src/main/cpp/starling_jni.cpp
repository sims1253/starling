// JNI bridge over the flat Starling C API (cpp/include/starling_ggml.h).
// The engine serializes calls internally, so this layer is deliberately
// thin: convert types, translate errors to null, and never leak native
// memory across the boundary (transcripts are copied into jstrings and
// the malloc'd buffer is released here).
#include <jni.h>

#include "starling_ggml.h"

namespace {

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

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_dev_starling_mobile_engine_StarlingNative_abiVersion(JNIEnv*, jclass) {
  return starling_ggml_abi_version();
}

extern "C" JNIEXPORT jlong JNICALL
Java_dev_starling_mobile_engine_StarlingNative_load(JNIEnv* env, jclass,
                                                    jstring gguf_path) {
  const char* path = env->GetStringUTFChars(gguf_path, nullptr);
  if (path == nullptr) {
    return 0;
  }
  starling_ggml_ctx* ctx =
      starling_ggml_load(STARLING_GGML_PARAKEET_TDT, path);
  env->ReleaseStringUTFChars(gguf_path, path);
  return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_starling_mobile_engine_StarlingNative_transcribe(
    JNIEnv* env, jclass, jlong handle, jfloatArray samples, jint sample_rate) {
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
  return toJString(
      env,
      starling_ggml_last_error(
          reinterpret_cast<starling_ggml_ctx*>(handle)),
      /* owned = */ false);
}
