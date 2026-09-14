# Shared wire contract

`openapi.json` describes the supported native HTTP adapter. It is language
independent so TypeScript, Kotlin, Swift, and third-party clients share one
protocol reference. See [API behavior and limits](../../docs/api.md).

The TypeScript client lives in `../dictation/`. Kotlin and Swift use their native
HTTP and recording APIs; they do not embed JavaScript or the deprecated Python
backend. Keep field names and accepted formats aligned with this document.

Live contract tests in `backends/native/tests/` run the real HTTP handlers with
a test engine. A successful contract test does not measure speech recognition.
