# Domain glossary

## Take

One microphone capture, its optional live transcription session, and the
settlement of its saved audio and transcription. `TakeRecorder` owns this
lifetime. Stopping or canceling a take does not discard its ordinary audio.
A take from a private field is ephemeral and is deleted when it settles.

## Recording

The stored audio and metadata identified by a recording ID. Transcription
retries add results to the same recording. A recording survives its activity,
keyboard, or recognition client.

## Staged draft

Editable text awaiting an explicit delivery action. A keyboard draft can
combine several takes, so its lifetime extends beyond any one capture.
