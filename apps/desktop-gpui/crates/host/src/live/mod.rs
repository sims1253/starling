//! Live transcription of a take while it records (#357, moved into the
//! host by #220): the `/stream` client ([`stream`]), the worker that pumps
//! a take's journaled audio into it and brings previews back ([`pump`]),
//! and the stream timeline trace ([`trace`], #226).

pub mod pump;
pub mod stream;
pub mod trace;
