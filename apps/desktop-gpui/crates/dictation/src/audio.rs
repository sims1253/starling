//! Port of `packages/dictation/src/audio.ts` — dependency-free, deterministic
//! PCM mixing, resampling, and canonical PCM16 WAV encode/decode. See
//! `apps/desktop-gpui/PORT.md` ("Audio"). Error messages are kept verbatim
//! from the TypeScript source.

/// Server-side capture format every upload is normalized to.
pub const STARLING_SAMPLE_RATE: u32 = 16_000;

/// Interleaved floating-point PCM in the range -1..=1.
#[derive(Clone, Debug)]
pub struct PcmAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Canonical upload WAV plus the durations the UI displays.
#[derive(Clone, Debug)]
pub struct PreparedWav {
    pub wav: Vec<u8>,
    pub duration_ms: f64,
    pub duration_seconds: f64,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct AudioFormatError {
    pub message: String,
}

fn error(message: &str) -> AudioFormatError {
    AudioFormatError {
        message: message.to_string(),
    }
}

/// Mirrors `assertPcm` in audio.ts (minus the non-integer checks that the
/// `u32`/`u16` types make unrepresentable). Returns the effective channel
/// count. Validation order matches the TS source.
fn assert_pcm(audio: &PcmAudio) -> Result<usize, AudioFormatError> {
    if audio.sample_rate == 0 {
        return Err(error("sampleRate must be a positive integer"));
    }

    let channels = audio.channels as usize;

    if channels < 1 || channels > 64 {
        return Err(error("channels must be an integer between 1 and 64"));
    }

    if audio.samples.is_empty() {
        return Err(error("audio contains no samples"));
    }

    if audio.samples.len() % channels != 0 {
        return Err(error(
            "interleaved sample count is not divisible by channels",
        ));
    }

    Ok(channels)
}

/// Mix interleaved PCM to mono without modifying the source.
pub fn mix_to_mono(audio: &PcmAudio) -> Result<Vec<f32>, AudioFormatError> {
    let channels = assert_pcm(audio)?;

    if channels == 1 {
        return Ok(audio.samples.clone());
    }

    let frames = audio.samples.len() / channels;
    let mut mono = Vec::with_capacity(frames);

    for frame in 0..frames {
        let base = frame * channels;
        // Summed in f64 like JS, then rounded once on the f32 store.
        let mut sum = 0.0f64;
        for channel in 0..channels {
            sum += audio.samples[base + channel] as f64;
        }
        mono.push((sum / channels as f64) as f32);
    }

    Ok(mono)
}

/// Resample interleaved floating-point PCM to mono 16 kHz.
///
/// Linear interpolation is intentionally dependency-free and deterministic.
/// Native recording APIs should still request 16 kHz directly when possible.
pub fn resample_to_16k(audio: &PcmAudio) -> Result<Vec<f32>, AudioFormatError> {
    let mono = mix_to_mono(audio)?;

    if audio.sample_rate == STARLING_SAMPLE_RATE {
        return Ok(mono);
    }

    let output_length = ((mono.len() as f64 * f64::from(STARLING_SAMPLE_RATE))
        / f64::from(audio.sample_rate))
    .round()
    .max(1.0) as usize;

    let mut output = Vec::with_capacity(output_length);
    let ratio = f64::from(audio.sample_rate) / f64::from(STARLING_SAMPLE_RATE);
    let last = mono.len() - 1;

    for index in 0..output_length {
        let position = index as f64 * ratio;
        let left = (position.floor() as usize).min(last);
        let right = (left + 1).min(last);
        let fraction = position - left as f64;
        let left_sample = f64::from(mono[left]);
        output.push((left_sample + (f64::from(mono[right]) - left_sample) * fraction) as f32);
    }

    Ok(output)
}

/// Mirrors `pcm16` in audio.ts: non-finite becomes 0, clamps to -1..=1, then
/// rounds half away from zero with the asymmetric scales JS applies
/// (`0x8000` for negatives, `0x7fff` for non-negatives). The multiply happens
/// in f64 because JS numbers are f64 even for Float32Array elements.
fn pcm16(sample: f32) -> i16 {
    let finite = if sample.is_finite() { sample } else { 0.0 };
    let clamped = finite.max(-1.0).min(1.0);

    if clamped < 0.0 {
        (f64::from(clamped) * f64::from(0x8000)).round() as i16
    } else {
        (f64::from(clamped) * f64::from(0x7fff)).round() as i16
    }
}

/// Encode arbitrary floating-point PCM as mono 16 kHz PCM16 WAV.
pub fn encode_wav_16k(audio: &PcmAudio) -> Result<Vec<u8>, AudioFormatError> {
    let samples = resample_to_16k(audio)?;
    let data_size = samples.len() as u64 * 2;

    if data_size > u64::from(0xffff_ffffu32 - 36) {
        return Err(error("audio is too large for a WAV file"));
    }

    let mut bytes = Vec::with_capacity(44 + data_size as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36u32 + data_size as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&STARLING_SAMPLE_RATE.to_le_bytes());
    bytes.extend_from_slice(&(STARLING_SAMPLE_RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_size as u32).to_le_bytes());

    for &sample in &samples {
        bytes.extend_from_slice(&pcm16(sample).to_le_bytes());
    }

    Ok(bytes)
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn ascii_at(bytes: &[u8], offset: usize, length: usize) -> String {
    bytes[offset..offset + length]
        .iter()
        .map(|&byte| byte as char)
        .collect()
}

/// Decode the PCM16 WAV subset accepted by both serving backends.
pub fn decode_pcm16_wav(input: &[u8]) -> Result<PcmAudio, AudioFormatError> {
    let bytes = input;

    if bytes.len() < 44 || ascii_at(bytes, 0, 4) != "RIFF" || ascii_at(bytes, 8, 4) != "WAVE" {
        return Err(error("expected a RIFF/WAVE file"));
    }

    let mut offset = 12usize;
    // (channels, sample_rate, bits_per_sample) from the fmt chunk.
    let mut format: Option<(u16, u32, u16)> = None;
    let mut data_offset: Option<usize> = None;
    let mut data_size = 0usize;

    while offset + 8 <= bytes.len() {
        let id = ascii_at(bytes, offset, 4);
        let size = read_u32(bytes, offset + 4) as usize;
        let body = offset + 8;

        if size > bytes.len() - body {
            return Err(error("WAV chunk exceeds the available payload"));
        }

        if id == "fmt " {
            if size < 16 {
                return Err(error("WAV fmt chunk is truncated"));
            }

            let encoding = read_u16(bytes, body);

            if encoding != 1 {
                return Err(error("only PCM WAV audio is supported"));
            }

            format = Some((
                read_u16(bytes, body + 2),
                read_u32(bytes, body + 4),
                read_u16(bytes, body + 14),
            ));
        } else if id == "data" {
            data_offset = Some(body);
            data_size = size;
            break;
        }

        offset = body + size + (size % 2);
    }

    let (channels, sample_rate, bits_per_sample) = if let Some(format) = format {
        format
    } else {
        return Err(error("WAV is missing fmt or data"));
    };

    if data_offset.is_none() {
        return Err(error("WAV is missing fmt or data"));
    }

    if bits_per_sample != 16 {
        return Err(error("only 16-bit WAV audio is supported"));
    }

    if channels < 1 || channels > 64 || sample_rate < 1 {
        return Err(error("WAV has invalid channel or sample-rate metadata"));
    }

    let bytes_per_frame = channels as usize * 2;

    if data_size == 0 || data_size % bytes_per_frame != 0 {
        return Err(error("WAV PCM data is empty or has an incomplete frame"));
    }

    let start = data_offset.unwrap();
    let samples = bytes[start..start + data_size]
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32_768.0)
        .collect();

    Ok(PcmAudio {
        samples,
        sample_rate,
        channels,
    })
}

/// Validate/decode WAV bytes and return the canonical upload format.
///
/// The TS function also accepted `PcmAudio` sources directly; in Rust those
/// callers use [`encode_wav_16k`] instead.
pub fn prepare_wav_16k(bytes: &[u8]) -> Result<PreparedWav, AudioFormatError> {
    let pcm = decode_pcm16_wav(bytes)?;
    let wav = encode_wav_16k(&pcm)?;
    let frame_count = (wav.len() - 44) / 2;
    let duration_seconds = frame_count as f64 / f64::from(STARLING_SAMPLE_RATE);

    Ok(PreparedWav {
        wav,
        duration_ms: duration_seconds * 1_000.0,
        duration_seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message_of<T: std::fmt::Debug>(result: Result<T, AudioFormatError>) -> String {
        result
            .expect_err("expected an AudioFormatError")
            .to_string()
    }

    fn mono(samples: &[f32], sample_rate: u32) -> PcmAudio {
        PcmAudio {
            samples: samples.to_vec(),
            sample_rate,
            channels: 1,
        }
    }

    /// Builds a RIFF chunk with a size field and word padding for odd bodies.
    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + body.len() + body.len() % 2);
        out.extend_from_slice(id);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        if body.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    fn wav_from_chunks(chunks: &[Vec<u8>]) -> Vec<u8> {
        let riff_size: usize = 4 + chunks.iter().map(|chunk| chunk.len()).sum::<usize>();
        let mut out = Vec::with_capacity(8 + riff_size);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(riff_size as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        for chunk in chunks {
            out.extend_from_slice(chunk);
        }
        out
    }

    fn fmt_body(channels: u16, sample_rate: u32, bits_per_sample: u16, encoding: u16) -> Vec<u8> {
        let mut body = Vec::with_capacity(16);
        body.extend_from_slice(&encoding.to_le_bytes());
        body.extend_from_slice(&channels.to_le_bytes());
        body.extend_from_slice(&sample_rate.to_le_bytes());
        body.extend_from_slice(
            &(sample_rate * u32::from(channels) * u32::from(bits_per_sample / 8)).to_le_bytes(),
        );
        body.extend_from_slice(&(channels * bits_per_sample / 8).to_le_bytes());
        body.extend_from_slice(&bits_per_sample.to_le_bytes());
        body
    }

    #[test]
    fn prepares_stereo_48k_to_canonical_mono_16k() {
        let frames = 4_800usize;
        let mut samples = Vec::with_capacity(frames * 2);
        for index in 0..frames {
            let t = index as f32 / 10.0;
            samples.push(t.sin());
            samples.push(t.cos());
        }

        let stereo = PcmAudio {
            samples,
            sample_rate: 48_000,
            channels: 2,
        };

        let wav = encode_wav_16k(&stereo).expect("encode succeeds");
        let prepared = prepare_wav_16k(&wav).expect("prepare succeeds");
        let decoded = decode_pcm16_wav(&prepared.wav).expect("decode succeeds");

        assert_eq!(decoded.sample_rate, STARLING_SAMPLE_RATE);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples.len(), 1_600);
        assert_eq!(prepared.duration_seconds, 0.1);
        assert_eq!(prepared.duration_ms, 100.0);
        // Canonical upload shape: RIFF/WAVE with a 44-byte header.
        assert_eq!(&prepared.wav[0..4], b"RIFF");
        assert_eq!(&prepared.wav[8..12], b"WAVE");
    }

    #[test]
    fn clamps_non_finite_and_out_of_range_samples() {
        let audio = mono(&[-2.0, -1.0, f32::NAN, 1.0, 2.0], 16_000);
        let bytes = encode_wav_16k(&audio).expect("encode succeeds");
        let decoded = decode_pcm16_wav(&bytes).expect("decode succeeds");

        let rounded: Vec<i32> = decoded
            .samples
            .iter()
            .map(|sample| (sample * 32_768.0).round() as i32)
            .collect();

        assert_eq!(rounded, vec![-32_768, -32_768, 0, 32_767, 32_767]);
    }

    #[test]
    fn encodes_extremes_with_the_asymmetric_clamp() {
        let audio = mono(&[1.0, -1.0], 16_000);
        let bytes = encode_wav_16k(&audio).expect("encode succeeds");

        // +1.0 scales by 0x7fff, -1.0 scales by 0x8000.
        assert_eq!(&bytes[44..46], &0x7fffu16.to_le_bytes());
        assert_eq!(&bytes[46..48], &0x8000u16.to_le_bytes());
    }

    #[test]
    fn rejects_truncated_wav_chunks_instead_of_treating_them_as_pcm() {
        let audio = mono(&[0.0, 0.0], 16_000);
        let mut bytes = encode_wav_16k(&audio).expect("encode succeeds");
        bytes[40..44].copy_from_slice(&2_000u32.to_le_bytes());

        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "WAV chunk exceeds the available payload"
        );
    }

    #[test]
    fn writes_the_canonical_44_byte_header() {
        let audio = mono(&[0.0], 16_000);
        let bytes = encode_wav_16k(&audio).expect("encode succeeds");

        assert_eq!(bytes.len(), 46);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 38);
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(bytes[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(bytes[22..24].try_into().unwrap()), 1);
        assert_eq!(
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            16_000
        );
        assert_eq!(
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            32_000
        );
        assert_eq!(u16::from_le_bytes(bytes[32..34].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(bytes[34..36].try_into().unwrap()), 16);
        assert_eq!(&bytes[36..40], b"data");
        assert_eq!(u32::from_le_bytes(bytes[40..44].try_into().unwrap()), 2);
        assert_eq!(&bytes[44..46], &[0, 0]);
    }

    #[test]
    fn resample_matches_lengths_and_interpolates_linearly() {
        // Downsampling 8 kHz → 16 kHz: 3 frames become 6, interpolated.
        let audio = mono(&[0.0, 1.0, 0.0], 8_000);
        assert_eq!(
            resample_to_16k(&audio).expect("resample succeeds"),
            vec![0.0, 0.5, 1.0, 0.5, 0.0, 0.0]
        );

        // The output length never collapses to zero: round(1 * 16000/96000)
        // is 0, clamped to 1.
        let single = mono(&[0.5], 96_000);
        assert_eq!(
            resample_to_16k(&single).expect("resample succeeds").len(),
            1
        );

        // Already-16 kHz audio is returned untouched.
        let native = mono(&[0.25, -0.75], 16_000);
        assert_eq!(
            resample_to_16k(&native).expect("resample succeeds"),
            vec![0.25, -0.75]
        );
    }

    #[test]
    fn mixes_interleaved_stereo_to_mono() {
        let stereo = PcmAudio {
            samples: vec![1.0, -1.0, 0.5, 0.5],
            sample_rate: 16_000,
            channels: 2,
        };
        assert_eq!(mix_to_mono(&stereo).expect("mix succeeds"), vec![0.0, 0.5]);

        let mono_audio = mono(&[0.25, -0.25], 16_000);
        assert_eq!(
            mix_to_mono(&mono_audio).expect("mix succeeds"),
            vec![0.25, -0.25]
        );
    }

    #[test]
    fn validates_pcm_shape_like_assert_pcm() {
        let zero_rate = mono(&[0.0], 0);
        assert_eq!(
            message_of(mix_to_mono(&zero_rate)),
            "sampleRate must be a positive integer"
        );

        for channels in [0u16, 65, 200] {
            let audio = PcmAudio {
                samples: vec![0.0; 4],
                sample_rate: 16_000,
                channels,
            };
            assert_eq!(
                message_of(mix_to_mono(&audio)),
                "channels must be an integer between 1 and 64"
            );
        }

        let empty = mono(&[], 16_000);
        assert_eq!(message_of(mix_to_mono(&empty)), "audio contains no samples");

        let ragged = PcmAudio {
            samples: vec![0.0, 0.0, 0.0],
            sample_rate: 16_000,
            channels: 2,
        };
        assert_eq!(
            message_of(mix_to_mono(&ragged)),
            "interleaved sample count is not divisible by channels"
        );
    }

    #[test]
    fn round_trips_encode_decode_at_an_arbitrary_rate() {
        let frames = 1_600usize;
        let samples: Vec<f32> = (0..frames).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = mono(&samples, 44_100);

        let bytes = encode_wav_16k(&audio).expect("encode succeeds");
        let expected_len = (frames as f64 * 16_000.0 / 44_100.0).round() as usize;
        assert_eq!(expected_len, 580);
        assert_eq!(
            u32::from_le_bytes(bytes[40..44].try_into().unwrap()),
            580 * 2
        );

        let decoded = decode_pcm16_wav(&bytes).expect("decode succeeds");
        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples.len(), expected_len);

        // Decoded audio stays within 1 LSB of the original mono mix.
        let original = resample_to_16k(&audio).expect("resample succeeds");
        for (decoded_sample, original_sample) in decoded.samples.iter().zip(&original) {
            assert!((decoded_sample - original_sample).abs() <= 1.5 / 32_768.0);
        }
    }

    #[test]
    fn prepare_reports_durations_and_canonical_bytes() {
        let audio = mono(&vec![0.0; 16_000], 16_000);
        let bytes = encode_wav_16k(&audio).expect("encode succeeds");

        let prepared = prepare_wav_16k(&bytes).expect("prepare succeeds");
        assert_eq!(prepared.wav, bytes);
        assert_eq!(prepared.duration_seconds, 1.0);
        assert_eq!(prepared.duration_ms, 1_000.0);
    }

    #[test]
    fn rejects_non_riff_and_short_inputs() {
        assert_eq!(
            message_of(decode_pcm16_wav(b"this is not a wav file at all")),
            "expected a RIFF/WAVE file"
        );
        assert_eq!(
            message_of(decode_pcm16_wav(&[0u8; 43])),
            "expected a RIFF/WAVE file"
        );
    }

    #[test]
    fn rejects_non_pcm_or_non_16bit_formats() {
        let fmt_float = chunk(b"fmt ", &fmt_body(1, 16_000, 16, 3));
        let data = chunk(b"data", &[0, 0, 0, 0]);
        let bytes = wav_from_chunks(&[fmt_float, data.clone()]);
        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "only PCM WAV audio is supported"
        );

        let fmt_8bit = chunk(b"fmt ", &fmt_body(1, 16_000, 8, 1));
        let bytes = wav_from_chunks(&[fmt_8bit, data.clone()]);
        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "only 16-bit WAV audio is supported"
        );

        let fmt_short = chunk(b"fmt ", &fmt_body(1, 16_000, 16, 1)[..15]);
        let bytes = wav_from_chunks(&[fmt_short, data]);
        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "WAV fmt chunk is truncated"
        );
    }

    #[test]
    fn rejects_invalid_channel_or_sample_rate_metadata() {
        let data = chunk(b"data", &[0, 0, 0, 0]);

        for channels in [0u16, 65] {
            let fmt = chunk(b"fmt ", &fmt_body(channels, 16_000, 16, 1));
            let bytes = wav_from_chunks(&[fmt, data.clone()]);
            assert_eq!(
                message_of(decode_pcm16_wav(&bytes)),
                "WAV has invalid channel or sample-rate metadata"
            );
        }

        let fmt = chunk(b"fmt ", &fmt_body(1, 0, 16, 1));
        let bytes = wav_from_chunks(&[fmt, data]);
        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "WAV has invalid channel or sample-rate metadata"
        );
    }

    #[test]
    fn rejects_empty_or_ragged_pcm_data() {
        let fmt = chunk(b"fmt ", &fmt_body(1, 16_000, 16, 1));

        let bytes = wav_from_chunks(&[fmt.clone(), chunk(b"data", &[])]);
        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "WAV PCM data is empty or has an incomplete frame"
        );

        // One stray byte: not a whole 16-bit mono frame.
        let bytes = wav_from_chunks(&[fmt, chunk(b"data", &[0x7f])]);
        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "WAV PCM data is empty or has an incomplete frame"
        );
    }

    #[test]
    fn rejects_wavs_missing_fmt_or_data() {
        // Fixtures must clear the 44-byte minimum or the "expected a
        // RIFF/WAVE file" check wins, exactly like the TS source.
        let fmt = chunk(b"fmt ", &fmt_body(1, 16_000, 16, 1));
        let data = chunk(b"data", &[0u8; 32]);
        let filler = chunk(b"LIST", &[0u8; 16]);

        let bytes = wav_from_chunks(&[data]);
        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "WAV is missing fmt or data"
        );

        let bytes = wav_from_chunks(&[fmt, filler]);
        assert_eq!(
            message_of(decode_pcm16_wav(&bytes)),
            "WAV is missing fmt or data"
        );
    }

    #[test]
    fn accepts_extra_chunks_and_word_padding_before_data() {
        let fmt = chunk(b"fmt ", &fmt_body(1, 16_000, 16, 1));

        // Odd-sized chunk before `data`: the walker must skip its pad byte,
        // otherwise the data chunk would be read at a misaligned offset.
        let metadata = chunk(b"LIST", b"abc");
        assert_eq!(metadata.len() % 2, 0); // 8-byte header + 3 body + 1 pad

        let mut data_body = Vec::new();
        data_body.extend_from_slice(&(-32_768i16).to_le_bytes());
        data_body.extend_from_slice(&32_767i16.to_le_bytes());
        let data = chunk(b"data", &data_body);

        let bytes = wav_from_chunks(&[fmt, metadata, data]);
        let decoded = decode_pcm16_wav(&bytes).expect("decode succeeds");

        assert_eq!(decoded.sample_rate, 16_000);
        assert_eq!(decoded.channels, 1);
        assert_eq!(decoded.samples, vec![-1.0, 32_767.0 / 32_768.0]);
    }
}
