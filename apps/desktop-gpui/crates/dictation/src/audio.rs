//! Port of `packages/dictation/src/audio.ts` — dependency-free, deterministic
//! PCM mixing, resampling, and canonical PCM16 WAV encode/decode. See
//! `apps/desktop-gpui/PORT.md` ("Audio"). Error messages are kept verbatim
//! from the TypeScript source.

/// Server-side capture format every upload is normalized to.
pub const STARLING_SAMPLE_RATE: u32 = 16_000;

/// Highest input sample rate the sinc resampler accepts (R15).
///
/// Professional audio hardware tops out at 384 kHz and every real capture
/// device sits below it; past that only the kernel pays. Its half-width is
/// `ceil(4 / cutoff) ≈ 8.9 × sample_rate / 16_000`, so a mislabeled or
/// hostile import claiming a GHz-scale rate would make the resampler spend
/// pathological taps — and time — per output sample. Rates above the
/// ceiling are rejected outright, as a device would refuse them, never
/// clamped: silently resampling a mislabeled file would fabricate audio at
/// the wrong speed. With this ceiling the kernel half-width is bounded at
/// 214 taps.
pub const MAX_RESAMPLE_INPUT_RATE: u32 = 384_000;

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
/// Anti-aliased and still dependency-free (issue #122), a port of
/// `resampleTo16k` in `packages/dictation/src/audio.ts`: each output sample
/// is a Blackman-windowed sinc kernel evaluated at its exact fractional
/// input position — a windowed-sinc low-pass whose cutoff tracks the lower
/// of the two Nyquist frequencies, so content above the output band is
/// attenuated (>= 40 dB past the transition band; ~85 dB for 48/44.1 kHz
/// inputs) instead of folding into it at full amplitude the way plain
/// linear interpolation did (a 12 kHz tone at 48 kHz became a 4 kHz tone
/// at unchanged level). Deterministic output, mono mixdown, and duration
/// are preserved; edges are handled by replicating the first/last input
/// sample. Input rates above [`MAX_RESAMPLE_INPUT_RATE`] are rejected
/// rather than resampled (R15). Native recording APIs should still request
/// 16 kHz directly when possible.
///
/// Whole-recording batch contract: this runs once over the finished buffer
/// (capture drains at Stop, imports arrive whole), so there is deliberately
/// no inter-chunk filter state to carry; streaming/stateful resampling
/// belongs to a capture-pipeline redesign, not to this function.
pub fn resample_to_16k(audio: &PcmAudio) -> Result<Vec<f32>, AudioFormatError> {
    let mono = mix_to_mono(audio)?;

    if audio.sample_rate == STARLING_SAMPLE_RATE {
        return Ok(mono);
    }

    // R15: reject rates no real device produces instead of letting the
    // kernel width grow with them (~8.9 taps per 16 kHz of input rate).
    if audio.sample_rate > MAX_RESAMPLE_INPUT_RATE {
        return Err(error(&format!(
            "sampleRate must not exceed {MAX_RESAMPLE_INPUT_RATE} Hz"
        )));
    }

    let output_length = ((mono.len() as f64 * f64::from(STARLING_SAMPLE_RATE))
        / f64::from(audio.sample_rate))
    .round()
    .max(1.0) as usize;

    // Kernel design: cutoff at 90% of the lower Nyquist (7.2 kHz passband
    // edge for 16 kHz output) with a half-width of four sinc main-lobe zero
    // crossings on each side of every output position. The 3-term Blackman
    // window yields ~-74 dB stopband sidelobes and keeps aliases past the
    // transition band >= 40 dB down (>= 85 dB for 48 kHz and 44.1 kHz
    // inputs); tapCount covers upsampling too (ratio < 1 keeps the full
    // input band). With [`MAX_RESAMPLE_INPUT_RATE`] enforced above,
    // `half_taps` is bounded at 214.
    let ratio = f64::from(audio.sample_rate) / f64::from(STARLING_SAMPLE_RATE);
    let cutoff = 0.45 * f64::min(1.0, 1.0 / ratio);
    let half_taps = (4.0 / cutoff).ceil() as i64;

    let mut output = Vec::with_capacity(output_length);

    for index in 0..output_length {
        let position = index as f64 * ratio;
        let center = position.floor() as i64;
        let fraction = position - center as f64;
        output.push(sinc_sample(&mono, center, fraction, half_taps, cutoff));
    }

    Ok(output)
}

/// One windowed-sinc output sample at fractional input position
/// `center + fraction` (R16). Pure: the kernel parameters and the input
/// determine the output. Taps that fall outside the input replicate the
/// nearest edge, as the batch contract documents.
///
/// A kernel whose weight sum is not positive resolves to nearest-edge
/// replication of the input at `center` — never a fabricated `0.0`, which
/// would quietly punch silence into the take and erase whatever the real
/// neighbors say. For every cutoff the legal rate range can produce
/// (`0 < cutoff <= 0.45`) the kernel's DC gain stays above 1.1, so the
/// branch is defensive; it exists so that a degenerate kernel cannot fail
/// silently *and invisibly*.
fn sinc_sample(mono: &[f32], center: i64, fraction: f64, half_taps: i64, cutoff: f64) -> f32 {
    let last_input = mono.len() as i64 - 1;

    // Weighted sinc interpolation centered at `position` (in input
    // samples). Normalizing by the weight sum pins the DC gain to
    // exactly 1.
    let mut sum = 0.0f64;
    let mut weight_sum = 0.0f64;

    for tap in -half_taps..=half_taps {
        let offset = tap as f64 - fraction;
        let cosine = (std::f64::consts::PI * offset / half_taps as f64).cos();
        let window = 0.42 + 0.5 * cosine + 0.08 * (2.0 * cosine * cosine - 1.0);
        let angle = 2.0 * std::f64::consts::PI * cutoff * offset;
        let sinc = if angle == 0.0 { 1.0 } else { angle.sin() / angle };
        let coefficient = window * sinc;

        let sample_index = center + tap;
        let sample = f64::from(mono[sample_index.clamp(0, last_input) as usize]);

        sum += sample * coefficient;
        weight_sum += coefficient;
    }

    if weight_sum > 0.0 {
        (sum / weight_sum) as f32
    } else {
        // R16: replicate the actual input nearest the output position —
        // edge value when clamping pinned every tap to one end — instead
        // of fabricating silence.
        f64::from(mono[center.clamp(0, last_input) as usize]) as f32
    }
}

/// ECMAScript `Math.round` for finite `x`: the closest integer, with ties
/// going toward positive infinity (`Math.round(-1.5)` is `-1`). Rust's
/// `f64::round` instead rounds ties away from zero — exactly the
/// divergence G07 pinned down.
///
/// One deliberate, documented divergence from the letter of ECMAScript:
/// for `-0.5 <= x < 0` JS produces negative zero (`Math.round(-0.5)` and
/// `Math.round(-0.2)` are both `-0`), while the `floor + 1.0` form here
/// returns `+0.0`. Harmless for the only caller, [`pcm16`]: `-0.0` and
/// `+0.0` both cast to the i16 sample `0`, so the encoded WAV is
/// byte-identical either way.
fn js_round(x: f64) -> f64 {
    let floor = x.floor();
    let fraction = x - floor;
    if fraction < 0.5 {
        floor
    } else {
        floor + 1.0
    }
}

/// Mirrors `pcm16` in audio.ts: non-finite becomes 0, clamps to -1..=1, then
/// rounds with JS `Math.round` semantics — ties toward positive infinity,
/// not the half-away-from-zero of `f64::round` (G07: `-2^-16` scales to
/// exactly -0.5, which `Math.round` maps to 0, where `f64::round` gave -1)
/// — with the asymmetric scales JS applies (`0x8000` for negatives,
/// `0x7fff` for non-negatives). The multiply happens in f64 because JS
/// numbers are f64 even for Float32Array elements.
fn pcm16(sample: f32) -> i16 {
    let finite = if sample.is_finite() { sample } else { 0.0 };
    let clamped = finite.max(-1.0).min(1.0);

    if clamped < 0.0 {
        js_round(f64::from(clamped) * f64::from(0x8000)) as i16
    } else {
        js_round(f64::from(clamped) * f64::from(0x7fff)) as i16
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
    fn pcm16_matches_the_shared_rounding_contract_fixture() {
        // Shared with the TypeScript encoder (G07): packages/dictation's
        // vitest suite consumes the same file, and audio.ts is the semantic
        // source of the contract. The negative half-tie cases fail under a
        // deliberate switch back to round-half-away-from-zero.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../test-fixtures/pcm-rounding.json"
        );
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("read shared fixture {path}: {err}"));
        let fixture: serde_json::Value =
            serde_json::from_str(&raw).expect("fixture is valid JSON");
        let cases = fixture["cases"].as_array().expect("fixture cases array");
        assert!(cases.len() >= 15, "fixture must keep its coverage");

        for case in cases {
            let input = match &case["input"] {
                serde_json::Value::Number(number) => {
                    number.as_f64().expect("finite fixture input") as f32
                }
                serde_json::Value::String(literal) => match literal.as_str() {
                    "NaN" => f32::NAN,
                    "Infinity" => f32::INFINITY,
                    "-Infinity" => f32::NEG_INFINITY,
                    other => panic!("unsupported non-finite literal {other}"),
                },
                other => panic!("unsupported fixture input {other}"),
            };
            let expected = case["expected"].as_i64().expect("expected i16") as i16;
            assert_eq!(pcm16(input), expected, "fixture case {case}");
        }
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
    fn resample_preserves_lengths_and_passes_native_through() {
        // Downsampling 8 kHz → 16 kHz: 3 frames become 6, all finite.
        // (Exact sample values are no longer pinned to linear interpolation;
        // the anti-aliasing suite below owns spectral behavior.)
        let audio = mono(&[0.0, 1.0, 0.0], 8_000);
        let output = resample_to_16k(&audio).expect("resample succeeds");
        assert_eq!(output.len(), 6);
        assert!(output.iter().all(|sample| sample.is_finite()));

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
    fn rejects_input_rates_beyond_the_resampling_ceiling() {
        // R15: a mislabeled or hostile rate field must be rejected, not
        // resampled — the kernel half-width grows as ~8.9 × rate/16000,
        // so a u32::max rate would demand ~2.4M taps per output sample.
        assert_eq!(
            message_of(resample_to_16k(&mono(&[0.5], u32::MAX))),
            "sampleRate must not exceed 384000 Hz"
        );
        assert_eq!(
            message_of(resample_to_16k(&mono(&[0.5], MAX_RESAMPLE_INPUT_RATE + 1))),
            format!("sampleRate must not exceed {MAX_RESAMPLE_INPUT_RATE} Hz")
        );

        // The ceiling itself still resamples (output length clamps to >= 1).
        assert_eq!(
            resample_to_16k(&mono(&[0.5], MAX_RESAMPLE_INPUT_RATE))
                .expect("384 kHz is the documented ceiling")
                .len(),
            1
        );
    }

    #[test]
    fn a_mislabeled_import_rate_is_rejected_at_prepare_time() {
        // End to end: the WAV decoder accepts any nonzero rate, so the
        // ceiling is what stops a mislabeled import from reaching the
        // sinc path.
        let fmt = chunk(b"fmt ", &fmt_body(1, 1_000_000, 16, 1));
        let data = chunk(b"data", &[0, 0, 0, 0]);
        let bytes = wav_from_chunks(&[fmt, data]);
        assert_eq!(
            message_of(prepare_wav_16k(&bytes)),
            format!("sampleRate must not exceed {MAX_RESAMPLE_INPUT_RATE} Hz")
        );
    }

    #[test]
    fn a_degenerate_kernel_replicates_the_edge_instead_of_fabricating_silence() {
        // R16: when a kernel's weight sum is not positive, the output must
        // be the replicated input sample nearest the position — 0.0 would
        // quietly punch silence into the take.
        //
        // These parameters cannot come from a legal rate (production cutoff
        // is capped at 0.45); cutoff just past Nyquist drives the windowed
        // sinc's DC sum negative at the half-sample position, exercising the
        // defensive branch deterministically.
        let input = [0.1, 0.2, 0.3, 0.4, 0.75, 0.6, 0.5, 0.4, 0.3];
        let center = 4i64;
        let value = sinc_sample(&input, center, 0.5, 4, 1.001);
        assert_eq!(value, input[center as usize]);
        assert_ne!(value, 0.0, "degenerate kernels must not emit silence");

        // Outside the input, the fallback clamps to the nearest edge.
        assert_eq!(sinc_sample(&input, -50, 0.5, 4, 1.001), input[0]);
        assert_eq!(sinc_sample(&input, 500, 0.5, 4, 1.001), input[input.len() - 1]);
    }

    #[test]
    fn the_sinc_kernel_holds_dc_at_exact_unity_gain() {
        // Every tap of a constant input contributes the same sample, so
        // sum / weight_sum must return it bit-exactly — the normalized
        // kernel's DC gain is pinned to 1 (the property the degenerate
        // fallback protects when the weights themselves go wrong).
        let input = [0.25f32; 64];
        assert_eq!(sinc_sample(&input, 32, 0.0, 9, 0.45), 0.25);
        assert_eq!(sinc_sample(&input, 32, 0.37, 9, 0.45), 0.25);
    }

    #[test]
    fn js_round_ties_go_toward_positive_infinity_as_positive_zero() {
        // Documented divergence (R17): ECMAScript yields -0 for
        // -0.5 <= x < 0; the floor+1 form yields +0. Both encode to the
        // same pcm16 sample, and the tie direction itself still matches
        // JS for every nonzero result.
        assert_eq!(js_round(-1.5), -1.0);
        assert_eq!(js_round(-0.5), 0.0);
        assert!(js_round(-0.5).is_sign_positive());
        assert_eq!(js_round(-0.2), 0.0);
        assert!(js_round(-0.2).is_sign_positive());
        assert_eq!(js_round(0.5), 1.0);
        assert_eq!(js_round(1.5), 2.0);
    }

    /// Amplitude of `frequency_hz` in `samples` at `sample_rate`, by complex
    /// correlation over a whole number of cycles (leakage-free for pure
    /// tones whose cycle count is an integer).
    fn tone_amplitude(samples: &[f32], sample_rate: u32, frequency_hz: f64) -> f64 {
        let cycles = ((samples.len() as f64 * frequency_hz) / f64::from(sample_rate)).floor();
        let window = ((cycles * f64::from(sample_rate)) / frequency_hz).round() as usize;
        let mut real = 0.0f64;
        let mut imaginary = 0.0f64;

        for (index, &sample) in samples.iter().take(window).enumerate() {
            let phase = 2.0 * std::f64::consts::PI * frequency_hz * index as f64
                / f64::from(sample_rate);
            real += f64::from(sample) * phase.cos();
            imaginary -= f64::from(sample) * phase.sin();
        }

        if window == 0 {
            return 0.0;
        }
        2.0 * (real * real + imaginary * imaginary).sqrt() / window as f64
    }

    /// The seeded linear interpolation this port originally shipped with
    /// (issue #122 regression control): reads every `ratio`-th input sample.
    fn linear_resample(mono_samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let output_length = ((mono_samples.len() as f64 * f64::from(STARLING_SAMPLE_RATE))
            / f64::from(sample_rate))
        .round()
        .max(1.0) as usize;
        let ratio = f64::from(sample_rate) / f64::from(STARLING_SAMPLE_RATE);
        let last = mono_samples.len() - 1;
        let mut output = Vec::with_capacity(output_length);

        for index in 0..output_length {
            let position = index as f64 * ratio;
            let left = (position.floor() as usize).min(last);
            let right = (left + 1).min(last);
            let fraction = position - left as f64;
            let left_sample = f64::from(mono_samples[left]);
            output
                .push((left_sample + (f64::from(mono_samples[right]) - left_sample) * fraction)
                    as f32);
        }

        output
    }

    #[test]
    fn suppresses_the_12khz_alias_from_48khz_input() {
        // Issue #122: a 12 kHz tone at 48 kHz decimated 3:1 aliases to 4 kHz
        // at unchanged level (the linear kernel reads positions 0, 3, 6, …
        // of sin(pi/2 * n) → 0, -0.8, 0, 0.8 …).
        let amplitude = 0.8f64;
        let samples: Vec<f32> = (0..48_000)
            .map(|index| {
                (amplitude * (std::f64::consts::PI * index as f64 / 2.0).sin()) as f32
            })
            .collect();
        let audio = mono(&samples, 48_000);

        let output = resample_to_16k(&audio).expect("resample succeeds");
        assert_eq!(output.len(), 16_000);

        let leaked = tone_amplitude(&output, STARLING_SAMPLE_RATE, 4_000.0);
        assert!(
            leaked <= amplitude * 10.0f64.powf(-40.0 / 20.0),
            "12 kHz must not fold to 4 kHz above -40 dB: leaked {leaked}"
        );

        // The fixture has teeth: the seeded linear implementation passes the
        // alias at (essentially) full amplitude.
        let linear = linear_resample(&samples, 48_000);
        let linear_leak = tone_amplitude(&linear, STARLING_SAMPLE_RATE, 4_000.0);
        assert!(
            linear_leak > amplitude * 0.5,
            "linear control must fail this fixture: leaked {linear_leak}"
        );
    }

    #[test]
    fn passes_a_1khz_tone_within_one_db_at_common_rates() {
        for rate in [48_000u32, 44_100] {
            let amplitude = 0.7f64;
            let samples: Vec<f32> = (0..rate)
                .map(|index| {
                    (amplitude
                        * (2.0 * std::f64::consts::PI * 1_000.0 * index as f64
                            / f64::from(rate))
                        .sin()) as f32
                })
                .collect();
            let audio = mono(&samples, rate);

            let output = resample_to_16k(&audio).expect("resample succeeds");
            assert_eq!(output.len(), 16_000, "1 s at {rate} Hz → 16_000 frames");

            // 1000 whole cycles at 16 kHz: exact correlation bin.
            let passed = tone_amplitude(&output, STARLING_SAMPLE_RATE, 1_000.0);
            let ratio = passed / amplitude;
            assert!(
                (10.0f64.powf(-1.0 / 20.0)..=10.0f64.powf(1.0 / 20.0)).contains(&ratio),
                "{rate} Hz in-band 1 kHz tone must stay within 1 dB: {ratio}"
            );
        }
    }

    #[test]
    fn resample_passes_dc_with_unity_gain() {
        let samples = vec![0.25f32; 1_000];
        let output = resample_to_16k(&mono(&samples, 48_000)).expect("resample succeeds");
        assert_eq!(output.len(), 333); // round(1000 * 16000/48000)
        for sample in &output {
            assert!(
                (sample - 0.25).abs() <= 1e-9,
                "DC must pass at unity gain, got {sample}"
            );
        }
    }

    #[test]
    fn resample_spreads_an_impulse_without_inventing_energy() {
        let mut samples = vec![0.0f32; 4_800];
        samples[2_400] = 1.0;
        let output = resample_to_16k(&mono(&samples, 48_000)).expect("resample succeeds");
        assert_eq!(output.len(), 1_600);
        assert!(output.iter().all(|sample| sample.is_finite()));

        let peak = output.iter().fold(0.0f32, |max, sample| max.max(sample.abs()));
        assert!(peak <= 1.3, "windowed-sinc overshoot stays bounded: {peak}");

        let energy: f64 = output.iter().map(|sample| f64::from(*sample).powi(2)).sum();
        // Band-limiting keeps roughly the in-band share (~7.2/24 kHz = 0.3);
        // no energy may be invented, none silently nulled.
        assert!(
            (0.05..=1.5).contains(&energy),
            "impulse energy out of bounds: {energy}"
        );
    }

    #[test]
    fn resample_replicates_a_trailing_edge_without_smearing_it_away() {
        // 10 ms of silence, then a full-scale step for the last 10 samples:
        // boundary replication must represent the edge (duration kept, no
        // blow-up, final output close to the step level).
        let mut samples = vec![0.0f32; 470];
        samples.resize(480, 1.0);
        let output = resample_to_16k(&mono(&samples, 48_000)).expect("resample succeeds");
        assert_eq!(output.len(), 160);
        assert!(output.iter().all(|sample| sample.is_finite()));
        assert!(output.iter().all(|sample| sample.abs() <= 1.3));
        assert!(
            output.last().copied().unwrap_or(0.0) >= 0.5,
            "edge step must survive into the final output"
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
