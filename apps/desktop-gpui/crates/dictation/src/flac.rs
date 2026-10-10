//! Lossless at-rest history audio (#342).
//!
//! A finalized take is kept as FLAC holding exactly the PCM16 samples the
//! request path sends ([`crate::audio::request_pcm16`]: 16 kHz mono, the
//! `pcm16` quantizer). Decoding returns those samples bit for bit, and
//! [`crate::audio::pcm16_to_f32`] maps them back to floats that quantize
//! to the same values, so a retry from FLAC produces byte-identical
//! request audio to a retry from the original journal.
//!
//! What is *not* kept: the journal's f32 precision below one PCM16 step
//! and, for a device that captured above 16 kHz, the band above 8 kHz.
//! Neither ever reaches a transcription — every request is this PCM16
//! at 16 kHz — which is the fidelity contract the store compresses to.
//!
//! The encoder (`flacenc`) and the decoder (`claxon`) are independent
//! implementations; [`encode`] output is decoded and compared sample for
//! sample before the store lets it replace a journal, and [`decode`]
//! checks the STREAMINFO MD5 signature on every load.

use std::io::Read;

use flacenc::component::BitRepr;
use flacenc::error::Verify;
use md5::{Digest, Md5};

use crate::audio::STARLING_SAMPLE_RATE;

/// File extension of FLAC audio in the store's trees.
pub const FLAC_EXT: &str = "flac";

/// Why FLAC audio could not be produced or read back.
#[derive(Debug, thiserror::Error)]
pub enum FlacError {
    #[error("FLAC encode failed: {0}")]
    Encode(String),
    #[error("FLAC audio is unreadable: {0}")]
    Decode(String),
    #[error("FLAC audio is not 16 kHz mono 16-bit (got {rate} Hz, {channels} channel(s), {bits} bits)")]
    Format { rate: u32, channels: u32, bits: u32 },
    #[error("FLAC audio failed verification: {0}")]
    Mismatch(String),
}

/// Shortest audio [`encode`] accepts: a FLAC stream holding fewer
/// samples than the format's minimum block size is refused by the
/// decoder, so such a take (a millisecond) simply stays a journal.
pub const MIN_SAMPLES: usize = 16;

/// Encode 16 kHz mono PCM16 samples as a FLAC stream. Fewer than
/// [`MIN_SAMPLES`] samples are refused.
pub fn encode(samples: &[i16]) -> Result<Vec<u8>, FlacError> {
    if samples.len() < MIN_SAMPLES {
        return Err(FlacError::Encode(format!(
            "{} samples is shorter than the {MIN_SAMPLES}-sample minimum",
            samples.len()
        )));
    }
    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|(_, err)| FlacError::Encode(err.to_string()))?;
    let widened: Vec<i32> = samples.iter().map(|&sample| i32::from(sample)).collect();
    let source = flacenc::source::MemSource::from_samples(
        &widened,
        1,
        16,
        STARLING_SAMPLE_RATE as usize,
    );
    let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|err| FlacError::Encode(err.to_string()))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|err| FlacError::Encode(err.to_string()))?;
    Ok(sink.as_slice().to_vec())
}

/// Decode a FLAC stream written by [`encode`]: 16 kHz mono 16-bit only.
/// Frame CRCs are checked by the decoder; the sample count and the
/// STREAMINFO MD5 signature are checked here, so a damaged file is an
/// error, never quietly different audio.
pub fn decode(reader: impl Read) -> Result<Vec<i16>, FlacError> {
    let mut reader =
        claxon::FlacReader::new(reader).map_err(|err| FlacError::Decode(err.to_string()))?;
    let info = reader.streaminfo();
    if info.sample_rate != STARLING_SAMPLE_RATE || info.channels != 1 || info.bits_per_sample != 16
    {
        return Err(FlacError::Format {
            rate: info.sample_rate,
            channels: info.channels,
            bits: info.bits_per_sample,
        });
    }
    let mut samples = Vec::with_capacity(info.samples.unwrap_or(0) as usize);
    let mut md5 = Md5::new();
    for sample in reader.samples() {
        let sample = sample.map_err(|err| FlacError::Decode(err.to_string()))?;
        let sample = i16::try_from(sample)
            .map_err(|_| FlacError::Decode(format!("sample {sample} exceeds 16 bits")))?;
        md5.update(sample.to_le_bytes());
        samples.push(sample);
    }
    if let Some(expected) = info.samples {
        if expected != samples.len() as u64 {
            return Err(FlacError::Mismatch(format!(
                "STREAMINFO declares {expected} samples, {} decoded",
                samples.len()
            )));
        }
    }
    // An all-zero signature means "not computed" (FLAC format); every
    // file [`encode`] writes carries one.
    if info.md5sum != [0u8; 16] && md5.finalize().as_slice() != info.md5sum {
        return Err(FlacError::Mismatch(
            "the MD5 signature of the decoded samples does not match STREAMINFO".to_string(),
        ));
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::{encode_wav_16k_parts, pcm16_to_f32, request_pcm16};

    /// xorshift noise: deterministic, full-scale, incompressible.
    fn noise(len: usize, mut state: u32) -> Vec<i16> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as i16
            })
            .collect()
    }

    #[test]
    fn the_pcm16_inverse_is_exact_for_every_value() {
        // Exhaustive: the store's whole retry guarantee rests on this.
        for value in i16::MIN..=i16::MAX {
            let back = request_pcm16(&[pcm16_to_f32(value)], STARLING_SAMPLE_RATE)
                .expect("quantize");
            assert_eq!(back, vec![value], "value {value}");
        }
    }

    #[test]
    fn round_trips_are_sample_exact_across_signal_shapes() {
        let mut edge = vec![i16::MIN, i16::MAX, 0, -1, 1, i16::MIN + 1, i16::MAX - 1];
        edge.extend(std::iter::repeat_n(i16::MAX, 5_000));
        edge.extend(std::iter::repeat_n(i16::MIN, 5_000));
        let cases: Vec<(&str, Vec<i16>)> = vec![
            ("minimum length", (0..MIN_SAMPLES as i16).map(|i| i * 999).collect()),
            ("one short block", (0..100).map(|i| (i * 37) as i16).collect()),
            ("silence", vec![0; 48_000]),
            ("full-scale edges", edge),
            ("noise", noise(70_001, 0x9e37_79b9)),
            (
                "tone",
                (0..64_000)
                    .map(|i| ((i as f64 * 0.0863).sin() * 20_000.0).round() as i16)
                    .collect(),
            ),
        ];
        for (name, samples) in cases {
            let bytes = encode(&samples).expect(name);
            let decoded = decode(bytes.as_slice()).expect(name);
            assert_eq!(decoded, samples, "{name}");
        }
    }

    #[test]
    fn speech_like_audio_compresses() {
        let tone: Vec<i16> = (0..160_000)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                let envelope = 0.6 + 0.4 * (2.0 * std::f64::consts::PI * 10.0 * t).sin();
                (12_000.0 * envelope * (2.0 * std::f64::consts::PI * 220.0 * t).sin()) as i16
            })
            .collect();
        let bytes = encode(&tone).expect("encode");
        assert!(
            bytes.len() * 10 < tone.len() * 2 * 6,
            "{} bytes for {} PCM16 bytes",
            bytes.len(),
            tone.len() * 2
        );
    }

    #[test]
    fn a_damaged_file_is_an_error_not_different_audio() {
        let samples = noise(20_000, 7);
        let mut bytes = encode(&samples).expect("encode");
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0x40;
        assert!(decode(bytes.as_slice()).is_err());

        let truncated = encode(&samples).expect("encode");
        assert!(decode(&truncated[..truncated.len() - 10]).is_err());
    }

    #[test]
    fn other_formats_are_refused() {
        let config = flacenc::config::Encoder::default().into_verified().unwrap();
        let source = flacenc::source::MemSource::from_samples(&[0i32; 4_000], 2, 16, 16_000);
        let stream = flacenc::encode_with_fixed_block_size(&config, source, 1_000).unwrap();
        let mut sink = flacenc::bitsink::ByteSink::new();
        stream.write(&mut sink).unwrap();
        assert!(matches!(
            decode(sink.as_slice()),
            Err(FlacError::Format { channels: 2, .. })
        ));
        assert!(encode(&[]).is_err());
        assert!(encode(&[7; MIN_SAMPLES - 1]).is_err());
    }

    #[test]
    fn decoded_audio_rebuilds_the_request_wav_byte_for_byte() {
        // A 48 kHz capture: the request path resamples, FLAC stores the
        // result, and the WAV rebuilt from FLAC matches the original.
        let captured: Vec<f32> = (0..48_000 * 2)
            .map(|i| ((i as f32 * 0.013).sin() * 0.4) + ((i as f32 * 0.31).sin() * 0.05))
            .collect();
        let direct = encode_wav_16k_parts(&captured, 48_000, 1).expect("direct wav");
        let stored = request_pcm16(&captured, 48_000).expect("pcm16");
        let decoded = decode(encode(&stored).expect("encode").as_slice()).expect("decode");
        let floats: Vec<f32> = decoded.iter().map(|&q| pcm16_to_f32(q)).collect();
        let rebuilt = encode_wav_16k_parts(&floats, STARLING_SAMPLE_RATE, 1).expect("rebuilt");
        assert_eq!(rebuilt, direct);
    }
}
