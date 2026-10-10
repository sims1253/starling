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
    #[error(
        "FLAC audio is not 16 kHz mono 16-bit (got {rate} Hz, {channels} channel(s), {bits} bits)"
    )]
    Format { rate: u32, channels: u32, bits: u32 },
    #[error("FLAC audio failed verification: {0}")]
    Mismatch(String),
}

/// Shortest audio [`encode`] accepts: the format's 16-sample minimum
/// block size. Only a stream's last frame may be shorter, so a take
/// below it (a millisecond) is not worth a FLAC file and stays a journal.
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
    let source =
        flacenc::source::MemSource::from_samples(&widened, 1, 16, STARLING_SAMPLE_RATE as usize);
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|err| FlacError::Encode(err.to_string()))?;
    // flacenc folds the short last frame into STREAMINFO's minimum block
    // size. In a fixed-blocksize stream the minimum excludes the last
    // block and equals the maximum (RFC 9639 §8.2); with them unequal,
    // libFLAC (`flac -t`) maps frame numbers to the wrong sample numbers
    // and warns on every frame.
    stream
        .stream_info_mut()
        .set_block_sizes(config.block_size, config.block_size)
        .map_err(|err| FlacError::Encode(err.to_string()))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|err| FlacError::Encode(err.to_string()))?;
    Ok(sink.as_slice().to_vec())
}

/// The most samples [`decode`] reserves up front from STREAMINFO's count
/// (ten minutes at 16 kHz); a longer take grows the buffer as it decodes.
const MAX_RESERVED_SAMPLES: u64 = 10 * 60 * STARLING_SAMPLE_RATE as u64;

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
    // The declared count is only checked after decoding: a damaged header
    // must not size the allocation.
    let reserve = info.samples.unwrap_or(0).min(MAX_RESERVED_SAMPLES);
    let mut samples = Vec::with_capacity(reserve as usize);
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
            let back =
                request_pcm16(&[pcm16_to_f32(value)], STARLING_SAMPLE_RATE).expect("quantize");
            assert_eq!(back, vec![value], "value {value}");
        }
    }

    #[test]
    fn round_trips_are_sample_exact_across_signal_shapes() {
        let mut edge = vec![i16::MIN, i16::MAX, 0, -1, 1, i16::MIN + 1, i16::MAX - 1];
        edge.extend(std::iter::repeat_n(i16::MAX, 5_000));
        edge.extend(std::iter::repeat_n(i16::MIN, 5_000));
        let cases: Vec<(&str, Vec<i16>)> = vec![
            (
                "minimum length",
                (0..MIN_SAMPLES as i16).map(|i| i * 999).collect(),
            ),
            (
                "one short block",
                (0..100).map(|i| (i * 37) as i16).collect(),
            ),
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

    #[test]
    fn a_header_declaring_a_huge_count_is_an_error_not_an_allocation() {
        let mut bytes = encode(&noise(8_192, 11)).expect("encode");
        // STREAMINFO's 36-bit total sample count: the low bits of the
        // big-endian u64 at offset 18 ("fLaC", block header, 10 bytes of
        // block and frame sizes).
        let mut packed = u64::from_be_bytes(bytes[18..26].try_into().expect("8 bytes"));
        packed |= (1u64 << 36) - 1;
        bytes[18..26].copy_from_slice(&packed.to_be_bytes());
        match decode(bytes.as_slice()) {
            Err(FlacError::Mismatch(reason)) => assert!(reason.contains("declares"), "{reason}"),
            other => panic!("expected a count mismatch, got {other:?}"),
        }
    }

    /// Lengths around the 4096-sample block: shorter than one block, an
    /// exact multiple, and a short last frame.
    fn block_edge_cases() -> Vec<Vec<i16>> {
        [MIN_SAMPLES, 100, 4_096, 8_192, 70_001]
            .into_iter()
            .map(|len| {
                (0..len)
                    .map(|i| ((i as f64 * 0.0863).sin() * 20_000.0).round() as i16)
                    .collect()
            })
            .collect()
    }

    /// CRC-8 (polynomial 0x07) that closes a FLAC frame header.
    fn crc8(bytes: &[u8]) -> u8 {
        bytes.iter().fold(0u8, |mut crc, &byte| {
            crc ^= byte;
            for _ in 0..8 {
                crc = if crc & 0x80 != 0 {
                    (crc << 1) ^ 0x07
                } else {
                    crc << 1
                };
            }
            crc
        })
    }

    /// One parsed frame header: (variable-blocking flag, coded frame or
    /// sample number, block size).
    type FrameHeader = (bool, u64, usize);

    /// Parse a 16 kHz mono 16-bit frame header at `at`; `None` unless the
    /// sync code, the fixed fields and the header CRC-8 all match.
    fn frame_header_at(bytes: &[u8], at: usize) -> Option<FrameHeader> {
        let header = bytes.get(at..)?;
        if header.len() < 6 || header[0] != 0xFF || header[1] & 0xFE != 0xF8 {
            return None;
        }
        // 16 kHz rate code; mono, 16 bits, reserved bit clear.
        if header[2] & 0x0F != 0x05 || header[3] != 0x08 {
            return None;
        }
        let lead = header[4].leading_ones() as usize;
        let (mut number, len) = match lead {
            0 => (u64::from(header[4]), 1),
            2..=7 => (u64::from(header[4] & (0x7F >> lead)), lead),
            _ => return None,
        };
        for &byte in header.get(5..4 + len)? {
            if byte & 0xC0 != 0x80 {
                return None;
            }
            number = (number << 6) | u64::from(byte & 0x3F);
        }
        let mut end = 4 + len;
        let block_size = match header[2] >> 4 {
            1 => 192,
            tag @ 2..=5 => 576 << (tag - 2),
            6 => {
                end += 1;
                usize::from(*header.get(end - 1)?) + 1
            }
            7 => {
                end += 2;
                usize::from(u16::from_be_bytes(
                    header.get(end - 2..end)?.try_into().ok()?,
                )) + 1
            }
            tag @ 8..=15 => 256 << (tag - 8),
            _ => return None,
        };
        (crc8(&header[..end]) == *header.get(end)?).then_some((
            header[1] & 1 == 1,
            number,
            block_size,
        ))
    }

    /// One step of the CRC-16 (polynomial 0x8005) that closes a FLAC frame.
    fn crc16(crc: u16, byte: u8) -> u16 {
        let mut crc = crc ^ (u16::from(byte) << 8);
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x8005
            } else {
                crc << 1
            };
        }
        crc
    }

    /// Every frame header of a stream [`encode`] wrote, in order. Frames
    /// are walked boundary to boundary: a frame ends at the first point
    /// where the bytes so far are followed by their own CRC-16 and then by
    /// the end of the stream or another valid header.
    fn frame_headers(bytes: &[u8]) -> Vec<FrameHeader> {
        assert_eq!(&bytes[..4], b"fLaC");
        let mut at = 4;
        loop {
            let last = bytes[at] & 0x80 != 0;
            let len = u32::from_be_bytes([0, bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
            at += 4 + len as usize;
            if last {
                break;
            }
        }
        let mut headers = Vec::new();
        while at < bytes.len() {
            let header =
                frame_header_at(bytes, at).unwrap_or_else(|| panic!("no frame header at {at}"));
            headers.push(header);
            let mut crc = 0u16;
            let mut end = None;
            for offset in at..bytes.len().saturating_sub(1) {
                let footer = u16::from_be_bytes([bytes[offset], bytes[offset + 1]]);
                // A header is at least 6 bytes, CRC-8 included.
                if offset >= at + 6
                    && footer == crc
                    && (offset + 2 == bytes.len() || frame_header_at(bytes, offset + 2).is_some())
                {
                    end = Some(offset + 2);
                    break;
                }
                crc = crc16(crc, bytes[offset]);
            }
            at = end.unwrap_or_else(|| panic!("frame at {at} has no CRC-16 footer"));
        }
        headers
    }

    #[test]
    fn frame_headers_number_a_fixed_blocksize_stream() {
        for samples in block_edge_cases() {
            let bytes = encode(&samples).expect("encode");
            let len = samples.len();
            // STREAMINFO min and max block size (after "fLaC" and the
            // 4-byte block header): equal, so the stream is fixed-blocksize.
            assert_eq!(&bytes[8..12], &[0x10, 0x00, 0x10, 0x00], "{len} samples");
            let headers = frame_headers(&bytes);
            assert_eq!(headers.len(), len.div_ceil(4_096), "{len} samples");
            let mut decoded = 0;
            for (index, &(variable, number, block_size)) in headers.iter().enumerate() {
                assert!(
                    !variable,
                    "{len} samples: frame {index} is variable-blocking"
                );
                assert_eq!(number, index as u64, "{len} samples");
                let expected = (len - index * 4_096).min(4_096);
                assert_eq!(block_size, expected, "{len} samples: frame {index}");
                decoded += block_size;
            }
            assert_eq!(decoded, len);
        }
    }

    #[test]
    fn the_reference_decoder_reads_our_output_without_warnings() {
        use std::process::Command;
        if Command::new("flac").arg("--version").output().is_err() {
            eprintln!("skipping: the reference `flac` tool is not installed");
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        for samples in block_edge_cases() {
            let len = samples.len();
            let path = dir.path().join(format!("{len}.flac"));
            std::fs::write(&path, encode(&samples).expect("encode")).expect("write");
            // `-w` turns every warning, frame numbering included, into a
            // failed exit.
            let test = Command::new("flac")
                .args(["-t", "-s", "-w"])
                .arg(&path)
                .output()
                .expect("flac -t");
            let stderr = String::from_utf8_lossy(&test.stderr);
            assert!(test.status.success(), "flac -t, {len} samples: {stderr}");
            assert!(
                !stderr.contains("WARNING"),
                "flac -t, {len} samples: {stderr}"
            );

            let decode = Command::new("flac")
                .args(["-d", "-c", "-s", "-w", "--force-raw-format"])
                .args(["--endian=little", "--sign=signed"])
                .arg(&path)
                .output()
                .expect("flac -d");
            let stderr = String::from_utf8_lossy(&decode.stderr);
            assert!(decode.status.success(), "flac -d, {len} samples: {stderr}");
            assert!(
                !stderr.contains("WARNING"),
                "flac -d, {len} samples: {stderr}"
            );
            let decoded: Vec<i16> = decode
                .stdout
                .chunks_exact(2)
                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            assert_eq!(decoded, samples, "flac -d, {len} samples");
        }
    }
}
