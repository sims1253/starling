//! Magnitude spectrum + waveform levels for the live waveform (52 bars).
//!
//! Replaces the Web Audio `AnalyserNode` that fed `useRecorder.ts`: the bar
//! mapping loop (`stride = max(1, bins/bars)`, `max(0.045, v.powf(1.45))`) is
//! kept, but the analyser's Blackman window and dB smoothing are swapped for a
//! plain Hann window with linear magnitudes. Bit-exact parity is explicitly
//! not a goal; a lively waveform is.
//!
//! Scaling: with a periodic Hann window, a full-scale sine sitting on a bin
//! has peak |X[k]| ≈ sum(w)/2 = window_len/4, because Hann halves the coherent
//! gain. Dividing by `window_len/4` — the unwindowed `window_len/2` rule from
//! the port brief plus Hann's factor of two — maps a full-scale bin-centered
//! tone to ≈1.0, so normal speech lands in the visible part of 0.045..=1.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

const MIN_WINDOW: usize = 256;
const LEVEL_FLOOR: f32 = 0.045;
const LEVEL_EXPONENT: f32 = 1.45;

/// Hann-windowed magnitude spectrum (one-sided) of the samples. Input length is
/// truncated/zero-padded to a power of two >= 256. Returns `len/2` magnitudes
/// normalized to 0..=1.
pub fn magnitude_spectrum(samples: &[f32]) -> Vec<f32> {
    let window_len = samples.len().next_power_of_two().max(MIN_WINDOW);
    let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(window_len);
    let step = std::f32::consts::TAU / window_len as f32;
    for (index, &sample) in samples.iter().enumerate().take(window_len) {
        let window = 0.5 * (1.0 - (step * index as f32).cos()); // periodic Hann
        buffer.push(Complex::new(sample * window, 0.0));
    }
    buffer.resize(window_len, Complex::ZERO);

    plan_fft(window_len).process(&mut buffer);

    let norm = window_len as f32 / 4.0; // see module docs for the scaling
    buffer[..window_len / 2]
        .iter()
        .map(|bin| ((bin.re * bin.re + bin.im * bin.im).sqrt() / norm).clamp(0.0, 1.0))
        .collect()
}

/// Map magnitudes to `bars` levels using the AnalyserNode loop from
/// useRecorder.ts: `stride = max(1, bins/bars)`, `level = max(0.045,
/// v.powf(1.45))`, returned clamped to [0.045, 1.0]. The caller applies the
/// extra 0.06 idle floor.
pub fn waveform_levels(magnitudes: &[f32], bars: usize) -> Vec<f32> {
    if bars == 0 {
        return Vec::new();
    }
    let stride = (magnitudes.len() / bars).max(1);
    (0..bars)
        .map(|bar| {
            let value = magnitudes.get(bar * stride).copied().unwrap_or(0.0);
            value
                .powf(LEVEL_EXPONENT)
                .max(LEVEL_FLOOR)
                .clamp(LEVEL_FLOOR, 1.0)
        })
        .collect()
}

/// Forward plans per window size, cached because the length is constant while
/// recording.
fn plan_fft(window_len: usize) -> Arc<dyn Fft<f32>> {
    static PLANS: OnceLock<Mutex<HashMap<usize, Arc<dyn Fft<f32>>>>> = OnceLock::new();
    let mut plans = PLANS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("fft plan cache not poisoned");
    plans
        .entry(window_len)
        .or_insert_with(|| FftPlanner::<f32>::new().plan_fft_forward(window_len))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_at(bin: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (std::f32::consts::TAU * bin as f32 * i as f32 / len as f32).sin())
            .collect()
    }

    fn peak_bin(magnitudes: &[f32]) -> usize {
        magnitudes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("no NaN magnitudes"))
            .expect("non-empty magnitudes")
            .0
    }

    #[test]
    fn mid_band_sine_peaks_at_its_bin_and_drives_a_lively_bar() {
        // bin 126 = 14 * stride (512/52 = 9), i.e. a mid-band bin a bar samples.
        let magnitudes = magnitude_spectrum(&sine_at(126, 1024));
        assert_eq!(magnitudes.len(), 512);
        assert!((peak_bin(&magnitudes) as i32 - 126).abs() <= 1);
        assert!(magnitudes[126] > 0.5, "peak magnitude {}", magnitudes[126]);

        let levels = waveform_levels(&magnitudes, 52);
        assert_eq!(levels.len(), 52);
        assert!(levels[14] > 0.5, "bar level {}", levels[14]);
    }

    #[test]
    fn silence_maps_to_the_idle_floor() {
        let levels = waveform_levels(&magnitude_spectrum(&[0.0; 1024]), 52);
        assert_eq!(levels.len(), 52);
        assert!(levels.iter().all(|&level| level == LEVEL_FLOOR));
    }

    #[test]
    fn levels_match_bar_count_and_stay_in_range() {
        assert_eq!(waveform_levels(&[2.0; 10], 4), vec![1.0; 4]);
        assert_eq!(waveform_levels(&[], 7), vec![LEVEL_FLOOR; 7]);
        assert!(waveform_levels(&[0.5], 0).is_empty());
    }

    #[test]
    fn dc_input_peaks_at_bin_zero_and_short_input_is_padded() {
        let magnitudes = magnitude_spectrum(&[1.0; 512]);
        assert_eq!(magnitudes.len(), 256);
        assert!(magnitudes[0] > 0.99, "dc magnitude {}", magnitudes[0]);
        assert!(magnitudes[100] < 1e-4);

        // 300 samples zero-pad up to the next power of two, 512.
        assert_eq!(magnitude_spectrum(&[1.0; 300]).len(), 256);
    }
}
