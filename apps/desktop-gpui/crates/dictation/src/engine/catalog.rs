//! The verified model catalog (#363 step 2, reusing #222's machinery).
//!
//! Every entry pins an exact Hugging Face revision, byte size, and sha256,
//! so a download is only ever "fetch these exact bytes" — no floating
//! `resolve/main` that can change under the user. The manager accepts an
//! injected catalog (tests point entries at a local HTTP server), which
//! is why the fields are owned `String`s rather than `&'static str`.

/// One downloadable model. `slug` is the server's `--model` form (what a
/// transcription request must send as `model`); `id` is this app's stable
/// model identity used in settings, provenance, and the registry.
#[derive(Clone, Debug, PartialEq)]
pub struct CatalogEntry {
    /// App-stable model identity (persisted as the active model).
    pub id: String,
    /// Human label for pickers.
    pub label: String,
    /// The server `--model` slug (e.g. `parakeet`).
    pub slug: String,
    /// Pinned download URL (exact revision).
    pub url: String,
    /// File name inside the models dir (last URL path segment).
    pub file_name: String,
    /// Exact expected size in bytes.
    pub size_bytes: u64,
    /// Expected sha256 (hex).
    pub sha256: String,
    /// Whether this is a recommended default (#348).
    pub recommended: bool,
    /// One-line honest note about trade-offs.
    pub note: String,
}

impl CatalogEntry {
    /// Builds an entry, deriving `file_name` from the URL's last path
    /// segment — the name the file is downloaded and stored under. The
    /// eight parameters are exactly the catalog's columns.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: &str,
        label: &str,
        slug: &str,
        url: &str,
        size_bytes: u64,
        sha256: &str,
        recommended: bool,
        note: &str,
    ) -> CatalogEntry {
        CatalogEntry {
            id: id.to_string(),
            label: label.to_string(),
            slug: slug.to_string(),
            url: url.to_string(),
            file_name: url_file_name(url),
            size_bytes,
            sha256: sha256.to_string(),
            recommended,
            note: note.to_string(),
        }
    }
}

/// The last path segment of a URL, percent-decoding left alone (the HF
/// file names in this catalog contain no escapes).
pub fn url_file_name(url: &str) -> String {
    url.rsplit('/')
        .next()
        .filter(|segment| !segment.is_empty())
        .unwrap_or(url)
        .to_string()
}

/// The built-in catalog. Sizes and digests are exact; verify-on-install
/// treats any deviation as a broken download, never a silent pass.
pub fn default_catalog() -> Vec<CatalogEntry> {
    vec![
        CatalogEntry::new(
            "parakeet-v3-q4km-s16",
            "Parakeet TDT 0.6B v3 (q4_k_m)",
            "parakeet",
            "https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf/resolve/96402b32bd374742aa1da3c66af30aa64cea3fdb/parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf",
            552_670_624,
            "2b5ea37e3193c71b3ad2f859b4238ff4501898ba73d63899568faee1daae9982",
            true,
            "Smallest recommended file; 25 European languages.",
        ),
        CatalogEntry::new(
            "parakeet-v3-q8",
            "Parakeet TDT 0.6B v3 (q8_0)",
            "parakeet",
            "https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf/resolve/96402b32bd374742aa1da3c66af30aa64cea3fdb/parakeet-tdt-0.6b-v3-q8_0.gguf",
            906_000_288,
            "cbab40be5510f86f825ccb19bdea0876938358a2266a24a1ab8f1fccf0759922",
            false,
            "Closest to full precision; larger.",
        ),
        CatalogEntry::new(
            "moss-2b-q4e8",
            "MOSS Transcribe preview 2B (q4)",
            "moss",
            "https://huggingface.co/scholzmx/moss-transcribe-preview-2b-gguf/resolve/4e647503eca9769761c06d9d2a5212104f4592b8/moss-transcribe-preview-2b-q4e8-fullimx.gguf",
            1_552_993_088,
            "5658f3107a72bc7d74c3c428615fde9c95b439a436b9a3b1f387cf2f82a439ff",
            false,
            "Autoregressive; slower, more memory.",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_catalog_pins_exact_values() {
        let catalog = default_catalog();
        assert_eq!(catalog.len(), 3);

        let parakeet_q4 = &catalog[0];
        assert_eq!(parakeet_q4.id, "parakeet-v3-q4km-s16");
        assert_eq!(parakeet_q4.label, "Parakeet TDT 0.6B v3 (q4_k_m)");
        assert_eq!(parakeet_q4.slug, "parakeet");
        assert_eq!(parakeet_q4.size_bytes, 552_670_624);
        assert_eq!(
            parakeet_q4.sha256,
            "2b5ea37e3193c71b3ad2f859b4238ff4501898ba73d63899568faee1daae9982"
        );
        assert!(parakeet_q4.recommended);
        assert_eq!(
            parakeet_q4.note,
            "Smallest recommended file; 25 European languages."
        );
        assert_eq!(
            parakeet_q4.url,
            "https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf/resolve/96402b32bd374742aa1da3c66af30aa64cea3fdb/parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf"
        );

        let parakeet_q8 = &catalog[1];
        assert_eq!(parakeet_q8.id, "parakeet-v3-q8");
        assert_eq!(parakeet_q8.size_bytes, 906_000_288);
        assert_eq!(
            parakeet_q8.sha256,
            "cbab40be5510f86f825ccb19bdea0876938358a2266a24a1ab8f1fccf0759922"
        );
        assert!(!parakeet_q8.recommended);

        let moss = &catalog[2];
        assert_eq!(moss.id, "moss-2b-q4e8");
        assert_eq!(moss.slug, "moss");
        assert_eq!(moss.size_bytes, 1_552_993_088);
        assert_eq!(
            moss.sha256,
            "5658f3107a72bc7d74c3c428615fde9c95b439a436b9a3b1f387cf2f82a439ff"
        );
    }

    #[test]
    fn file_names_are_the_last_url_segment() {
        for entry in default_catalog() {
            assert_eq!(entry.file_name, url_file_name(&entry.url));
            assert!(!entry.file_name.contains('/'));
            assert!(entry.file_name.ends_with(".gguf"));
        }
        assert_eq!(url_file_name("https://host/a/b.gguf?x=1"), "b.gguf?x=1");
        assert_eq!(url_file_name("https://host/"), "https://host/");
    }
}
