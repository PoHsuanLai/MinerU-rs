//! Weight loading for Burn modules.
//!
//! This module wraps `burn-store` so every model crate loads `.pth` PyTorch
//! state-dicts and `.safetensors` files the same way, with the same key-remapping
//! ergonomics and the same *"every source key was consumed"* safety check.
//!
//! The single biggest correctness risk when porting a PyTorch model to Burn is a
//! silent key mismatch: a tensor in the checkpoint whose name does not line up
//! with any Burn module field is simply dropped, leaving that layer initialised
//! with random weights. [`load_weights`] surfaces exactly those keys via
//! [`Error::UnmappedKeys`] so a model crate can assert full coverage.

use std::path::Path;

use burn::prelude::Backend;
use burn_store::{KeyRemapper, ModuleSnapshot, ModuleStore, PytorchStore, SafetensorsStore};

use crate::error::{Error, Result};

/// Regex-based rename rules applied to source (checkpoint) tensor keys before they
/// are matched against Burn module field paths.
///
/// PyTorch state-dict keys frequently differ from Burn's field-path naming — a
/// `backbone.conv.weight` in PyTorch may need to become `backbone.conv.weight`
/// unchanged, or a prefix may need stripping / rewriting. Build a [`KeyRemap`]
/// with [`KeyRemap::rename`] (regex + replacement) and hand it to [`load_weights`].
///
/// Rules are applied in insertion order; replacement strings may reference regex
/// capture groups with `$1`, `$2`, ….
///
/// # Examples
///
/// ```
/// use mineru_burn_common::weights::KeyRemap;
///
/// let remap = KeyRemap::new()
///     .rename(r"^backbone\.(.*)$", "encoder.$1")
///     .expect("valid regex");
/// assert_eq!(
///     remap.apply_str("backbone.conv.weight").as_deref(),
///     Some("encoder.conv.weight"),
/// );
/// ```
#[derive(Default, Clone)]
pub struct KeyRemap {
    rules: Vec<(String, String)>,
}

impl KeyRemap {
    /// Creates an empty remapper (an identity transform).
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a rename rule: source keys matching the `from` regex have the matched
    /// portion rewritten using the `to` replacement (with `$n` capture refs).
    ///
    /// Returns [`Error::Config`] if `from` is not a valid regular expression.
    pub fn rename(mut self, from: impl Into<String>, to: impl Into<String>) -> Result<Self> {
        let from = from.into();
        let to = to.into();
        // Validate the pattern eagerly so bad rules fail at construction, not load.
        KeyRemapper::new()
            .add_pattern(&from, &to)
            .map_err(|e| Error::Config(format!("invalid remap pattern {from:?}: {e}")))?;
        self.rules.push((from, to));
        Ok(self)
    }

    /// Returns `true` if no rename rules have been added.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Builds the `burn-store` [`KeyRemapper`] for this rule set.
    ///
    /// The patterns were already validated in [`KeyRemap::rename`], so this only
    /// fails if a rule was constructed some other way; that is mapped to
    /// [`Error::Config`] rather than panicking.
    fn build(&self) -> Result<KeyRemapper> {
        let mut remapper = KeyRemapper::new();
        for (from, to) in &self.rules {
            remapper = remapper
                .add_pattern(from, to)
                .map_err(|e| Error::Config(format!("invalid remap pattern {from:?}: {e}")))?;
        }
        Ok(remapper)
    }

    /// Applies the rules to a single key, for testing and diagnostics.
    ///
    /// Returns `Some(remapped)` if any rule matched, or `None` if the key was left
    /// unchanged by every rule.
    pub fn apply_str(&self, key: &str) -> Option<String> {
        let mut current = key.to_string();
        let mut changed = false;
        for (from, to) in &self.rules {
            // Patterns are pre-validated in `rename`; a failure here can only mean a
            // rule reached this vec without going through `rename`, so skip it.
            let Ok(re) = regex::Regex::new(from) else {
                continue;
            };
            if re.is_match(&current) {
                current = re.replace_all(&current, to.as_str()).into_owned();
                changed = true;
            }
        }
        changed.then_some(current)
    }
}

/// How strictly to treat source keys that never matched a module field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Coverage {
    /// Every source key must be consumed. Leftover keys become
    /// [`Error::UnmappedKeys`]. This is the recommended default for a finished port.
    #[default]
    Strict,
    /// Leftover source keys are logged at `warn` level but tolerated. Useful while
    /// bringing a model up, when the checkpoint carries extra tensors (optimizer
    /// state, EMA copies) that the inference module intentionally ignores.
    Lenient,
}

/// Loads weights from a `.pth` or `.safetensors` file into a Burn `module`.
///
/// The file format is chosen from the path extension (`pth`/`pt` → PyTorch,
/// `safetensors` → safetensors); anything else is rejected with [`Error::Config`].
/// `remap` renames source keys before matching (pass `KeyRemap::new()` for none).
///
/// After applying the record, unconsumed source keys are handled according to
/// `coverage`: [`Coverage::Strict`] returns [`Error::UnmappedKeys`], while
/// [`Coverage::Lenient`] only logs them.
///
/// # Errors
///
/// - [`Error::Config`] for an unrecognised extension or bad remap rule.
/// - [`Error::WeightLoad`] if the file cannot be read or a tensor fails to apply.
/// - [`Error::UnmappedKeys`] under [`Coverage::Strict`] when keys are left over.
pub fn load_weights<B, M>(
    module: &mut M,
    path: impl AsRef<Path>,
    remap: &KeyRemap,
    coverage: Coverage,
) -> Result<()>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    load_weights_ignoring::<B, M>(module, path, remap, coverage, &[])
}

/// Like [`load_weights`], but treats any *remapped* source key equal to one of
/// `ignore` as intentionally consumed, so it never counts toward the
/// [`Coverage::Strict`] unmapped-key check.
///
/// This is for checkpoint tensors that inference does not use — training-only
/// buffers such as a denoising / contrastive embedding — which a module tree
/// legitimately has no field for. Prefer routing every real weight through the
/// [`KeyRemap`]; reach for `ignore` only when there is genuinely no field to load
/// into, and document why at the call site.
///
/// The `ignore` entries are matched against keys *after* `remap` is applied (the
/// same keys `burn-store` reports as unused), so pass the post-remap name.
///
/// # Errors
///
/// - [`Error::Config`] for an unrecognised extension or bad remap rule.
/// - [`Error::WeightLoad`] if the file cannot be read or a tensor fails to apply.
/// - [`Error::UnmappedKeys`] under [`Coverage::Strict`] when keys other than the
///   ignored ones are left over.
pub fn load_weights_ignoring<B, M>(
    module: &mut M,
    path: impl AsRef<Path>,
    remap: &KeyRemap,
    coverage: Coverage,
    ignore: &[&str],
) -> Result<()>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    let remapper = remap.build()?;

    // `PytorchStore` / `SafetensorsStore` share the `apply_to` shape but not a
    // common object-safe supertype we can name here, so dispatch per format.
    let result = match ext.as_str() {
        "pth" | "pt" => {
            let mut store = PytorchStore::from_file(path)
                // Tolerate module fields the checkpoint does not provide; the
                // *reverse* direction (unused source keys) is what `Coverage`
                // guards, and that is read off `ApplyResult` below.
                .allow_partial(true);
            if !remap.is_empty() {
                store = store.remap(remapper);
            }
            store
                .apply_to(module)
                .map_err(|e| Error::WeightLoad(format!("{path:?}: {e:?}")))?
        }
        "safetensors" => {
            let mut store = SafetensorsStore::from_file(path).allow_partial(true);
            if !remap.is_empty() {
                store = store.remap(remapper);
            }
            store
                .apply_to(module)
                .map_err(|e| Error::WeightLoad(format!("{path:?}: {e:?}")))?
        }
        other => {
            return Err(Error::Config(format!(
                "unsupported weight file extension {other:?} for {path:?} (expected pth, pt, or safetensors)"
            )));
        }
    };

    // A hard apply error is never masked, regardless of coverage policy.
    if !result.errors.is_empty() {
        return Err(Error::WeightLoad(format!(
            "{path:?}: {} tensor(s) failed to apply: {:?}",
            result.errors.len(),
            result.errors,
        )));
    }

    // Drop intentionally-ignored keys before the coverage check so training-only
    // tensors with no inference field never trip `Coverage::Strict`.
    let unused: Vec<String> = result
        .unused
        .into_iter()
        .filter(|k| !ignore.contains(&k.as_str()))
        .collect();

    assert_all_keys_consumed(&unused, coverage)
}

/// Turns the `unused` list from a load into a [`Coverage`] verdict.
///
/// Split out so model crates that drive `burn-store` directly can reuse the exact
/// same policy check on their own [`burn_store::ApplyResult::unused`].
///
/// # Errors
///
/// Returns [`Error::UnmappedKeys`] under [`Coverage::Strict`] when `unused` is
/// non-empty. Under [`Coverage::Lenient`] it logs and returns `Ok(())`.
pub fn assert_all_keys_consumed(unused: &[String], coverage: Coverage) -> Result<()> {
    if unused.is_empty() {
        return Ok(());
    }
    match coverage {
        Coverage::Strict => Err(Error::UnmappedKeys {
            keys: unused.to_vec(),
        }),
        Coverage::Lenient => {
            tracing::warn!(
                count = unused.len(),
                keys = ?unused,
                "weight load left source keys unmapped (Coverage::Lenient)",
            );
            Ok(())
        }
    }
}

/// Uniform entry point for loading weights into a module.
///
/// A thin convenience over [`load_weights`] for the common case where a model
/// type wants a one-call method. Model crates can implement this (usually via the
/// blanket impl below) and call `module.load_weights(path, &remap, coverage)`.
pub trait LoadWeights<B: Backend>: ModuleSnapshot<B> + Sized {
    /// Loads weights from `path` into `self`. See [`load_weights`].
    fn load_weights(
        &mut self,
        path: impl AsRef<Path>,
        remap: &KeyRemap,
        coverage: Coverage,
    ) -> Result<()> {
        load_weights(self, path, remap, coverage)
    }
}

impl<B: Backend, M: ModuleSnapshot<B>> LoadWeights<B> for M {}
