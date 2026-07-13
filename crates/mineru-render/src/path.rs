//! Image-path joining.
//!
//! Mirrors Python's `f"{img_buket_path}/{image_path}"`: a plain `/`-join with
//! trailing/leading separators trimmed so the result never doubles or drops the
//! separator. Kept string-based (not [`std::path::PathBuf`]) because these paths
//! are URL-ish references embedded in Markdown and JSON, not filesystem lookups.

/// Joins an image directory and a relative image reference with a single `/`.
///
/// An empty `dir` yields the bare `image`; an empty `image` yields the bare
/// `dir`.
pub(crate) fn join_image(dir: &str, image: &str) -> String {
    let dir = dir.trim_end_matches('/');
    let image = image.trim_start_matches('/');
    match (dir.is_empty(), image.is_empty()) {
        (true, _) => image.to_owned(),
        (_, true) => dir.to_owned(),
        _ => format!("{dir}/{image}"),
    }
}
