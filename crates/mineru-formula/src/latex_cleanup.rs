// ============================================================================
// FIDELITY NOTES — Python patterns that could NOT be replicated with a bare
// `regex::Regex` and how they are approximated here.
//
// Rust's `regex` crate (v1) has no lookahead / lookbehind / backreferences, so
// every Python pattern that relied on those is re-expressed. All of the
// approximations below are behaviourally EXACT for the inputs the original
// regex accepted; none are lossy heuristics.
//
//   * `LEFT_COUNT_PATTERN  = r'\\left(?![a-zA-Z])'`
//     `RIGHT_COUNT_PATTERN = r'\\right(?![a-zA-Z])'`
//       The negative-lookahead `(?![a-zA-Z])` is emulated by matching the base
//       token plus one trailing character with an optional group and rejecting
//       matches whose trailing char is ASCII-alphabetic. Implemented as a
//       hand-rolled counter (`count_left` / `count_right`) so end-of-string
//       (no trailing char) is also counted, exactly as the lookahead does.
//
//   * `QQUAD_PATTERN = r'\\qquad(?!\s)'`
//       Same negative-lookahead shape. Emulated with a `replace_all` closure
//       driven by the pattern `\\qquad(\s?)`: a match whose captured trailing
//       char is whitespace is left untouched (the lookahead would have failed),
//       otherwise the trailing char is preserved and a space is inserted —
//       identical to the Python substitution `\\qquad `.
//
//   * `process_latex` — pattern `r'\\(.)'` with a `match.start()`-based peek at
//     `input_string[pos]` to test whether the char *after* the captured letter
//     is also a letter. Regex closures in Rust don't expose the match offset
//     conveniently across the whole string, so this is reimplemented as a
//     single left-to-right `char` scanner that reproduces the Python branch
//     logic verbatim (special char / two-letter command / else insert space).
//     `.` in Python's `re` (no DOTALL) does not match `\n`; the scanner
//     replicates that by treating `\` followed by `\n` as "no captured char",
//     leaving it verbatim — matching `re.sub`'s behaviour of not substituting.
//
//   * `remove_up_commands` — pattern `\\up([a-zA-Z]+)` with a closure. Ported
//     directly with a `regex` closure; exact.
//
//   * `fix_latex_left_right` delimiter fix used `LEFT_PATTERN = r'(\\left)(\S*)'`
//     (greedy `\S*`). `\S` (any non-whitespace, Unicode) is supported by the
//     `regex` crate directly, so `LEFT_PATTERN` / `RIGHT_PATTERN` are exact.
//
// Everything else (`fix_unbalanced_braces`, `fix_left_right_pairs`,
// `find_group_end`, `is_escaped`, `fix_latex_environments`,
// `remove_unsupported_commands`, `REPLACEMENTS_PATTERNS`, trailing-backslash
// strip) is a direct structural port and is behaviourally exact.
// ============================================================================

//! Port of MinerU's `latex_rm_whitespace` LaTeX-cleanup pass.
//!
//! This module is a faithful Rust translation of
//! `mineru/model/mfr/utils.py::latex_rm_whitespace` and its helper functions.
//! It normalises the raw LaTeX emitted by the formula-recognition model:
//! balancing braces, repairing `\left` / `\right` pairs, closing math
//! environments, dropping unsupported commands, applying a fixed replacement
//! table, spacing backslash commands, and stripping trailing backslashes.
//!
//! The public entry point is [`latex_rm_whitespace`]. All helpers mirror the
//! Python originals one-to-one and are kept private.

use regex::Regex;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Compiled regexes (valid literals authored here → `.expect` is acceptable
// inside the LazyLock initializer only).
// ---------------------------------------------------------------------------

static LEFT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\\left)(\S*)").expect("valid LEFT_PATTERN"));
static RIGHT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\\right)(\S*)").expect("valid RIGHT_PATTERN"));

static UP_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\up([a-zA-Z]+)").expect("valid UP_PATTERN"));

static COMMANDS_TO_REMOVE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\\(?:lefteqn|boldmath|ensuremath|centering|textsubscript|sides|textsl|textcent|emph|protect|null)")
        .expect("valid COMMANDS_TO_REMOVE_PATTERN")
});

// `\qquad` followed by an optional single character; used to emulate the
// `(?!\s)` negative lookahead in a `replace_all` closure.
static QQUAD_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\\qquad(\s?)").expect("valid QQUAD_PATTERN"));

/// Ordered replacement table, mirroring Python's `REPLACEMENTS_PATTERNS`.
///
/// Python 3.7+ preserves `dict` insertion order, and `latex_rm_whitespace`
/// iterates the dict in that order, so a `Vec` of `(pattern, replacement)`
/// keeps the substitutions in the exact same sequence.
static REPLACEMENTS_PATTERNS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    let raw: [(&str, &str); 15] = [
        (r"\\underbar", r"\underline"),
        (r"\\Bar", r"\hat"),
        (r"\\Hat", r"\hat"),
        (r"\\Tilde", r"\tilde"),
        (r"\\slash", r"/"),
        (r"\\textperthousand", "‰"),
        (r"\\sun", "☉"),
        (r"\\textunderscore", r"\_"),
        (r"\\fint", "⨏"),
        (r"\\up ", r"\ "),
        (r"\\vline = ", r"\models "),
        (r"\\vDash ", r"\models "),
        (r"\\sq \\sqcup ", r"\square "),
        (r"\\copyright", "©"),
        (r"\\Dot", r"\dot"),
    ];
    raw.into_iter()
        .map(|(p, r)| (Regex::new(p).expect("valid REPLACEMENTS pattern"), r))
        .collect()
});

// Environments known to KaTeX/MathJax, mirroring Python's `ENV_TYPES`.
const ENV_TYPES: [&str; 12] = [
    "array", "matrix", "pmatrix", "bmatrix", "vmatrix", "Bmatrix", "Vmatrix", "cases", "aligned",
    "gathered", "align", "align*",
];

// Whitelisted delimiters for `\left` / `\right`, mirroring `valid_delims_list`
// (21 entries, in the same order as the Python source).
const VALID_DELIMS_LIST: [&str; 21] = [
    "(", ")", "[", "]", "{", "}", "/", "|", r"\{", r"\}", r"\lceil", r"\rceil", r"\lfloor",
    r"\rfloor", r"\backslash", r"\uparrow", r"\downarrow", r"\Uparrow", r"\Downarrow", r"\|", r"\.",
];

// ---------------------------------------------------------------------------
// Escape helpers
// ---------------------------------------------------------------------------

/// Check whether the char at `pos` (index into `chars`) is escaped, i.e.
/// preceded by an odd number of backslashes. Mirrors Python `is_escaped`.
fn is_escaped(chars: &[char], pos: usize) -> bool {
    let mut backslash_count = 0usize;
    let mut j = pos as isize - 1;
    while j >= 0 && chars[j as usize] == '\\' {
        backslash_count += 1;
        j -= 1;
    }
    backslash_count % 2 == 1
}

// ---------------------------------------------------------------------------
// fix_unbalanced_braces
// ---------------------------------------------------------------------------

/// Detect whether the braces in a LaTeX formula are balanced and delete any
/// braces that cannot be paired. Mirrors Python `fix_unbalanced_braces`.
fn fix_unbalanced_braces(latex_formula: &str) -> String {
    let chars: Vec<char> = latex_formula.chars().collect();
    let mut stack: Vec<usize> = Vec::new();
    let mut unmatched: std::collections::HashSet<usize> = std::collections::HashSet::new();

    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '{' || c == '}' {
            // Count preceding consecutive backslashes.
            let mut backslash_count = 0usize;
            let mut j = i as isize - 1;
            while j >= 0 && chars[j as usize] == '\\' {
                backslash_count += 1;
                j -= 1;
            }

            // Odd number of backslashes → escaped brace, does not participate.
            if backslash_count % 2 == 1 {
                i += 1;
                continue;
            }

            if c == '{' {
                stack.push(i);
            } else {
                // c == '}'
                if stack.pop().is_none() {
                    unmatched.insert(i);
                }
            }
        }
        i += 1;
    }

    // All still-open left braces are unmatched too.
    for idx in stack {
        unmatched.insert(idx);
    }

    chars
        .iter()
        .enumerate()
        .filter(|(i, _)| !unmatched.contains(i))
        .map(|(_, c)| *c)
        .collect()
}

// ---------------------------------------------------------------------------
// fix_latex_left_right (+ fix_left_right_pairs, find_group_end)
// ---------------------------------------------------------------------------

/// Count `\left` occurrences that are NOT followed by an ASCII letter,
/// emulating `LEFT_COUNT_PATTERN = r'\\left(?![a-zA-Z])'`.
fn count_left(chars: &[char]) -> usize {
    count_token_not_followed_by_letter(chars, &['\\', 'l', 'e', 'f', 't'])
}

/// Count `\right` occurrences that are NOT followed by an ASCII letter,
/// emulating `RIGHT_COUNT_PATTERN = r'\\right(?![a-zA-Z])'`.
fn count_right(chars: &[char]) -> usize {
    count_token_not_followed_by_letter(chars, &['\\', 'r', 'i', 'g', 'h', 't'])
}

/// Non-overlapping count of `token` occurrences where the char immediately
/// after the token is not an ASCII letter (or the token ends the string).
/// This reproduces `re.findall` of `TOKEN(?![a-zA-Z])`.
fn count_token_not_followed_by_letter(chars: &[char], token: &[char]) -> usize {
    let n = chars.len();
    let m = token.len();
    let mut count = 0usize;
    let mut i = 0usize;
    while i + m <= n {
        if &chars[i..i + m] == token {
            let ok = match chars.get(i + m) {
                Some(next) => !next.is_ascii_alphabetic(),
                None => true,
            };
            if ok {
                count += 1;
                i += m; // non-overlapping advance past the token
                continue;
            }
        }
        i += 1;
    }
    count
}

/// Repair `\left` / `\right` commands, mirroring Python `fix_latex_left_right`.
///
/// 1. Ensure each is followed by a valid delimiter (else append `.`).
/// 2. Balance the counts: if equal, run [`fix_left_right_pairs`]; otherwise
///    strip all `\left`/`\right` (with an optional trailing `.`).
fn fix_latex_left_right(s: &str) -> String {
    // Step 1: delimiter fix (fix_delimiter=True in Python).
    let fix_delim = |caps: &regex::Captures| -> String {
        let cmd = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let rest = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        if rest.is_empty() || !VALID_DELIMS_LIST.contains(&rest) {
            format!("{cmd}.")
        } else {
            caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
        }
    };

    let s = LEFT_PATTERN.replace_all(s, &fix_delim).into_owned();
    let s = RIGHT_PATTERN.replace_all(&s, &fix_delim).into_owned();

    // Step 2: balance counts.
    let chars: Vec<char> = s.chars().collect();
    let left_count = count_left(&chars);
    let right_count = count_right(&chars);

    if left_count == right_count {
        fix_left_right_pairs(&s)
    } else {
        // Remove all \left / \right (each with an optional trailing '.').
        remove_left_right(&chars)
    }
}

/// Emulate `LEFT_RIGHT_REMOVE_PATTERN = r'\\left\.?|\\right\.?'` → replace with "".
fn remove_left_right(chars: &[char]) -> String {
    let left: [char; 5] = ['\\', 'l', 'e', 'f', 't'];
    let right: [char; 6] = ['\\', 'r', 'i', 'g', 'h', 't'];
    let n = chars.len();
    let mut out = String::new();
    let mut i = 0usize;
    while i < n {
        if i + left.len() <= n && chars[i..i + left.len()] == left {
            i += left.len();
            if chars.get(i) == Some(&'.') {
                i += 1;
            }
            continue;
        }
        if i + right.len() <= n && chars[i..i + right.len()] == right {
            i += right.len();
            if chars.get(i) == Some(&'.') {
                i += 1;
            }
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Detect and repair the case where a `\left` and its `\right` do not sit at
/// the same brace-nesting depth, moving the `\right` to the end of the
/// `\left`'s brace group. Mirrors Python `fix_left_right_pairs`.
fn fix_left_right_pairs(latex_formula: &str) -> String {
    let chars: Vec<char> = latex_formula.chars().collect();
    let n = chars.len();

    let mut brace_stack: Vec<usize> = Vec::new();
    // (position, depth, delimiter)
    let mut left_stack: Vec<(usize, usize, char)> = Vec::new();
    // (start, end, target)
    let mut adjustments: Vec<(usize, usize, usize)> = Vec::new();

    let left_tok: [char; 5] = ['\\', 'l', 'e', 'f', 't'];
    let right_tok: [char; 6] = ['\\', 'r', 'i', 'g', 'h', 't'];

    let mut i = 0usize;
    while i < n {
        // Skip escaped characters (odd run of backslashes immediately before i).
        if i > 0 && chars[i - 1] == '\\' {
            let mut backslash_count = 0usize;
            let mut j = i as isize - 1;
            while j >= 0 && chars[j as usize] == '\\' {
                backslash_count += 1;
                j -= 1;
            }
            if backslash_count % 2 == 1 {
                i += 1;
                continue;
            }
        }

        // Detect \left (Python guard: i + 5 < len, so char at i+5 must exist).
        if i + 5 < n && chars[i..i + 5] == left_tok {
            let delimiter = chars[i + 5];
            left_stack.push((i, brace_stack.len(), delimiter));
            i += 6;
            continue;
        }
        // Detect \right (Python guard: i + 6 < len, so char at i+6 must exist).
        else if i + 6 < n && chars[i..i + 6] == right_tok {
            let _delimiter = chars[i + 6];
            if let Some((left_pos, left_depth, _left_delim)) = left_stack.pop() {
                if left_depth != brace_stack.len() {
                    if let Some(target_pos) = find_group_end(&chars, left_pos, left_depth) {
                        adjustments.push((i, i + 7, target_pos));
                    }
                }
            }
            i += 7;
            continue;
        }

        // Handle braces.
        if chars[i] == '{' {
            brace_stack.push(i);
        } else if chars[i] == '}' && !brace_stack.is_empty() {
            brace_stack.pop();
        }

        i += 1;
    }

    if adjustments.is_empty() {
        return latex_formula.to_string();
    }

    // Apply adjustments back-to-front (by start descending) to keep indices valid.
    let mut result: Vec<char> = chars;
    adjustments.sort_by_key(|adj| std::cmp::Reverse(adj.0));

    for (start, end, target) in adjustments {
        // Clamp to current bounds defensively (Python relies on prior indices).
        if start > result.len() || end > result.len() || start > end {
            continue;
        }
        let right_part: Vec<char> = result[start..end].to_vec();
        result.drain(start..end);
        let insert_at = target.min(result.len());
        // Insert the extracted \right segment at the target position.
        for (k, c) in right_part.into_iter().enumerate() {
            result.insert(insert_at + k, c);
        }
    }

    result.into_iter().collect()
}

/// Find the end position (index of the closing `}`) of the brace group at the
/// given nesting depth, starting at `pos`. Mirrors Python `find_group_end`.
fn find_group_end(text: &[char], pos: usize, depth: usize) -> Option<usize> {
    let mut current_depth = depth as isize;
    let target = depth as isize;
    let mut i = pos;
    while i < text.len() {
        if text[i] == '{' && (i == 0 || !is_escaped(text, i)) {
            current_depth += 1;
        } else if text[i] == '}' && (i == 0 || !is_escaped(text, i)) {
            current_depth -= 1;
            if current_depth < target {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// fix_latex_environments
// ---------------------------------------------------------------------------

/// Ensure `\begin{env}` / `\end{env}` counts match, prepending missing
/// `\begin` or appending missing `\end`. Mirrors Python `fix_latex_environments`.
fn fix_latex_environments(mut s: String) -> String {
    for env in ENV_TYPES {
        // Compile per-env patterns on the fly; the env names are known-good
        // literals so escaping the `*` in `align*` keeps them regex-valid.
        let env_escaped = regex::escape(env);
        let begin_re = match Regex::new(&format!(r"\\begin\{{{env_escaped}\}}")) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let end_re = match Regex::new(&format!(r"\\end\{{{env_escaped}\}}")) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let begin_count = begin_re.find_iter(&s).count();
        let end_count = end_re.find_iter(&s).count();

        if begin_count != end_count {
            if end_count > begin_count {
                // Extract an existing format argument, if any.
                let format_re =
                    match Regex::new(&format!(r"\\begin\{{{env_escaped}\}}\{{([^}}]*)\}}")) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                let default_format = if env == "array" { "{c}" } else { "" };
                let format_str = match format_re.captures(&s) {
                    Some(caps) => {
                        let inner = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                        format!("{{{inner}}}")
                    }
                    None => default_format.to_string(),
                };

                let missing_count = end_count - begin_count;
                let begin_command = format!("\\begin{{{env}}}{format_str} ");
                s = format!("{}{}", begin_command.repeat(missing_count), s);
            } else {
                let missing_count = begin_count - end_count;
                let end_command = format!(" \\end{{{env}}}");
                s = format!("{}{}", s, end_command.repeat(missing_count));
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------
// remove_up_commands / remove_unsupported_commands
// ---------------------------------------------------------------------------

/// Remove unnecessary `\up...` commands. Mirrors Python `remove_up_commands`:
/// `\uparrow`, `\updownarrow`(via "downarrow"), `\uplus`, `\upsilon` are kept,
/// everything else `\upFOO` becomes `\FOO`.
fn remove_up_commands(s: &str) -> String {
    UP_PATTERN
        .replace_all(s, |caps: &regex::Captures| {
            let g1 = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            if matches!(g1, "arrow" | "downarrow" | "lus" | "silon") {
                caps.get(0).map(|m| m.as_str()).unwrap_or("").to_string()
            } else {
                format!("\\{g1}")
            }
        })
        .into_owned()
}

/// Remove unsupported LaTeX commands. Mirrors Python `remove_unsupported_commands`.
fn remove_unsupported_commands(s: &str) -> String {
    COMMANDS_TO_REMOVE_PATTERN.replace_all(s, "").into_owned()
}

// ---------------------------------------------------------------------------
// process_latex
// ---------------------------------------------------------------------------

/// Process backslashes in LaTeX:
/// 1. `\` followed by a special char (`#$%&~_^|\{}` or whitespace) → unchanged.
/// 2. `\` followed by two ASCII letters → unchanged.
/// 3. otherwise → insert a space after the backslash.
///
/// Mirrors Python `process_latex` (regex `\\(.)` + closure). Reimplemented as a
/// single left-to-right scanner because the Python closure peeks at an absolute
/// offset into the source string. Python's `.` (no DOTALL) does not match a
/// newline, so `\` immediately followed by `\n` is left verbatim.
fn process_latex(input_string: &str) -> String {
    let chars: Vec<char> = input_string.chars().collect();
    let n = chars.len();
    // Special chars: "#$%&~_^|\\{} \t\n\r\v\f" — note `\v`(0x0B) and `\f`(0x0C).
    const SPECIAL: &[char] = &[
        '#', '$', '%', '&', '~', '_', '^', '|', '\\', '{', '}', ' ', '\t', '\n', '\r', '\u{000B}',
        '\u{000C}',
    ];

    let mut out = String::with_capacity(input_string.len());
    let mut i = 0usize;
    while i < n {
        if chars[i] == '\\' {
            // The regex `\\(.)` requires a captured char after the backslash,
            // and `.` does not match '\n'.
            match chars.get(i + 1) {
                Some(&next_char) if next_char != '\n' => {
                    if SPECIAL.contains(&next_char) {
                        // Rule 1: keep `\<special>` unchanged.
                        out.push('\\');
                        out.push(next_char);
                    } else if next_char.is_ascii_alphabetic() {
                        // Rule 2: check the char AFTER the captured letter.
                        let following_is_letter = matches!(
                            chars.get(i + 2),
                            Some(c) if c.is_ascii_alphabetic()
                        );
                        if following_is_letter {
                            out.push('\\');
                            out.push(next_char);
                        } else {
                            // Rule 3: insert a space.
                            out.push('\\');
                            out.push(' ');
                            out.push(next_char);
                        }
                    } else {
                        // Rule 3: insert a space.
                        out.push('\\');
                        out.push(' ');
                        out.push(next_char);
                    }
                    i += 2;
                    continue;
                }
                _ => {
                    // No captured char (end of string, or '\n' which `.` rejects)
                    // → the regex does not match; leave the backslash verbatim.
                    out.push('\\');
                    i += 1;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Remove unnecessary whitespace and normalise raw LaTeX, mirroring Python
/// `latex_rm_whitespace`.
///
/// The order of operations matches the Python source exactly:
/// 1. [`fix_unbalanced_braces`]
/// 2. [`fix_latex_left_right`]
/// 3. [`fix_latex_environments`]
/// 4. [`remove_up_commands`]
/// 5. [`remove_unsupported_commands`]
/// 6. apply the `REPLACEMENTS_PATTERNS` table in order
/// 7. [`process_latex`] (backslash / space handling)
/// 8. `\qquad` trailing-space insertion (`QQUAD_PATTERN`)
/// 9. strip all trailing backslashes
pub fn latex_rm_whitespace(s: &str) -> String {
    let mut s = fix_unbalanced_braces(s);
    s = fix_latex_left_right(&s);
    s = fix_latex_environments(s);

    s = remove_up_commands(&s);
    s = remove_unsupported_commands(&s);

    // Apply all replacements in insertion order.
    for (pattern, replacement) in REPLACEMENTS_PATTERNS.iter() {
        s = pattern.replace_all(&s, *replacement).into_owned();
    }

    // Handle backslashes and spaces.
    s = process_latex(&s);

    // Ensure a space after `\qquad` (emulate `\\qquad(?!\s)` → `\\qquad `).
    s = QQUAD_PATTERN
        .replace_all(&s, |caps: &regex::Captures| {
            let trailing = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            if trailing.is_empty() {
                // End of string / no trailing char: lookahead succeeds → add space.
                "\\qquad ".to_string()
            } else {
                // Captured a whitespace char → lookahead `(?!\s)` would have
                // FAILED, so the original text must be preserved unchanged.
                format!("\\qquad{trailing}")
            }
        })
        .into_owned();

    // Strip trailing backslashes.
    while s.ends_with('\\') {
        s.pop();
    }

    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_trailing_backslashes() {
        // Trailing backslashes are removed; `\\` at the end collapses fully.
        assert_eq!(latex_rm_whitespace(r"x + y\\"), "x + y");
        assert_eq!(latex_rm_whitespace(r"abc\"), "abc");
    }

    #[test]
    fn applies_replacement_pattern() {
        // `\underbar` → `\underline` via REPLACEMENTS_PATTERNS.
        // process_latex keeps `\u...` intact because it's a multi-letter command.
        let out = latex_rm_whitespace(r"\underbar{x}");
        assert!(
            out.contains(r"\underline"),
            "expected \\underline in output, got: {out}"
        );
    }

    #[test]
    fn qquad_gets_trailing_space() {
        // `\qquad` not already followed by whitespace gains a space.
        let out = latex_rm_whitespace(r"a\qquad{b}");
        assert!(
            out.contains(r"\qquad "),
            "expected '\\qquad ' in output, got: {out}"
        );
        // Already-spaced `\qquad ` must not gain a second space.
        let out2 = latex_rm_whitespace(r"a\qquad b");
        assert!(
            !out2.contains(r"\qquad  "),
            "did not expect double space, got: {out2}"
        );
    }

    #[test]
    fn removes_unmatched_braces() {
        // A stray closing brace with no opener is deleted.
        assert_eq!(fix_unbalanced_braces("a}b"), "ab");
        // A stray opening brace with no closer is deleted.
        assert_eq!(fix_unbalanced_braces("a{b"), "ab");
        // Balanced braces are preserved.
        assert_eq!(fix_unbalanced_braces("a{b}c"), "a{b}c");
        // Escaped braces are ignored by the matcher.
        assert_eq!(fix_unbalanced_braces(r"a\{b"), r"a\{b");
    }

    #[test]
    fn balances_left_right_by_stripping_when_unequal() {
        // One \left, no \right → counts differ → all \left/\right removed.
        let out = latex_rm_whitespace(r"\left( x");
        assert!(
            !out.contains(r"\left"),
            "expected \\left removed, got: {out}"
        );
    }
}
