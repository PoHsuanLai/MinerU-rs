//! Diagnostic probe: what grid does our recovery build from the reference's own
//! cell polygons?
//!
//! Not a correctness gate — a measurement, and a deliberately narrow one. Feeding
//! the *reference's* polygons in means every upstream stage (segmentation,
//! morphology, line extraction) is held identical, so any divergence in the
//! reported row/column counts is recovery's alone. Our wired engine builds a
//! 32-cell table where the reference builds 128 from the same crop, and the two
//! produce near-identical polygons (34 vs 32), so this is where the difference
//! has to be.
//!
//! Dump the reference's polygons as one whitespace-separated `x0 y0 x1 y1 ...`
//! line per 4-point polygon, then:
//!
//! ```text
//! MINERU_PROBE_POLYS=/tmp/py_polys.txt \
//!   cargo test -p mineru-table --test recover_probe -- --ignored --nocapture
//! ```

use mineru_table::unet::recover::{get_benchmark_cols, get_benchmark_rows, get_rows, Poly};

/// Reads `x0 y0 x1 y1 x2 y2 x3 y3` per line into polygons.
fn read_polys(path: &str) -> Option<Vec<Poly>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Vec<f32> = line
            .split_whitespace()
            .filter_map(|t| t.parse::<f32>().ok())
            .collect();
        let [x0, y0, x1, y1, x2, y2, x3, y3] = v[..] else {
            eprintln!("skipping malformed line with {} floats", v.len());
            continue;
        };
        out.push([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]);
    }
    Some(out)
}

#[test]
#[ignore = "diagnostic; needs MINERU_PROBE_POLYS pointing at a reference polygon dump"]
fn grid_from_reference_polygons() {
    let Ok(path) = std::env::var("MINERU_PROBE_POLYS") else {
        eprintln!("set MINERU_PROBE_POLYS");
        return;
    };
    let Some(polys) = read_polys(&path) else {
        eprintln!("could not read {path}");
        return;
    };
    println!("polygons: {}", polys.len());

    // The reference's defaults (`TableRecover.__call__`).
    let rows = get_rows(&polys, 10.0);
    println!(
        "get_rows -> {} rows; sizes: {:?}",
        rows.len(),
        rows.iter().map(Vec::len).collect::<Vec<_>>()
    );

    let col = get_benchmark_cols(&rows, &polys, 15.0);
    println!("get_benchmark_cols -> col_nums: {}", col.col_nums);

    let row = get_benchmark_rows(&rows, &polys);
    println!("get_benchmark_rows -> row_nums: {}", row.row_nums);
    println!("=> logical grid: {} x {}", row.row_nums, col.col_nums);
}
