//! Line-recovery for wired tables (`TableRecover`).
//!
//! Port of `unet_table/table_recover.py`. Given the quadrilateral cell polygons
//! recovered from the UNet line-segmentation mask, this infers the logical table
//! grid: which physical cells belong to which row, the benchmark column x-starts,
//! and each cell's `[row_start, row_end, col_start, col_end]` span.
//!
//! Polygons follow the Python convention `poly[cell][corner] = (x, y)` with four
//! corners in counter-clockwise order: `0` top-left, `1` bottom-left, `2`
//! bottom-right, `3` top-right. This module consumes them as `[[f32; 2]; 4]`.

/// A cell quadrilateral: four `(x, y)` corners, CCW from top-left.
pub type Poly = [[f32; 2]; 4];

/// A cell's logical grid span `[row_start, row_end, col_start, col_end]`.
pub use crate::matching::LogicPoint;

/// L2 distance between two corner points.
fn l2(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

/// Groups cell indices into rows by comparing successive top-left y coordinates.
///
/// Port of `TableRecover.get_rows`.
pub fn get_rows(polygons: &[Poly], rows_thresh: f32) -> Vec<Vec<usize>> {
    let n = polygons.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![vec![0]];
    }
    let y: Vec<f32> = polygons.iter().map(|p| p[0][1]).collect();
    let minus: Vec<f32> = (1..n).map(|i| y[i] - y[i - 1]).collect();

    // Indices in `minus` where the row breaks.
    let mut split_idxs: Vec<usize> = minus
        .iter()
        .enumerate()
        .filter(|(_, &d)| d.abs() > rows_thresh)
        .map(|(i, _)| i)
        .collect();

    if split_idxs.is_empty() {
        return vec![(0..n).collect()];
    }
    if split_idxs.last() != Some(&minus.len()) {
        split_idxs.push(minus.len());
    }

    let mut result: Vec<Vec<usize>> = Vec::new();
    let mut start_idx = 0usize;
    for (row_num, &idx) in split_idxs.iter().enumerate() {
        if row_num != 0 {
            start_idx = split_idxs[row_num - 1] + 1;
        }
        result.push((start_idx..=idx).collect());
    }
    result
}

/// Benchmark column geometry produced by [`get_benchmark_cols`].
pub struct ColBenchmark {
    /// Sorted column start x-coordinates.
    pub longest_x_start: Vec<f32>,
    /// Width of each column (last is `max_x - last_start`).
    pub each_col_widths: Vec<f32>,
    /// Number of columns.
    pub col_nums: usize,
}

/// Builds the benchmark column x-starts from the longest row, then reconciles
/// every other row's cell boundaries into it.
///
/// Port of `TableRecover.get_benchmark_cols`.
pub fn get_benchmark_cols(rows: &[Vec<usize>], polygons: &[Poly], col_thresh: f32) -> ColBenchmark {
    let longest = rows
        .iter()
        .max_by_key(|r| r.len())
        .cloned()
        .unwrap_or_default();

    let mut longest_x_start: Vec<f32> = longest.iter().map(|&i| polygons[i][0][0]).collect();
    let longest_x_end: Vec<f32> = longest.iter().map(|&i| polygons[i][2][0]).collect();
    let mut min_x = longest_x_start.first().copied().unwrap_or(0.0);
    let mut max_x = longest_x_end.last().copied().unwrap_or(0.0);

    // Inserts a candidate boundary into the sorted column start list.
    fn update(
        col_x: &mut Vec<f32>,
        cur_v: f32,
        min_x: &mut f32,
        max_x: &mut f32,
        insert_last: bool,
        col_thresh: f32,
    ) {
        for i in 0..col_x.len() {
            let v = col_x[i];
            if cur_v - col_thresh <= v && v <= cur_v + col_thresh {
                break;
            }
            if cur_v < *min_x {
                col_x.insert(0, cur_v);
                *min_x = cur_v;
                break;
            }
            if cur_v > *max_x {
                if insert_last {
                    col_x.push(cur_v);
                }
                *max_x = cur_v;
                break;
            }
            if cur_v < v {
                col_x.insert(i, cur_v);
                break;
            }
        }
    }

    for row in rows {
        let starts: Vec<f32> = row.iter().map(|&i| polygons[i][0][0]).collect();
        let ends: Vec<f32> = row.iter().map(|&i| polygons[i][2][0]).collect();
        for (cur_start, cur_end) in starts.iter().zip(ends.iter()) {
            update(
                &mut longest_x_start,
                *cur_start,
                &mut min_x,
                &mut max_x,
                true,
                col_thresh,
            );
            update(
                &mut longest_x_start,
                *cur_end,
                &mut min_x,
                &mut max_x,
                false,
                col_thresh,
            );
        }
    }

    let mut each_col_widths: Vec<f32> = longest_x_start
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect();
    if let Some(&last) = longest_x_start.last() {
        each_col_widths.push(max_x - last);
    }
    let col_nums = longest_x_start.len();
    ColBenchmark {
        longest_x_start,
        each_col_widths,
        col_nums,
    }
}

/// Row-height benchmark produced by [`get_benchmark_rows`].
pub struct RowBenchmark {
    /// Height of each row (last is the max cell height in the bottom row).
    pub each_row_heights: Vec<f32>,
    /// Number of rows.
    pub row_nums: usize,
}

/// Estimates per-row heights from the leftmost cell of each row.
///
/// Port of `TableRecover.get_benchmark_rows`.
pub fn get_benchmark_rows(rows: &[Vec<usize>], polygons: &[Poly]) -> RowBenchmark {
    let benchmark_y: Vec<f32> = rows
        .iter()
        .filter_map(|r| r.first())
        .map(|&i| polygons[i][0][1])
        .collect();

    let mut each_row_heights: Vec<f32> = benchmark_y.windows(2).map(|w| w[1] - w[0]).collect();

    if let Some(bottom) = rows.last() {
        let max_height = bottom
            .iter()
            .map(|&i| l2(polygons[i][1], polygons[i][0]))
            .fold(0.0f32, f32::max);
        each_row_heights.push(max_height);
    }
    RowBenchmark {
        each_row_heights,
        row_nums: benchmark_y.len(),
    }
}

/// Computes per-cell logical spans by fitting each cell's width/height against
/// the column/row benchmarks.
///
/// Port of `TableRecover.get_merge_cells`. Returns one [`LogicPoint`] per input
/// polygon, in polygon order.
pub fn get_merge_cells(
    polygons: &[Poly],
    rows: &[Vec<usize>],
    row_nums: usize,
    col: &ColBenchmark,
    row: &RowBenchmark,
) -> Vec<LogicPoint> {
    const MERGE_THRESH: f32 = 10.0;
    let mut logic: Vec<Option<LogicPoint>> = vec![None; polygons.len()];

    for (cur_row, col_list) in rows.iter().enumerate() {
        // Track cumulative column consumption within this row.
        let mut consumed: Vec<usize> = Vec::new();
        for &one_col in col_list {
            let b = &polygons[one_col];
            let box_width = l2(b[3], b[0]);

            // Nearest benchmark column to the cell's left edge.
            let loc_col_idx = col
                .longest_x_start
                .iter()
                .enumerate()
                .min_by(|a, c| (a.1 - b[0][0]).abs().total_cmp(&(c.1 - b[0][0]).abs()))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let col_start = consumed.iter().sum::<usize>().max(loc_col_idx);

            // Column span fit.
            let mut col_span = col.col_nums - col_start;
            let mut i = col_start;
            while i < col.col_nums {
                let cum: f32 = col.each_col_widths[col_start..=i].iter().sum();
                if i == col_start && cum > box_width {
                    col_span = 1;
                    break;
                } else if (cum - box_width).abs() <= MERGE_THRESH {
                    col_span = i + 1 - col_start;
                    break;
                } else if cum > box_width {
                    let idx = if (cum - box_width).abs()
                        < (cum - col.each_col_widths[i] - box_width).abs()
                    {
                        i
                    } else {
                        i.saturating_sub(1)
                    };
                    col_span = idx + 1 - col_start;
                    break;
                }
                i += 1;
            }
            consumed.push(col_span);
            let col_end = col_span + col_start - 1;

            // Row span fit.
            let box_height = l2(b[1], b[0]);
            let row_start = cur_row;
            let mut row_span = row_nums - row_start;
            let mut j = row_start;
            while j < row_nums {
                let cum: f32 = row.each_row_heights[row_start..=j].iter().sum();
                if j == row_start && cum > box_height {
                    row_span = 1;
                    break;
                } else if (box_height - cum).abs() <= MERGE_THRESH {
                    row_span = j + 1 - row_start;
                    break;
                } else if cum > box_height {
                    let idx = if (cum - box_height).abs()
                        < (cum - row.each_row_heights[j] - box_height).abs()
                    {
                        j
                    } else {
                        j.saturating_sub(1)
                    };
                    row_span = idx + 1 - row_start;
                    break;
                }
                j += 1;
            }
            let row_end = row_span + row_start - 1;

            logic[one_col] = Some(LogicPoint {
                row_start,
                row_end,
                col_start,
                col_end,
            });
        }
    }

    logic
        .into_iter()
        .map(|lp| {
            lp.unwrap_or(LogicPoint {
                row_start: 0,
                row_end: 0,
                col_start: 0,
                col_end: 0,
            })
        })
        .collect()
}

/// Runs the full line-recovery: polygons → per-cell [`LogicPoint`]s.
///
/// Port of `TableRecover.__call__`.
pub fn recover(polygons: &[Poly], rows_thresh: f32, col_thresh: f32) -> Vec<LogicPoint> {
    if polygons.is_empty() {
        return Vec::new();
    }
    let rows = get_rows(polygons, rows_thresh);
    let col = get_benchmark_cols(&rows, polygons, col_thresh);
    let row = get_benchmark_rows(&rows, polygons);
    get_merge_cells(polygons, &rows, row.row_nums, &col, &row)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a CCW polygon from an axis-aligned rectangle.
    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Poly {
        // 0 top-left, 1 bottom-left, 2 bottom-right, 3 top-right.
        [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
    }

    #[test]
    fn single_cell_is_one_row() {
        let polys = vec![rect(0.0, 0.0, 10.0, 10.0)];
        let rows = get_rows(&polys, 10.0);
        assert_eq!(rows, vec![vec![0]]);
    }

    #[test]
    fn groups_two_rows_by_y() {
        // Row 0 at y=0, row 1 at y=100.
        let polys = vec![
            rect(0.0, 0.0, 10.0, 20.0),
            rect(10.0, 0.0, 20.0, 20.0),
            rect(0.0, 100.0, 10.0, 120.0),
            rect(10.0, 100.0, 20.0, 120.0),
        ];
        let rows = get_rows(&polys, 10.0);
        assert_eq!(rows, vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn recover_2x2_grid_gives_unit_spans() {
        let polys = vec![
            rect(0.0, 0.0, 10.0, 20.0),
            rect(10.0, 0.0, 20.0, 20.0),
            rect(0.0, 20.0, 10.0, 40.0),
            rect(10.0, 20.0, 20.0, 40.0),
        ];
        let logic = recover(&polys, 10.0, 5.0);
        assert_eq!(logic.len(), 4);
        assert_eq!(
            logic[0],
            LogicPoint {
                row_start: 0,
                row_end: 0,
                col_start: 0,
                col_end: 0
            }
        );
        assert_eq!(logic[1].col_start, 1);
        assert_eq!(logic[2].row_start, 1);
        assert_eq!(logic[3].row_start, 1);
        assert_eq!(logic[3].col_start, 1);
    }

    #[test]
    fn benchmark_cols_uses_longest_row() {
        // Row 0 has 3 cells, row 1 has 1 wide cell.
        let polys = vec![
            rect(0.0, 0.0, 10.0, 20.0),
            rect(10.0, 0.0, 20.0, 20.0),
            rect(20.0, 0.0, 30.0, 20.0),
            rect(0.0, 20.0, 30.0, 40.0),
        ];
        let rows = get_rows(&polys, 10.0);
        let col = get_benchmark_cols(&rows, &polys, 5.0);
        assert_eq!(col.col_nums, 3);
    }
}
