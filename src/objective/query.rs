//! Query-group plumbing shared by the ranking objectives (`rank:*` and
//! `rank:xendcg`): metadata validation and per-query gradient slicing.

use super::{GradPair, check_label_domain, check_label_width};
use crate::data::MetaInfo;
use crate::error::{HessboostError, Result};
use rayon::prelude::*;

/// Query batches at least this many rows in total compute their gradients in
/// parallel (each query writes only its own rows).
pub(super) const PARALLEL_QUERY_ROWS: usize = 4096;

/// One label per row passing `invalid` (true rejects the label), and
/// non-empty query groups that cover the rows in order, each with one
/// constant weight (lengths are checked before any group is sliced).
pub(super) fn validate_query_info(info: &MetaInfo, invalid: impl Fn(f32) -> bool) -> Result<()> {
    info.check_layout()?;
    check_label_width(info, 1)?;
    check_label_domain(info, invalid)?;
    let Some(group) = info.group else {
        return Err(HessboostError::invalid_param(
            "group_sizes",
            "ranking dataset requires group information",
        ));
    };
    if !group.partitions(info.n_rows) || group.iter_ranges().any(|(start, end)| start == end) {
        return Err(HessboostError::invalid_param(
            "group_sizes",
            format!(
                "dataset has {} rows, but its query groups are not non-empty consecutive \
                 row ranges covering them",
                info.n_rows
            ),
        ));
    }
    if let Some(weights) = info.weights {
        for (start, end) in group.iter_ranges() {
            if weights[start..end]
                .iter()
                .any(|weight| *weight != weights[start])
            {
                return Err(HessboostError::invalid_param(
                    "weights",
                    "ranking dataset requires one constant weight per query group",
                ));
            }
        }
    }
    Ok(())
}

/// Calls `f(scratch, query, start, rows)` for every query range, `rows` being
/// that query's disjoint slice of `out` starting at row `start`.
///
/// The ranges tile the rows in order. Batches of at least
/// [`PARALLEL_QUERY_ROWS`] rows with more than one query run in parallel, each
/// worker with its own `init()` scratch; every query writes only its own rows,
/// so the result does not depend on the thread count.
pub(super) fn for_each_query<S>(
    ranges: &[(usize, usize)],
    out: &mut [GradPair],
    init: impl Fn() -> S + Sync + Send,
    f: impl Fn(&mut S, usize, usize, &mut [GradPair]) + Sync + Send,
) {
    let n_rows = out.len();
    let mut queries = Vec::with_capacity(ranges.len());
    let mut rest = out;
    let mut offset = 0;
    for &(start, end) in ranges {
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(start - offset);
        let (rows, tail) = tail.split_at_mut(end - start);
        rest = tail;
        offset = end;
        queries.push((start, rows));
    }
    let process = |scratch: &mut S, (query, (start, rows)): (usize, (usize, &mut [GradPair]))| {
        f(scratch, query, start, rows);
    };
    if n_rows >= PARALLEL_QUERY_ROWS && queries.len() > 1 && rayon::current_num_threads() > 1 {
        queries
            .into_par_iter()
            .enumerate()
            .for_each_init(&init, process);
    } else {
        let mut scratch = init();
        for item in queries.into_iter().enumerate() {
            process(&mut scratch, item);
        }
    }
}
