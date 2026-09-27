//! The flat category pool a [`RegTree`](crate::tree::RegTree)'s categorical
//! splits index, as the XGBoost and LightGBM importers assemble it. Each
//! format decodes its own wire layout (segments, bit words) into per-split
//! category ids; the pool appends them and records each split's range.

use crate::tree::Node;

/// Why a split's categories cannot join the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PoolError {
    /// The split has no categories.
    Empty,
    /// The pool would outgrow the `u32` ranges nodes store.
    TooMany,
}

/// Category ids of a tree's categorical splits, concatenated in push order.
#[derive(Debug, Default)]
pub(crate) struct CategoryPool {
    ids: Vec<u32>,
}

impl CategoryPool {
    /// An empty pool with room for `capacity` ids.
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        CategoryPool {
            ids: Vec::with_capacity(capacity),
        }
    }

    /// Append `ids` as `node`'s category set and set its `cat_begin` /
    /// `cat_end`. On error the pool and `node` are unchanged.
    pub(crate) fn push_split(
        &mut self,
        node: &mut Node,
        ids: impl IntoIterator<Item = u32>,
    ) -> Result<(), PoolError> {
        let begin = self.ids.len();
        self.ids.extend(ids);
        let end = self.ids.len();
        let range = u32::try_from(begin).ok().zip(u32::try_from(end).ok());
        let error = match range {
            _ if end == begin => PoolError::Empty,
            Some((begin, end)) => {
                node.cat_begin = begin;
                node.cat_end = end;
                return Ok(());
            }
            None => PoolError::TooMany,
        };
        self.ids.truncate(begin);
        Err(error)
    }

    /// The pool, for [`RegTree::from_parts`](crate::tree::RegTree::from_parts).
    pub(crate) fn finish(self) -> Vec<u32> {
        self.ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_get_consecutive_ranges_and_failures_leave_no_trace() {
        let mut pool = CategoryPool::default();
        let mut a = Node::leaf(0.0, 0.0);
        let mut b = Node::leaf(0.0, 0.0);
        pool.push_split(&mut a, [3, 1]).unwrap();
        assert_eq!(pool.push_split(&mut b, []), Err(PoolError::Empty));
        assert_eq!((b.cat_begin, b.cat_end), (0, 0));
        pool.push_split(&mut b, [7]).unwrap();
        assert_eq!(
            (a.cat_begin, a.cat_end, b.cat_begin, b.cat_end),
            (0, 2, 2, 3)
        );
        assert_eq!(pool.finish(), [3, 1, 7]);
    }
}
