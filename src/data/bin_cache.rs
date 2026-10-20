//! Histogram indexes binned from a matrix's feature values, kept with the
//! values and shared by every matrix that shares them.
//!
//! A [`DMatrix`](super::DMatrix) and its copies with other metadata
//! (`with_labels`, `with_weights`, Python's `copy` with `set_info`) share
//! their feature values, and with them this cache of their latest binning
//! (its `max_bin`, sample weights, feature types and missing value): it is
//! computed once however many trainings, folds, or threads ask for it, and
//! freed with the values or once a request for another binning replaces it
//! (a training holding it keeps it until it finishes), so a matrix trained
//! under many `max_bin`s or weightings keeps one index, not one for each. A
//! hit is the very index a miss would build, so cached training is bit for
//! bit uncached training.
//!
//! The first request for a key builds its index; concurrent requests for
//! the key wait for it. A parallel build runs on a pool of its own (of the
//! caller's thread count) and publishes from there, so waiting cannot
//! deadlock: a training waiting for the index may be a job the builder's
//! own pool would otherwise run while it waits (two trainings of one
//! matrix started from one rayon pool), but the builder's pool never runs a
//! waiter's work. A serial build (a small matrix, or one thread) never
//! enters rayon, so it never picks up a waiter's work and runs on the
//! caller. A builder that cannot start a pool (a thread limit) abandons the
//! key and builds on its own thread, and every waiter then builds its own.
//!
//! A forked child inherits its parent's caches as they were at the fork,
//! possibly mid-build (an entry whose builder thread did not fork) or with a
//! lock held, and cannot tell; it bypasses them, binning as if uncached,
//! while the caches of matrices it builds itself work as usual.

use crate::data::ghist::GHistIndex;
use crate::data::meta::FeatureType;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// Everything besides the feature values a binning depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BinKey {
    max_bin: usize,
    /// The missing sentinel's bits, every NaN as one (they mark the same
    /// values missing).
    missing: u32,
    feature_types: Vec<FeatureType>,
    /// The sample weights' bits: they weight the quantile sketches.
    weights: Option<Vec<u32>>,
}

impl BinKey {
    pub(crate) fn new(
        max_bin: usize,
        missing: f32,
        feature_types: &[FeatureType],
        weights: Option<&[f32]>,
    ) -> Self {
        BinKey {
            max_bin,
            missing: if missing.is_nan() {
                f32::NAN.to_bits()
            } else {
                missing.to_bits()
            },
            feature_types: feature_types.to_vec(),
            weights: weights.map(|w| w.iter().map(|v| v.to_bits()).collect()),
        }
    }
}

/// The latest index built from one set of feature values, with any still
/// being built.
#[derive(Debug)]
pub(crate) struct BinCache {
    /// The process the cache was made in; any other one (a forked child)
    /// bypasses it.
    owner: u32,
    entries: Mutex<Vec<Arc<Entry>>>,
}

impl Default for BinCache {
    fn default() -> Self {
        BinCache {
            owner: std::process::id(),
            entries: Mutex::default(),
        }
    }
}

/// One key's index, built once.
#[derive(Debug)]
struct Entry {
    key: BinKey,
    state: Mutex<State>,
    done: Condvar,
}

#[derive(Debug)]
enum State {
    Building,
    Built(Arc<GHistIndex>),
    /// The builder gave up (its build unwound, or it built unshared):
    /// waiters claim the key again.
    Abandoned,
}

/// A lock whose holder cannot leave its data half-written (every critical
/// section here only swaps a whole value), so a panic elsewhere never
/// poisons it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Entry {
    fn publish(&self, index: Arc<GHistIndex>) {
        *lock(&self.state) = State::Built(index);
        self.done.notify_all();
    }

    fn abandon(&self) {
        let mut state = lock(&self.state);
        if matches!(*state, State::Building) {
            *state = State::Abandoned;
            self.done.notify_all();
        }
    }

    /// The built index, or `None` once the builder abandons the key.
    fn wait(&self) -> Option<Arc<GHistIndex>> {
        let mut state = lock(&self.state);
        loop {
            match &*state {
                State::Built(index) => return Some(Arc::clone(index)),
                State::Abandoned => return None,
                State::Building => {
                    state = self
                        .done
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
    }

    fn abandoned(&self) -> bool {
        matches!(*lock(&self.state), State::Abandoned)
    }

    fn building(&self) -> bool {
        matches!(*lock(&self.state), State::Building)
    }
}

/// Abandons its entry when dropped armed: the build unwound before
/// publishing.
struct AbandonOnUnwind<'a> {
    entry: &'a Entry,
    armed: bool,
}

impl Drop for AbandonOnUnwind<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.entry.abandon();
        }
    }
}

impl BinCache {
    /// The index for `key`: the cached one, or `build`'s (once per key, see
    /// the module docs). `parallel`: whether `build` uses rayon (then it
    /// runs on a pool of its own). Outside the process that made the cache,
    /// `build`'s, unshared.
    pub(crate) fn get_or_build(
        &self,
        key: &BinKey,
        parallel: bool,
        build: impl FnOnce() -> GHistIndex + Send,
    ) -> Arc<GHistIndex> {
        if self.owner != std::process::id() {
            return Arc::new(build());
        }
        let mut build = Some(build);
        loop {
            let (entry, claimed) = self.claim(key);
            if claimed {
                let build = build.take().expect("a request claims a key at most once");
                return publish(&entry, parallel, build);
            }
            if let Some(index) = entry.wait() {
                return index;
            }
        }
    }

    /// `key`'s entry, and whether this request claimed it (and must build
    /// it). Abandoned entries are dropped first, and a new key's entry
    /// replaces every built one: the cache keeps the latest binning (and
    /// those still being built, which requests may be waiting for).
    fn claim(&self, key: &BinKey) -> (Arc<Entry>, bool) {
        let mut entries = lock(&self.entries);
        entries.retain(|entry| !entry.abandoned());
        if let Some(entry) = entries.iter().find(|entry| entry.key == *key) {
            return (Arc::clone(entry), false);
        }
        entries.retain(|entry| entry.building());
        let entry = Arc::new(Entry {
            key: key.clone(),
            state: Mutex::new(State::Building),
            done: Condvar::new(),
        });
        entries.push(Arc::clone(&entry));
        (entry, true)
    }
}

/// Build a claimed entry's index and publish it: a parallel build on a pool
/// of its own and published from there, a serial one on this thread (see
/// the module docs). Without threads for that pool, abandon the key and
/// build on this thread.
fn publish(
    entry: &Entry,
    parallel: bool,
    build: impl FnOnce() -> GHistIndex + Send,
) -> Arc<GHistIndex> {
    if !parallel {
        return build_and_publish(entry, build);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rayon::current_num_threads())
        .build();
    if let Ok(pool) = pool {
        pool.install(|| build_and_publish(entry, build))
    } else {
        entry.abandon();
        Arc::new(build())
    }
}

/// Build `entry`'s index and publish it, abandoning the key if the build
/// unwinds.
fn build_and_publish(entry: &Entry, build: impl FnOnce() -> GHistIndex) -> Arc<GHistIndex> {
    let mut guard = AbandonOnUnwind { entry, armed: true };
    let index = Arc::new(build());
    entry.publish(Arc::clone(&index));
    guard.armed = false;
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DMatrix;
    use crate::data::quantile::HistCuts;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn index_of(data: &DMatrix) -> GHistIndex {
        GHistIndex::from_dmatrix(data, HistCuts::from_dmatrix(data, 16))
    }

    fn key(max_bin: usize, weights: Option<&[f32]>) -> BinKey {
        BinKey::new(max_bin, f32::NAN, &[FeatureType::Numerical; 2], weights)
    }

    /// Concurrent requests for one key, from trainings running as jobs of
    /// one rayon pool (a builder waiting on its own parallel work would
    /// steal waiters), build once and all get that index; another key, built
    /// serially, builds its own.
    #[test]
    fn concurrent_requests_build_once_without_deadlock() {
        use rayon::prelude::*;
        let data = DMatrix::from_dense(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 3, 2).unwrap();
        let cache = BinCache::default();
        let builds = AtomicUsize::new(0);
        let build = || {
            builds.fetch_add(1, Ordering::Relaxed);
            // Parallel work, so the build waits on rayon joins.
            let spin: u64 = (0..100_000u64).into_par_iter().map(|i| i % 7).sum();
            assert!(spin > 0);
            index_of(&data)
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let indexes: Vec<Arc<GHistIndex>> = pool.install(|| {
            (0..64)
                .into_par_iter()
                .map(|_| cache.get_or_build(&key(16, None), true, build))
                .collect()
        });
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert!(indexes.iter().all(|index| Arc::ptr_eq(index, &indexes[0])));
        let weighted =
            cache.get_or_build(&key(16, Some(&[1.0, 2.0, 1.0])), false, || index_of(&data));
        assert!(!Arc::ptr_eq(&weighted, &indexes[0]));
    }

    /// A new binning replaces the built ones: the cache keeps only the
    /// latest, so ever new keys (a `max_bin` search, say) do not keep every
    /// index alive, and a replaced key is built again.
    #[test]
    fn a_new_binning_replaces_the_built_ones() {
        let data = DMatrix::from_dense(&[0.0, 1.0, 2.0, 3.0], 2, 2).unwrap();
        let cache = BinCache::default();
        let builds = AtomicUsize::new(0);
        let build = || {
            builds.fetch_add(1, Ordering::Relaxed);
            index_of(&data)
        };
        let first = cache.get_or_build(&key(8, None), false, build);
        let second = cache.get_or_build(&key(16, None), false, build);
        assert_eq!(Arc::strong_count(&first), 1, "the cache let the first go");
        let hit = cache.get_or_build(&key(16, None), false, build);
        assert!(Arc::ptr_eq(&second, &hit));
        let again = cache.get_or_build(&key(8, None), false, build);
        assert!(!Arc::ptr_eq(&first, &again));
        assert_eq!(builds.load(Ordering::Relaxed), 3);
    }

    /// A build that unwinds abandons its key: the panic reaches its caller,
    /// and the next request builds the index.
    #[test]
    fn an_unwound_build_leaves_the_key_to_the_next_request() {
        let data = DMatrix::from_dense(&[0.0, 1.0, 2.0, 3.0], 2, 2).unwrap();
        let cache = BinCache::default();
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.get_or_build(&key(8, None), true, || panic!("build failed"))
        }));
        assert!(failed.is_err());
        let index = cache.get_or_build(&key(8, None), false, || index_of(&data));
        assert_eq!(index.n_rows(), 2);
    }

    /// A cache inherited from another process (a fork) is bypassed: an
    /// entry its parent was still building, whose builder thread the child
    /// lacks, does not hold the child's request up.
    #[test]
    fn an_inherited_cache_is_bypassed() {
        let data = DMatrix::from_dense(&[0.0, 1.0, 2.0, 3.0], 2, 2).unwrap();
        let inherited = BinCache {
            owner: std::process::id().wrapping_add(1),
            entries: Mutex::new(vec![Arc::new(Entry {
                key: key(8, None),
                state: Mutex::new(State::Building),
                done: Condvar::new(),
            })]),
        };
        let index = inherited.get_or_build(&key(8, None), false, || index_of(&data));
        assert_eq!(index.n_rows(), 2);
    }
}
