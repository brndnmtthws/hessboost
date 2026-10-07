//! The extension's process-local rayon pool, rebuilt in a forked child.
//!
//! Crate code parallelizes on the current rayon pool. Rayon's global pool
//! does not survive `os.fork()`: the child has none of its threads, so its
//! first parallel job would wait forever. Every native entry point that may
//! run rayon work therefore runs inside [`install`] (through
//! [`DetachExt::detached`](crate::errors::DetachExt::detached), or directly
//! in the training worker), on a pool of the current process.
//!
//! Every process of a fork lineage keeps its pool in the slot of its fork
//! generation. Python's fork hook ([`after_fork`], which the module
//! registers with `os.register_at_fork`) advances the generation in every
//! forked child, and a process that finds its slot claimed by another
//! process (one that forked it without Python's hooks) advances it itself.
//! A process thus only ever initializes a slot it claimed, which no ancestor
//! touched, so no slot holds a lock or a half-built pool of a thread the fork
//! left behind; inherited pools are never dropped. Pools are sized like
//! rayon's global pool: `RAYON_NUM_THREADS`, else the CPU count.
//!
//! Installing from a thread outside the pool hands the closure to a pool
//! thread and waits for it; work that never touches rayon detaches with
//! plain `Python::detach` instead. A thread of the pool itself (installed
//! work, or a Python callback that work runs) must not hand work to the
//! pool from another thread and wait: every pool thread may be waiting on
//! the caller, so the work would never start. [`on_pool_thread`] identifies
//! pool threads so callers can run such work inline.

use crate::errors::refuse;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Fork generations of one lineage, with a pool slot each.
const GENERATIONS: usize = 64;

/// One generation's pool and the process that claimed the slot (`0`:
/// unclaimed; no process has id 0).
struct Slot {
    owner: AtomicU32,
    threads: OnceLock<ThreadPool>,
}

static SLOTS: [Slot; GENERATIONS] = [const {
    Slot {
        owner: AtomicU32::new(0),
        threads: OnceLock::new(),
    }
}; GENERATIONS];

/// This process's fork generation: the slot of its pool.
static GENERATION: AtomicUsize = AtomicUsize::new(0);

/// `os.register_at_fork`'s `after_in_child` hook: a forked child moves to
/// a slot of its own.
#[pyfunction]
pub(crate) fn after_fork() {
    GENERATION.fetch_add(1, Ordering::AcqRel);
}

/// This process's pool, built on first use.
fn current() -> PyResult<&'static ThreadPool> {
    let pid = std::process::id();
    loop {
        let generation = GENERATION.load(Ordering::Acquire);
        let slot = SLOTS.get(generation).ok_or_else(|| {
            refuse(format!(
                "cannot start a thread pool after {GENERATIONS} nested forks"
            ))
        })?;
        match slot
            .owner
            .compare_exchange(0, pid, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {}
            Err(owner) if owner == pid => {}
            // Claimed by the process this one was forked from.
            Err(_) => {
                let _ = GENERATION.compare_exchange(
                    generation,
                    generation + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                continue;
            }
        }
        if let Some(threads) = slot.threads.get() {
            return Ok(threads);
        }
        let built = ThreadPoolBuilder::new()
            .build()
            .map_err(|error| PyOSError::new_err(format!("cannot start a thread pool: {error}")))?;
        // A racing thread of this process may initialize the slot first;
        // its pool serves both, and this one is dropped.
        return Ok(slot.threads.get_or_init(move || built));
    }
}

/// Runs `f` inside this process's pool, so its rayon work runs there.
pub(crate) fn install<T: Send>(f: impl FnOnce() -> T + Send) -> PyResult<T> {
    Ok(current()?.install(f))
}

/// Whether the current thread is one of this process's pool threads,
/// running installed work or a Python callback of it. Such a thread runs
/// nested work inline: it is already in the pool, and work it queued there
/// could wait for this very thread.
pub(crate) fn on_pool_thread() -> bool {
    SLOTS
        .get(GENERATION.load(Ordering::Acquire))
        .filter(|slot| slot.owner.load(Ordering::Acquire) == std::process::id())
        .and_then(|slot| slot.threads.get())
        .is_some_and(|threads| threads.current_thread_index().is_some())
}
