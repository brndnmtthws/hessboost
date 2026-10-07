//! The extension's process-local rayon pool, rebuilt in a forked child.
//!
//! Crate code parallelizes on the current rayon pool. Rayon's global pool
//! does not survive `os.fork()`: the child has none of its threads, so its
//! first parallel job waits forever. Every native entry point that may run
//! rayon work therefore runs inside [`install`] (through
//! [`DetachExt::detached`](crate::errors::DetachExt::detached), or directly
//! in the training worker), on a pool owned by the process that built it:
//! a process whose id differs from the builder's (a forked child) builds
//! its own. Stale pools are leaked, never dropped (their threads do not
//! exist in the child and their internal locks may be held). Pools are
//! sized like rayon's global pool: `RAYON_NUM_THREADS`, else the CPU count.
//!
//! The pools form a lock-free list of [`OnceBox`] slots (`once_cell`'s
//! racing cells: a losing initializer drops its value), so no thread can
//! hold a lock on them at fork time, as one initializing a `std`
//! `OnceLock`/`LazyLock` would. Installing from a thread outside the
//! pool hands the closure to a pool thread; work that never touches rayon
//! detaches with plain `Python::detach` instead.

use once_cell::race::OnceBox;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// One process's pool, followed by the slot of the next process's (a
/// forked descendant's).
struct Node {
    pid: u32,
    threads: ThreadPool,
    next: OnceBox<Node>,
}

/// The first pool built in this process or the ancestors it was forked from.
static POOLS: OnceBox<Node> = OnceBox::new();

/// This process's pool, built on first use. The list's last pool is the
/// newest; any other was inherited from an ancestor (even one whose
/// process id this process reuses).
fn current() -> &'static ThreadPool {
    let pid = std::process::id();
    let mut slot = &POOLS;
    loop {
        let pool = slot.get_or_init(|| {
            Box::new(Node {
                pid,
                threads: ThreadPoolBuilder::new()
                    .build()
                    .expect("failed to start the hessboost thread pool"),
                next: OnceBox::new(),
            })
        });
        if pool.pid == pid && pool.next.get().is_none() {
            return &pool.threads;
        }
        slot = &pool.next;
    }
}

/// Runs `f` inside this process's pool, so its rayon work runs there.
pub(crate) fn install<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    current().install(f)
}
