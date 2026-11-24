//! [`ComputeOnce`]: a lazily computed value on a frozen result class.

use std::sync::{Mutex, OnceLock};

/// A value computed on first use and then cached, settable through `&self`
/// (so a `#[pyclass(frozen)]` result can hold it).
///
/// The value is set only under the internal lock, so concurrent first calls
/// run the computation once: the others wait, then read it. A failed
/// computation is not cached; the next caller (including a thread that was
/// waiting) runs it again. Callers that release the GIL call
/// [`get_or_try_init`](Self::get_or_try_init) INSIDE `allow_threads`, so a
/// waiting thread holds no GIL, and the closure must make no Python calls.
pub(crate) struct ComputeOnce<T> {
    value: OnceLock<T>,
    lock: Mutex<()>,
}

impl<T> Default for ComputeOnce<T> {
    fn default() -> Self {
        Self {
            value: OnceLock::new(),
            lock: Mutex::new(()),
        }
    }
}

impl<T> ComputeOnce<T> {
    /// An empty cache.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The cached value, if computed.
    pub(crate) fn get(&self) -> Option<&T> {
        self.value.get()
    }

    /// The cached value, computing it with `f` under the lock if no caller
    /// has yet. An error from `f` is returned and nothing is cached.
    pub(crate) fn get_or_try_init<E>(&self, f: impl FnOnce() -> Result<T, E>) -> Result<&T, E> {
        if let Some(v) = self.value.get() {
            return Ok(v);
        }
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = self.value.get() {
            return Ok(v);
        }
        let v = f()?;
        Ok(self.value.get_or_init(|| v))
    }
}
