//! A numpy array's buffer read in place: the extension's one `unsafe`
//! block, behind `DMatrix(..., copy=False)`.
//!
//! [`BorrowedArray`] keeps a read-only, C-contiguous, aligned `float32` array
//! alive and hands the crate its buffer as a slice, so a matrix over it (and
//! its copies and views) reads the array without copying it. Numpy cannot
//! promise the buffer stays unchanged: an array sharing its memory (its base,
//! or another view) may be writeable, and `resize(refcheck=False)` frees a
//! buffer whatever references it. The Python API states the caller's side: the
//! array, and every array sharing its memory, stays unchanged while a matrix
//! borrows it.
//!
//! Nothing in numpy enforces that: `WRITEABLE` is a per-array flag any caller
//! can set again (a view of a read-only array included), and rust-numpy's
//! borrow guard carries the GIL token, so it cannot be held over a matrix's
//! life (its storage is shared and dropped on arbitrary threads). `new`
//! refuses what numpy can show is mutable now, and what it cannot show is
//! not: a writeable array, an array whose base chain is writeable (a
//! read-only view of a live array, which its owner can still write), a base
//! that is neither an array nor a buffer the buffer protocol reports as
//! read-only (an object exposing only an array interface, as `as_strided`
//! builds, or a capsule: either may hide writeable memory), and a buffer
//! another extension holds a `PyReadwriteArray` of (rust-numpy's shared
//! borrow, keyed on the array's base, taken and released by `new`). Past
//! that, the choice of `copy=False` is the caller's promise, as documented.

use crate::errors::refuse;
use numpy::{PyArray2, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::buffer::PyUntypedBuffer;
use pyo3::prelude::*;
use pyo3::types::PyMemoryView;

/// A read-only `float32` array's buffer, kept alive (see the module docs).
pub(crate) struct BorrowedArray {
    /// Keeps the array, and with it the buffer, alive.
    _array: Py<PyArray2<f32>>,
    /// The buffer's address (provenance exposed) and element count.
    address: usize,
    len: usize,
}

impl BorrowedArray {
    /// `array`'s buffer, when `array` is read-only, C-contiguous, and
    /// aligned.
    pub(crate) fn new(array: &Bound<'_, PyArray2<f32>>) -> PyResult<Self> {
        let writeable = is_writeable(array.as_any())?;
        if writeable {
            return Err(refuse(
                "copy=False borrows a read-only array: mark it with setflags(write=False), \
                 or load it with np.load(..., mmap_mode=\"r\")",
            ));
        }
        if !array.is_c_contiguous() || !array.is_aligned() {
            return Err(refuse(
                "copy=False borrows a C-contiguous, aligned array: pass \
                 np.ascontiguousarray(data) (which copies)",
            ));
        }
        ensure_read_only_bases(array)?;
        // A buffer another extension borrowed mutably has no flag of its own,
        // but rust-numpy's shared borrow is keyed on the array's base, so it
        // fails here while that borrow lives. (`try_readonly`'s guard cannot
        // live beyond this call; see the module docs.)
        if array.try_readonly().is_err() {
            return Err(refuse(
                "copy=False cannot borrow data: another Rust extension holds a mutable \
                 borrow of it, which it must release first",
            ));
        }
        Ok(BorrowedArray {
            _array: array.clone().unbind(),
            address: array.data().expose_provenance(),
            len: array.len(),
        })
    }
}

/// Refuses `array` when something it is a view of can still write the buffer,
/// or may: a view's own flag can be cleared while its base stays writeable,
/// and the base's owner could then change the values a matrix reads. A
/// memory view's `obj` is the buffer's owner (a view without one owns it),
/// and an array over one has the view as its base; a non-array owner (a
/// memory map, `bytes`, ...) is asked through the buffer protocol. (numpy
/// lets any flag be set again afterwards; past this, `copy=False` is the
/// caller's promise.)
fn ensure_read_only_bases(array: &Bound<'_, PyArray2<f32>>) -> PyResult<()> {
    let mut current = array.clone().into_any();
    for _ in 0..64 {
        let base = ndarray_attribute(current.py(), &current, "base")?;
        if base.is_none() {
            return Ok(());
        }
        let owner = match base.cast_into::<PyMemoryView>() {
            Ok(view) => {
                let exporter = view.getattr("obj")?;
                if exporter.is_none() {
                    view.into_any()
                } else {
                    exporter
                }
            }
            Err(base) => base.into_inner(),
        };
        let Ok(array) = owner.clone().cast_into::<PyUntypedArray>() else {
            return ensure_read_only_buffer(&owner);
        };
        if is_writeable(array.as_any())? {
            return Err(mutable_owner());
        }
        current = array.into_any();
    }
    Err(refuse(
        "copy=False cannot follow data's base chain: pass a read-only array of its own, \
         or load it with np.load(..., mmap_mode=\"r\")",
    ))
}

/// `array`'s writeable flag, read through `numpy.ndarray`'s own descriptor:
/// a subclass that shadows `flags` (or `base`) cannot hide a writeable buffer.
fn is_writeable(array: &Bound<'_, PyAny>) -> PyResult<bool> {
    let flags = ndarray_attribute(array.py(), array, "flags")?;
    flags.getattr("writeable")?.extract()
}

/// `array`'s `name` attribute, read through `numpy.ndarray`'s descriptor
/// rather than the instance, which a subclass may shadow.
fn ndarray_attribute<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let ndarray = py.get_type::<PyUntypedArray>();
    ndarray
        .getattr(name)?
        .call_method1("__get__", (array, ndarray))
}

/// `owner`'s buffer is read-only. An owner without a buffer to ask could
/// still write the memory, so it is refused too.
fn ensure_read_only_buffer(owner: &Bound<'_, PyAny>) -> PyResult<()> {
    match PyUntypedBuffer::get(owner) {
        Ok(buffer) if buffer.readonly() => Ok(()),
        Ok(_) => Err(mutable_owner()),
        Err(_) => Err(refuse(format!(
            "copy=False cannot tell whether data's base (a {}) can write its memory: it \
             exposes no buffer to ask; pass a read-only array of its own, or copy=True",
            owner.get_type().name()?
        ))),
    }
}

fn mutable_owner() -> PyErr {
    refuse(
        "copy=False needs a read-only buffer, but data's base can still write it, which \
         would change values a matrix reads: mark that array read-only \
         (setflags(write=False)), or load a read-only copy with np.load(..., mmap_mode=\"r\")",
    )
}

impl AsRef<[f32]> for BorrowedArray {
    #[allow(
        unsafe_code,
        reason = "the extension's one read of a buffer numpy owns (see the module docs)"
    )]
    fn as_ref(&self) -> &[f32] {
        if self.len == 0 {
            return &[];
        }
        let data = std::ptr::with_exposed_provenance::<f32>(self.address);
        // SAFETY: `data` is the data pointer of `_array`, a C-contiguous,
        // aligned `float32` array of `len` elements, so it points to `len`
        // initialized, aligned `f32`s. `_array` keeps the array alive as long
        // as `self` lives, and numpy keeps a live array's buffer in place
        // (`resize` refuses while other references hold it). The array is
        // flagged read-only, and `copy=False`'s contract has the caller keep
        // every array sharing its memory unchanged, so nothing writes the
        // memory while this shared borrow lives.
        unsafe { std::slice::from_raw_parts(data, self.len) }
    }
}
