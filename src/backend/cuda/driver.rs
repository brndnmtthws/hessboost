//! The backend's layer over cuda-core: the error type that tells a refused
//! call from a driver failure, borrowed ranges of device buffers (a kernel's
//! `&[T]` parameter is a range's address and length, cuda-oxide's slice
//! ABI), kernel launches with by-value parameters, and stream-ordered
//! zeroing and pageable copies that complete before they return.

use cuda_core::{CudaFunction, CudaStream, DeviceBuffer, DeviceCopy, DriverError, sys};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ops::{Bound, Range, RangeBounds};
use std::sync::Arc;

/// Why a backend operation did not complete on the device.
#[derive(Debug, Clone, Copy)]
pub(super) enum Error {
    /// A driver call failed; [`Error::poisons`] says whether the context
    /// survives it.
    Driver(DriverError),
    /// The backend declined before reaching the driver: a precondition or
    /// a capacity limit of this call. The device is unaffected.
    Refused(&'static str),
}

impl Error {
    /// Whether the error leaves the CUDA context unusable, so every later
    /// operation on the device must run on the host. Refusals, and driver
    /// failures that leave device state as it was (an allocation that does
    /// not fit, an argument the driver rejects, a launch asking for more
    /// resources than a block has), affect only the call that met them;
    /// anything else (a kernel fault, a lost device) is sticky, or is
    /// treated as such.
    pub(super) fn poisons(self) -> bool {
        match self {
            Self::Refused(_) => false,
            Self::Driver(DriverError(code)) => !matches!(
                code,
                sys::cudaError_enum_CUDA_ERROR_OUT_OF_MEMORY
                    | sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE
                    | sys::cudaError_enum_CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
            ),
        }
    }
}

impl From<DriverError> for Error {
    fn from(error: DriverError) -> Self {
        Self::Driver(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(error) => write!(f, "{error}"),
            Self::Refused(reason) => f.write_str(reason),
        }
    }
}

/// A backend operation's result.
pub(super) type Result<T, E = Error> = std::result::Result<T, E>;

/// A kernel launch's grid, block and dynamic shared memory.
#[derive(Debug, Clone, Copy)]
pub(super) struct LaunchConfig {
    pub(super) grid_dim: (u32, u32, u32),
    pub(super) block_dim: (u32, u32, u32),
    pub(super) shared_mem_bytes: u32,
}

/// A device range: an address and an element count, borrowed from the
/// buffer that holds it.
#[derive(Debug, Clone, Copy)]
pub(super) struct View<'a, T> {
    addr: sys::CUdeviceptr,
    len: usize,
    _buffer: PhantomData<&'a [T]>,
}

/// A device range borrowed for writing.
#[derive(Debug)]
pub(super) struct ViewMut<'a, T> {
    addr: sys::CUdeviceptr,
    len: usize,
    _buffer: PhantomData<&'a mut [T]>,
}

/// Device memory of `T`s: a whole buffer or a range of one.
pub(super) trait Memory<T> {
    /// The first element's address.
    fn addr(&self) -> sys::CUdeviceptr;
    /// The number of elements.
    fn count(&self) -> usize;
}

/// Device memory that may be written ([`DeviceBuffer`] or [`ViewMut`]).
pub(super) trait MemoryMut<T>: Memory<T> {}

impl<T> Memory<T> for DeviceBuffer<T> {
    fn addr(&self) -> sys::CUdeviceptr {
        self.cu_deviceptr()
    }
    fn count(&self) -> usize {
        self.len()
    }
}

impl<T> MemoryMut<T> for DeviceBuffer<T> {}

impl<T> Memory<T> for View<'_, T> {
    fn addr(&self) -> sys::CUdeviceptr {
        self.addr
    }
    fn count(&self) -> usize {
        self.len
    }
}

impl<T> Memory<T> for ViewMut<'_, T> {
    fn addr(&self) -> sys::CUdeviceptr {
        self.addr
    }
    fn count(&self) -> usize {
        self.len
    }
}

impl<T> MemoryMut<T> for ViewMut<'_, T> {}

impl<T> ViewMut<'_, T> {
    /// The first `mid` elements and the rest; panics past the end (an
    /// internal invariant, as for a slice).
    pub(super) fn split_at(self, mid: usize) -> (Self, Self) {
        assert!(
            mid <= self.len,
            "device split at {mid} past {} elements",
            self.len
        );
        let rest = ViewMut {
            addr: offset_addr::<T>(self.addr, mid),
            len: self.len - mid,
            _buffer: PhantomData,
        };
        let first = ViewMut {
            addr: self.addr,
            len: mid,
            _buffer: PhantomData,
        };
        (first, rest)
    }
}

/// `range` of `len` elements as `start..end`; panics past `len` (an
/// internal invariant, as for a slice index).
fn bounds(range: impl RangeBounds<usize>, len: usize) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end && end <= len,
        "device range {start}..{end} out of bounds of {len} elements"
    );
    start..end
}

/// The address of element `offset` past `base`.
fn offset_addr<T>(base: sys::CUdeviceptr, offset: usize) -> sys::CUdeviceptr {
    base + (offset * size_of::<T>()) as u64
}

/// Ranges of a device buffer.
pub(super) trait Slice<T> {
    /// `range` of the buffer, for reading.
    fn slice(&self, range: impl RangeBounds<usize>) -> View<'_, T>;
    /// `range` of the buffer, for writing.
    fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> ViewMut<'_, T>;
}

impl<T> Slice<T> for DeviceBuffer<T> {
    fn slice(&self, range: impl RangeBounds<usize>) -> View<'_, T> {
        let r = bounds(range, self.len());
        View {
            addr: offset_addr::<T>(self.cu_deviceptr(), r.start),
            len: r.len(),
            _buffer: PhantomData,
        }
    }

    fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> ViewMut<'_, T> {
        let r = bounds(range, self.len());
        ViewMut {
            addr: offset_addr::<T>(self.cu_deviceptr(), r.start),
            len: r.len(),
            _buffer: PhantomData,
        }
    }
}

/// A kernel parameter: the bytes the driver copies into the kernel's
/// parameter space at its position.
///
/// # Safety
///
/// [`LaunchArg::push`] appends exactly one parameter's bytes, in the layout
/// of the kernel parameter it is passed as: a by-value scalar or
/// `#[repr(C)]` struct of at most 8-byte alignment, or a device address.
pub(super) unsafe trait LaunchArg {
    /// Append this parameter to `params`.
    fn push(&self, params: &mut Params);
}

/// A launch's parameter bytes, each parameter starting on an 8-byte word.
#[derive(Default)]
pub(super) struct Params {
    words: Vec<MaybeUninit<u64>>,
    starts: Vec<usize>,
}

impl Params {
    /// Append `value`'s bytes as one parameter.
    pub(super) fn push_value<T: Copy>(&mut self, value: &T) {
        const { assert!(align_of::<T>() <= 8) };
        let start = self.words.len();
        let words = size_of::<T>().div_ceil(8).max(1);
        self.words.resize(start + words, MaybeUninit::zeroed());
        // SAFETY: the new words hold at least `size_of::<T>()` bytes and do
        // not overlap `value`; the words are only ever handed to the driver
        // as raw parameter memory, never read as `u64`s.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(value).cast::<u8>(),
                self.words.as_mut_ptr().add(start).cast::<u8>(),
                size_of::<T>(),
            );
        }
        self.starts.push(start);
    }
}

/// Parameters passed by value.
macro_rules! by_value {
    ($($ty:ty),* $(,)?) => {
        $(
            // SAFETY: a plain value of at most 8-byte alignment, pushed as
            // its own bytes.
            unsafe impl LaunchArg for $ty {
                fn push(&self, params: &mut Params) {
                    params.push_value(self);
                }
            }
        )*
    };
}
pub(super) use by_value;

by_value!(u32, i32, u64, f32, f64);

// SAFETY: a device buffer is passed as its address, an 8-byte pointer.
unsafe impl<T> LaunchArg for DeviceBuffer<T> {
    fn push(&self, params: &mut Params) {
        params.push_value(&self.cu_deviceptr());
    }
}

// SAFETY: as for a buffer.
unsafe impl<T> LaunchArg for View<'_, T> {
    fn push(&self, params: &mut Params) {
        params.push_value(&self.addr);
    }
}

// SAFETY: as for a buffer.
unsafe impl<T> LaunchArg for ViewMut<'_, T> {
    fn push(&self, params: &mut Params) {
        params.push_value(&self.addr);
    }
}

/// A launch argument as the caller lends it: shared for what the kernel
/// reads (and by-value parameters), exclusive for memory it writes. Either
/// way the borrow lasts until the launch is queued.
pub(super) trait Lent<'a> {
    type Target: ?Sized;
    fn target(&self) -> &Self::Target;
}

impl<'a, A: ?Sized> Lent<'a> for &'a A {
    type Target = A;
    fn target(&self) -> &A {
        self
    }
}

impl<'a, A: ?Sized> Lent<'a> for &'a mut A {
    type Target = A;
    fn target(&self) -> &A {
        self
    }
}

/// One kernel launch being assembled: its parameters in the kernel's
/// order, with the device memory they name borrowed until the launch is
/// queued.
pub(super) struct Launch<'a> {
    stream: &'a CudaStream,
    function: &'a CudaFunction,
    params: Params,
    _args: PhantomData<&'a ()>,
}

impl<'a> Launch<'a> {
    /// Push `arg` (a by-value parameter, or device memory as its address).
    pub(super) fn arg<L: Lent<'a>>(&mut self, arg: L) -> &mut Self
    where
        L::Target: LaunchArg,
    {
        arg.target().push(&mut self.params);
        self
    }

    /// Push `range` as a slice parameter (`&[T]` or `DisjointSlice<T>` in
    /// the kernel): its address, then its length.
    pub(super) fn slice<T, L: Lent<'a>>(&mut self, range: L) -> &mut Self
    where
        L::Target: Memory<T>,
    {
        let range = range.target();
        self.params.push_value(&range.addr());
        self.params.push_value(&(range.count() as u64));
        self
    }

    /// Push `range`, consecutive pairs of `T`s, as a slice of the kernel's
    /// pair type (`F32x2`, `I64x2` or `F64x2` over `f32`, `i64` or `f64`):
    /// its address, then its length in pairs.
    pub(super) fn pairs<T, L: Lent<'a>>(&mut self, range: L) -> &mut Self
    where
        L::Target: Memory<T>,
    {
        let range = range.target();
        debug_assert!(range.count().is_multiple_of(2), "an odd pair slice");
        self.params.push_value(&range.addr());
        self.params.push_value(&((range.count() / 2) as u64));
        self
    }

    /// Push `len` elements at device address `addr` as a slice parameter:
    /// staged descriptors, which the staging arena keeps until the stream
    /// synchronizes.
    pub(super) fn raw_slice(&mut self, addr: sys::CUdeviceptr, len: usize) -> &mut Self {
        self.params.push_value(&addr);
        self.params.push_value(&(len as u64));
        self
    }

    /// Queue the launch on the stream.
    ///
    /// # Safety
    ///
    /// The kernel's parameters are the pushed ones, in order, and the
    /// geometry, the memory they name and its lifetime satisfy the kernel's
    /// contract.
    pub(super) unsafe fn launch(&mut self, config: LaunchConfig) -> Result<()> {
        let base = self.params.words.as_mut_ptr();
        let mut pointers: Vec<*mut c_void> = self
            .params
            .starts
            .iter()
            .map(|&start| base.wrapping_add(start).cast())
            .collect();
        // SAFETY: the caller's; each pointer addresses one parameter's bytes,
        // alive until this call returns.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                self.function,
                config.grid_dim,
                config.block_dim,
                config.shared_mem_bytes,
                self.stream,
                &mut pointers,
            )
        }
        .map_err(Error::Driver)
    }
}

/// The stream operations the backend uses.
pub(super) trait StreamExt {
    /// Start a launch of `function` on this stream.
    fn launch_builder<'a>(&'a self, function: &'a CudaFunction) -> Launch<'a>;

    /// A zeroed buffer of `len` elements (zeroed in stream order).
    fn alloc_zeros<T: DeviceCopy>(self: &Arc<Self>, len: usize) -> Result<DeviceBuffer<T>>;

    /// Zero `dst` in stream order.
    fn memset_zeros<T, D: MemoryMut<T>>(&self, dst: &mut D) -> Result<()>;

    /// Copy `src` into the front of `dst` and wait for the stream: `src`
    /// is pageable host memory, free to reuse once this returns.
    fn write_sync<T: DeviceCopy, D: MemoryMut<T>>(&self, src: &[T], dst: &mut D) -> Result<()>;

    /// Copy `src` into `dst` (of the same length) and wait for the stream,
    /// so `dst` holds the values once this returns.
    fn read_sync<T: DeviceCopy, D: Memory<T>>(&self, src: &D, dst: &mut [T]) -> Result<()>;
}

impl StreamExt for CudaStream {
    fn launch_builder<'a>(&'a self, function: &'a CudaFunction) -> Launch<'a> {
        Launch {
            stream: self,
            function,
            params: Params::default(),
            _args: PhantomData,
        }
    }

    fn alloc_zeros<T: DeviceCopy>(self: &Arc<Self>, len: usize) -> Result<DeviceBuffer<T>> {
        DeviceBuffer::zeroed(self, len).map_err(Error::Driver)
    }

    fn memset_zeros<T, D: MemoryMut<T>>(&self, dst: &mut D) -> Result<()> {
        let bytes = dst.count() * size_of::<T>();
        if bytes == 0 {
            return Ok(());
        }
        self.context().bind_to_thread()?;
        // SAFETY: `dst` names `bytes` bytes of device memory of this
        // stream's context, borrowed for writing.
        unsafe { cuda_core::simt::memory::memset_d8_async(dst.addr(), 0, bytes, self.cu_stream()) }
            .map_err(Error::Driver)
    }

    fn write_sync<T: DeviceCopy, D: MemoryMut<T>>(&self, src: &[T], dst: &mut D) -> Result<()> {
        if src.len() > dst.count() {
            return Err(Error::Refused("host copy longer than its device range"));
        }
        if src.is_empty() {
            return Ok(());
        }
        self.context().bind_to_thread()?;
        // SAFETY: `dst` holds at least `src.len()` elements; `src` stays
        // borrowed until the synchronization below has completed the copy.
        let queued = unsafe {
            cuda_core::simt::memory::memcpy_htod_async(
                dst.addr(),
                src.as_ptr(),
                size_of_val(src),
                self.cu_stream(),
            )
        };
        let completed = self.synchronize();
        queued?;
        completed.map_err(Error::Driver)
    }

    fn read_sync<T: DeviceCopy, D: Memory<T>>(&self, src: &D, dst: &mut [T]) -> Result<()> {
        if dst.len() != src.count() {
            return Err(Error::Refused("device copy of a different length"));
        }
        if dst.is_empty() {
            return Ok(());
        }
        self.context().bind_to_thread()?;
        // SAFETY: `src` holds `dst.len()` elements; `dst` stays borrowed
        // until the synchronization below has completed the copy, and any
        // bytes of a device copy are valid `T`s (`DeviceCopy`).
        let queued = unsafe {
            cuda_core::simt::memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                src.addr(),
                size_of_val(dst),
                self.cu_stream(),
            )
        };
        let completed = self.synchronize();
        queued?;
        completed.map_err(Error::Driver)
    }
}
