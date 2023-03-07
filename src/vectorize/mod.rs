//! The [`Vectorizable`] trait and its service types
//!
//! The [`Vectorizable`] trait allows to turning slices of base types and
//! vectors into a slice-like container of vectors, both in separation and in
//! tandem. The rest of this module provides the related types and traits.
//!
//! Usually, it is enough to bring in the [`prelude`][crate::prelude], which
//! already contains the trait. It is seldom necessary to interact with this
//! module directly.
//!
//! # Examples
//!
//! ```rust
//! use slipstream::prelude::*;
//!
//! fn double(input: &[u32], output: &mut [u32]) {
//!     let two = u32x8::splat(2);
//!     for (i, mut o) in (input, output).vectorize() {
//!         *o = two * i;
//!     }
//! }
//! # double(&[], &mut [])
//! ```

mod data;
mod vectorizable;
mod vectors;

use crate::{inner::Repr, vector::align::Align, Vector};
use core::mem::MaybeUninit;

// === GENERAL IMPLEMENTATION NOTES ===

// Since we are building a slice-like type here, we can end up at the bottom of
// hot user loops and are very sensitive to unwise compiler inlining decisions,
// that are unfortunately a common occurence when using rustc with multiple
// codegen units (which in turn is the default for release builds).
//
// However, usual caveats of excessive inlining apply: it can cause code bloat,
// resulting in slow compilation and runtime trashing of the CPU instruction
// cache.
//
// With this in mind, we propose the following inlining discipline:
//
// - If it's a zero-const abstraction meant to produce a few instructions or
//   a short trivial loop in generated code (e.g. array_from_fn, UnalignedMut,
//   Iterator::next()...), then it should be marked inline(always).
// - If it's meant to happen once per SIMD vector of the dataset, then it should
//   be marked inline(always).
// - If it can happen multiple times per dataset but should not normally happen
//   once per element (e.g. slicing, operations that can be run over slices like
//   iteration), then it should be marked inline.
// - If it should happen at most once per dataset, then no particular inlining
//   directive should be used.
// - If it is unexpected (e.g. indexing failure), then it can be marked cold,
//   and inline(never) too if optimizer doesn't take the cold hint properly.

// === PUBLIC RE-EXPORTS FROM INNER MODULES ===

pub use data::{PaddedMut, UnalignedMut, Vectorized};
pub use vectorizable::{Vectorizable, VectorizeError};
pub use vectors::{
    AlignedVectors, IntoIter, Iter, PaddedVectors, Slice, UnalignedVectors, VectorIndex, Vectors,
};

// === COMMON UTILITIES THAT PROBABLY BELONG ELSEWHERE ===

/// Build an array using a mapping from index to value
///
/// This may look suspiciously like `core::array::from_fn()`, because it is
/// actually a clone of that function. Unfortunately, at the time of
/// writing, the real thing does not optimize well because the compiler fails to
/// inline some steps, so we need to clone it for performance...
#[inline(always)]
fn array_from_fn<const SIZE: usize, T>(mut idx_to_elem: impl FnMut(usize) -> T) -> [T; SIZE] {
    let mut array = PartialArray::new();
    for idx in 0..SIZE {
        array.push(idx_to_elem(idx));
    }
    array.collect()
}

/// Partially initialized array
struct PartialArray<T, const N: usize> {
    inner: MaybeUninit<[T; N]>,
    num_initialized: usize,
}
//
impl<T, const N: usize> PartialArray<T, N> {
    /// Prepare to iteratively initialize an array
    #[inline(always)]
    fn new() -> Self {
        Self {
            inner: MaybeUninit::uninit(),
            num_initialized: 0,
        }
    }

    /// Initialize the next array element
    #[inline(always)]
    fn push(&mut self, value: T) {
        assert!(self.num_initialized < N);
        unsafe {
            let ptr = self
                .inner
                .as_mut_ptr()
                .cast::<T>()
                .add(self.num_initialized);
            ptr.write(value);
            self.num_initialized += 1;
        }
    }

    /// Assume the array is fully initialized and collect its value
    #[inline(always)]
    fn collect(self) -> [T; N] {
        assert_eq!(self.num_initialized, N);
        unsafe {
            let result = self.inner.assume_init_read();
            core::mem::forget(self);
            result
        }
    }
}
//
impl<T, const N: usize> Drop for PartialArray<T, N> {
    /// Drop already initialized elements on panic
    #[inline(always)]
    fn drop(&mut self) {
        let ptr = self.inner.as_mut_ptr().cast::<T>();
        for idx in 0..self.num_initialized {
            unsafe { ptr.add(idx).drop_in_place() };
        }
    }
}

/// Query the configuration of a Vector type
///
/// # Safety
///
/// Users of this trait may rely on the provided information to be correct
/// for safety.
//
pub unsafe trait VectorInfo:
    AsRef<[Self::Scalar]> + Copy + From<Self::Array> + Into<Self::Array> + Sized + 'static
{
    /// Inner scalar type (commonly called B in generics)
    type Scalar: Copy + Sized + 'static;

    /// Number of vector lanes (commonly called S in generics)
    const LANES: usize;

    /// Equivalent array type (will always be [Self::Scalar; Self::LANES],
    /// but Rust does not support asserting this at the moment)
    type Array: Copy + Sized + 'static;

    /// Assert that Self is an overaligned Self::Array
    ///
    /// We hope this would be true, but it's not actually guaranteed by Rust
    /// for non-repr(transparent) types and we rely on it for safety here...
    #[inline(always)]
    fn assert_overaligned_array() {
        assert_eq!(
            core::mem::size_of::<Self>(),
            core::mem::size_of::<Self::Array>()
        )
    }

    /// Build from an index -> element mapping
    fn from_fn(idx_to_elem: impl FnMut(usize) -> Self::Scalar) -> Self;
}
//
unsafe impl<A: Align, B: Repr, const S: usize> VectorInfo for Vector<A, B, S> {
    type Scalar = B;
    const LANES: usize = S;
    type Array = [B; S];

    #[inline(always)]
    fn from_fn(idx_to_elem: impl FnMut(usize) -> Self::Scalar) -> Self {
        array_from_fn(idx_to_elem).into()
    }
}

// FIXME: Tests
