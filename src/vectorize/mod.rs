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

mod container;
mod data;
mod vectorizable;

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

pub use container::{
    Chunks, ChunksExact, GenericChunks, GenericChunksExact, IntoIter, Iter, RefChunks,
    RefChunksExact, RefIter, RefSlice, Slice, VectorIndex, Vectorized, VectorizedAligned,
    VectorizedPadded, VectorizedUnaligned,
};
pub use data::{PaddedMut, UnalignedMut, VectorizedData};
pub use vectorizable::{Vectorizable, VectorizeError};

// === COMMON UTILITIES THAT PROBABLY BELONG ELSEWHERE ===

/// Build an array using a mapping from index to value
///
/// This may look suspiciously like `core::array::from_fn()`, because it is
/// actually a clone of that function. Unfortunately, as of rustc 1.67, the real
/// thing does not optimize well because the compiler fails to inline some
/// steps, so we need to clone it for performance...
///
/// A fix has landed in nightly, and is scheduled to be featured in rustc 1.69.
/// So this code can eventually go away, once that release is old news...
/// https://github.com/rust-lang/rust/issues/108765
#[inline(always)]
fn array_from_fn<const SIZE: usize, T>(mut idx_to_elem: impl FnMut(usize) -> T) -> [T; SIZE] {
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

    // Use PartialArray to implement the array_from_fn functionality
    let mut array = PartialArray::new();
    for idx in 0..SIZE {
        array.push(idx_to_elem(idx));
    }
    array.collect()
}

/// Query the configuration of the [`Vector`] type
///
/// # Safety
///
/// Users of this trait may rely on the provided information to be correct
/// for safety.
pub unsafe trait VectorInfo:
    AsRef<[Self::Scalar]> + Copy + From<Self::Array> + Into<Self::Array> + Sized + 'static
{
    /// Inner scalar type (commonly called B in generics)
    type Scalar: Copy + Repr + Sized + 'static;

    /// Number of vector lanes (commonly called S in generics)
    const LANES: usize;

    /// Equivalent array type (will always be `[Self::Scalar; Self::LANES]`,
    /// but Rust does not support asserting this at the moment)
    type Array: Copy + Sized + 'static;

    /// Assert that Self is an overaligned Self::Array
    ///
    /// We hope this would be true, but it's not actually guaranteed by Rust
    /// for non-repr(transparent) types and we rely on it for safety here...
    #[inline(always)]
    #[doc(hidden)]
    fn assert_overaligned_array() {
        assert_eq!(
            core::mem::size_of::<Self>(),
            core::mem::size_of::<Self::Array>()
        )
    }

    /// Build from an index-to-element mapping
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::u32x16;
    use proptest::prelude::*;
    use std::cell::Cell;

    // === COMMON TEST HARNESS ===

    // Vector type we are going to test this module + submodules with
    //
    // We can afford to only test for a single vector type because all
    // functionality within this module is about shuffling data around without
    // interpreting it, so we just need a vector/element type with enough bytes
    // to assert that multi-byte operations work, multi-lanes operation work,
    // and (over-)alignment is handled correctly.
    pub(crate) type V = u32x16;
    pub(crate) type VScalar = <V as VectorInfo>::Scalar;
    pub(crate) type VArray = <V as VectorInfo>::Array;

    prop_compose! {
        /// Generate an arbitrary value of type V
        pub(crate) fn any_v()(array in any::<VArray>()) -> V {
            V::from(array)
        }
    }

    // === TESTS FOR THIS MODULE ===

    // Check that array_from_fn produces the expected output
    proptest! {
        #[test]
        fn array_from_fn(input in any::<[u64; 32]>()) {
            let output = super::array_from_fn(|idx| input[idx]);
            assert_eq!(output, input);
        }
    }

    // Check that array_from_fn handles Drop correctly
    #[derive(Debug)]
    struct Element;
    //
    thread_local! {
        pub static CREATED_ELEMENTS: Cell<usize> = Cell::new(0);
        pub static DROPPED_ELEMENTS: Cell<usize> = Cell::new(0);
    }
    //
    fn reset_element_counters() {
        CREATED_ELEMENTS.with(|c| c.set(0));
        DROPPED_ELEMENTS.with(|c| c.set(0));
    }
    //
    fn created_elements() -> usize {
        CREATED_ELEMENTS.with(|c| c.get())
    }
    //
    fn dropped_elements() -> usize {
        DROPPED_ELEMENTS.with(|c| c.get())
    }
    //
    impl Default for Element {
        fn default() -> Self {
            CREATED_ELEMENTS.with(|c| c.set(c.get() + 1));
            Self
        }
    }
    //
    impl Drop for Element {
        fn drop(&mut self) {
            DROPPED_ELEMENTS.with(|c| c.set(c.get() + 1));
        }
    }
    //
    #[test]
    fn array_from_fn_drop() {
        // Elements should be created and dropped the right number of times
        const SIZE: usize = 4;
        let arr: [Element; SIZE] = super::array_from_fn(|idx| {
            assert_eq!(created_elements(), idx);
            assert_eq!(dropped_elements(), 0);
            Element::default()
        });
        assert_eq!(created_elements(), SIZE);
        assert_eq!(dropped_elements(), 0);
        std::mem::drop(arr);
        assert_eq!(created_elements(), SIZE);
        assert_eq!(dropped_elements(), SIZE);
        reset_element_counters();

        // Panics should be handled correctly
        for panic_idx in 0..SIZE {
            std::panic::catch_unwind(|| {
                super::array_from_fn::<SIZE, Element>(|idx| {
                    assert_eq!(created_elements(), idx);
                    assert_eq!(dropped_elements(), 0);
                    if idx == panic_idx {
                        panic!();
                    }
                    Element::default()
                })
            })
            .expect_err("Should have panicked");
            assert_eq!(created_elements(), panic_idx);
            assert_eq!(dropped_elements(), panic_idx);
            reset_element_counters();
        }
    }

    // Check that the VectorInfo trait is implemented correctly
    #[test]
    fn vector_info() {
        // This test depends on the choice of V made above, assert it
        let _: u32x16 = V::default();
        let _: u32 = 0 as VScalar; // Assert V::Scalar is u32
        assert_eq!(V::LANES, 16);
        let _: [u32; 16] = VArray::default();
        V::assert_overaligned_array();
    }

    proptest! {
        #[test]
        fn v_from_fn(array in any::<VArray>()) {
            let output = V::from_fn(|idx| array[idx]);
            assert_eq!(VArray::from(output), array);
        }
    }
}
