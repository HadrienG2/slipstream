//! The [`Vectorizable`] trait and a lot of its service types.
//!
//! The [`Vectorizable`] trait allows to turning slices of base types to iterators of vectors, both
//! in separation and in tandem. The rest of this module provides the related types and traits.
//!
//! Usually, it is enough to bring in the [`prelude`][crate::prelude], which already contains the
//! trait. It is seldom necessary to interact with this module directly.
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

use core::iter::FusedIterator;
use core::marker::PhantomData;
use core::mem::{self, MaybeUninit};
use core::ops::*;
use core::ptr;
use core::slice;

use crate::inner::Repr;
use crate::vector::align::Align;
use crate::Vector;

/// FIXME: This is an experiment to see how Vectorize could be extended to be
///        more friendly to rustc's optimizer.
//
// Proposed code inlining discipline:
//
// - If it's a zero-const abstraction meant to be optimized out
//   (e.g. array_from_fn, UnalignedMut, Iterator::next()...), then it should be
//   marked inline(always).
// - If it's meant to happen once per element of a dataset, then it should be
//   marked inline(always).
// - If it can happen multiple times per dataset but should not normally happen
//   once per element (e.g. slicing, operations that can be run over slices like
//   iteration), then it should be marked inline
// - If it should happen at most once per dataset, then no particular inlining
//   directive is used.
// - If it is unexpected (e.g. indexing failure), then it can be marked cold,
//   and inline(never) too if optimizer doesn't take the cold hint properly.
pub mod experimental {
    use crate::{inner::Repr, vector::align::Align, Vector};
    use core::{
        borrow::{Borrow, BorrowMut},
        iter::FusedIterator,
        marker::PhantomData,
        mem::MaybeUninit,
        ops::{
            Bound, Deref, DerefMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
            RangeToInclusive,
        },
        ptr::NonNull,
    };

    // === Step 0: General utilities that probably belong elsewhere ===

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

    // === Step 1: Abstraction over SIMD data access ===

    /// Vectorizable data reinterpreted as a slice of `Vector`s
    ///
    /// This trait tells you what types you can expect out of the implementation
    /// of `Vectorizable` and the `Vectors` collection that it emits.
    ///
    /// To recapitulate the basic rules here:
    ///
    /// - For all supported types T, a `Vectors` collection built out of `&[T]`
    ///   or a shared reference to a collection of T has iterators and getters
    ///   that emit owned `Vector` values.
    /// - If built out of `&mut [Vector]`, or out of `&mut [Scalar]` with an
    ///   assertion that the data has optimal SIMD layout
    ///   (see [`Vectorizable::vectorize_aligned()`]), it emits `&mut Vector`.
    /// - If built out of `&mut [Scalar]` without asserting optimal SIMD layout,
    ///   it emits a proxy type that emulates `&mut Vector`.
    /// - If build out of a tuple of the above, it emits tuples of the above
    ///   element types.
    ///
    /// # Safety
    ///
    /// Unsafe code may rely on the correctness of implementations of this trait
    /// as part of their safety proofs. The definition of "correct" is an
    /// implementation detail of this crate, therefore this trait should not
    /// be implemented outside of this crate.
    pub unsafe trait Vectorized<V: VectorInfo>: Sized {
        /// Owned element of the output `Vectors` collection
        type Element: Sized;

        /// Borrowed element of the output `Vectors` collection
        type ElementRef<'result>: Sized
        where
            Self: 'result;

        /// Slice view of this dataset
        type Slice<'result>: Vectorized<V, Element = Self::ElementRef<'result>>
            + VectorizedSliceImpl<V>
        where
            Self: 'result;

        /// Reinterpretation of this data as SIMD data that may not be aligned
        type Unaligned: Vectorized<V> + VectorizedImpl<V>;

        /// Reinterpretation of this data as SIMD data with optimal layout
        type Aligned: Vectorized<V> + VectorizedImpl<V>;
    }

    /// Entity that can be treated as the base pointer of an &[Vector] or
    /// &mut [Vector] slice
    ///
    /// Implementors of this trait operate in the context of an underlying real
    /// or simulated slice of SIMD vectors, or of a tuple of several slices of
    /// equal length that is made to behave like a slice of tuples.
    ///
    /// The length of the underlying slice is not known by this type, it is
    /// stored as part of the higher-level `Vectors` collection that this type
    /// is used to implement.
    ///
    /// Instead, implementors of this type behave like the pointer that
    /// `[Vector]::as_ptr()` would return, and their main purpose is to
    /// implement the `[Vector]::get_unchecked(idx)` operation of the slice,
    /// like `*ptr.add(idx)` would in a real slice.
    ///
    /// # Safety
    ///
    /// Unsafe code may rely on the correctness of implementations of this trait
    /// and the higher-level `Vectorized` trait as part of their safety proofs.
    ///
    /// The safety preconditions on `Vectorized` are that `Element` should
    /// not outlive `Self`, and that it should be safe to transmute `ElementRef`
    /// to `Element` in scenarios where either `Element` is `Copy` or the
    /// transmute is abstracted in such a way that the user cannot abuse it to
    /// get two copies of the same element. In other words, Element should be
    /// the maximal-lifetime version of ElementRef.
    ///
    /// Further, Slice::ElementRef should be pretty much the same GAT as
    /// Self::ElementRef, with just a different Self lifetime bound.
    ///
    /// Furthermore, a `Vectorized` impl is only allowed to implement `Copy` if
    /// the underlying element type is `Copy`.
    #[doc(hidden)]
    pub unsafe trait VectorizedImpl<V: VectorInfo>: Vectorized<V> + Sized {
        /// Access the underlying slice at vector index `idx` without bounds
        /// checking, adding scalar padding values if data is missing at the
        /// end of the target vector.
        ///
        /// `is_last` is the truth that the last element of the slice is being
        /// targeted (and thus scalar data may require padding).
        ///
        /// # Safety
        ///
        /// - Index `idx` must be in within the bounds of the underlying slice.
        /// - `is_last` must be true if and only if the last element of the
        ///   slice is being accessed.
        unsafe fn get_unchecked(&mut self, idx: usize, is_last: bool) -> Self::ElementRef<'_>;

        /// Turn this data into the equivalent slice
        ///
        /// Lifetime-shrinking no-op if Self is already a slice, but turns
        /// owned data into slices.
        fn as_slice(&mut self) -> Self::Slice<'_>;

        /// Unsafely cast this data to the equivalent slice or collection of
        /// unaligned `Vector`s
        ///
        /// # Safety
        ///
        /// The underlying scalar data must have a number of elements that is
        /// a multiple of `V::LANES`.
        unsafe fn as_unaligned_unchecked(self) -> Self::Unaligned;

        /// Unsafely cast this data to the equivalent slice or collection of Vector.
        ///
        /// # Safety
        ///
        /// The underlying scalar data must be aligned like V and have a number
        /// of elements that is a multiple of `V::LANES`.
        unsafe fn as_aligned_unchecked(self) -> Self::Aligned;
    }

    /// `VectorizedImpl` that is a true slice, i.e. does not own its elements
    /// and can be split
    #[doc(hidden)]
    pub unsafe trait VectorizedSliceImpl<V: VectorInfo>: VectorizedImpl<V> + Sized {
        /// Divides this slice into two at an index, without doing bounds
        /// checking, and returns slices targeting the two halves of the dataset
        ///
        /// The first slice will contain all indices from `[0, mid)` (excluding
        /// the index `mid` itself) and the second will contain all indices from
        /// `mid` to the end of the dataset.
        ///
        /// # Safety
        ///
        /// - Calling this method with an out-of-bounds index is undefined
        ///   behavior even if the resulting reference is not used. The caller
        ///   has to ensure that mid <= len.
        /// - `len` must be the length of the underlying `Vectors` slice.
        //
        // NOTE: Not being able to say that the output has lower lifetime than
        //       Self will cause lifetime issues, but being able to say that a
        //       slice of a slice is a slice is more important, and
        //       unfortunately we don't have access to Rust's subtyping and
        //       variance since we're not manipulating true references.
        unsafe fn split_at_unchecked(self, mid: usize, len: usize) -> (Self, Self);
    }

    /// Read access to aligned SIMD data
    //
    // --- Internal docs start here ---
    //
    // Base pointer of an `&[Vector]` slice, tagged with lifetime information
    #[derive(Copy, Clone)]
    pub struct AlignedData<'target, V: VectorInfo>(NonNull<V>, PhantomData<&'target [V]>);
    //
    impl<'target, V: VectorInfo> From<&'target [V]> for AlignedData<'target, V> {
        fn from(data: &'target [V]) -> Self {
            unsafe { Self::from_data_ptr(NonNull::from(data)) }
        }
    }
    //
    impl<'target, V: VectorInfo> AlignedData<'target, V> {
        /// Construct from raw pointer to a slice of data
        ///
        /// # Safety
        ///
        /// It must be valid to access the slice behind `ptr` for lifetime 'target
        unsafe fn from_data_ptr(ptr: NonNull<[V]>) -> Self {
            Self(ptr.cast::<V>(), PhantomData)
        }

        /// Base pointer used by get_unchecked(idx)
        ///
        /// # Safety
        ///
        /// `idx` must be in range for the surrounding slice
        #[inline(always)]
        unsafe fn get_ptr(&self, idx: usize) -> NonNull<V> {
            unsafe { NonNull::new_unchecked(self.0.as_ptr().add(idx)) }
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for AlignedData<'target, V> {
        type Element = V;
        type ElementRef<'result> = V where Self: 'result;
        type Slice<'result> = AlignedData<'result, V> where Self: 'result;
        type Unaligned = Self;
        type Aligned = Self;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for AlignedData<'target, V> {
        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _is_last: bool) -> V {
            unsafe { *self.get_ptr(idx).as_ref() }
        }

        #[inline(always)]
        fn as_slice(&mut self) -> AlignedData<V> {
            AlignedData(self.0, PhantomData)
        }

        unsafe fn as_unaligned_unchecked(self) -> Self {
            self
        }

        unsafe fn as_aligned_unchecked(self) -> Self {
            self
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for AlignedData<'target, V> {
        #[inline(always)]
        unsafe fn split_at_unchecked(self, mid: usize, _len: usize) -> (Self, Self) {
            let wrap = |ptr| Self(ptr, PhantomData);
            (wrap(self.0), wrap(self.get_ptr(mid)))
        }
    }

    // Owned arrays of Vector must be stored as-is in the Vectors collection,
    // but otherwise behave like &[Vector]
    unsafe impl<V: VectorInfo, const SIZE: usize> Vectorized<V> for [V; SIZE] {
        type Element = V;
        type ElementRef<'result> = V;
        type Slice<'result> = AlignedData<'result, V>;
        type Unaligned = Self;
        type Aligned = Self;
    }
    //
    unsafe impl<V: VectorInfo, const SIZE: usize> VectorizedImpl<V> for [V; SIZE] {
        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _is_last: bool) -> V {
            unsafe { *<[V]>::get_unchecked(&self[..], idx) }
        }

        #[inline(always)]
        fn as_slice(&mut self) -> AlignedData<V> {
            (&self[..]).into()
        }

        unsafe fn as_unaligned_unchecked(self) -> Self {
            self
        }

        unsafe fn as_aligned_unchecked(self) -> Self {
            self
        }
    }

    /// Write access to aligned SIMD data
    //
    // --- Internal docs start here ---
    //
    // Base pointer of an `&mut [Vector]` slice, tagged with lifetime information
    pub struct AlignedDataMut<'target, V: VectorInfo>(
        AlignedData<'target, V>,
        PhantomData<&'target mut [V]>,
    );
    //
    impl<'target, V: VectorInfo> From<&'target mut [V]> for AlignedDataMut<'target, V> {
        fn from(data: &'target mut [V]) -> Self {
            Self(
                unsafe { AlignedData::from_data_ptr(NonNull::from(data)) },
                PhantomData,
            )
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for AlignedDataMut<'target, V> {
        type Element = &'target mut V;
        type ElementRef<'result> = &'result mut V where Self: 'result;
        type Slice<'result> = AlignedDataMut<'result, V> where Self: 'result;
        type Unaligned = Self;
        type Aligned = Self;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for AlignedDataMut<'target, V> {
        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _is_last: bool) -> &mut V {
            unsafe { self.0.get_ptr(idx).as_mut() }
        }

        #[inline(always)]
        fn as_slice(&mut self) -> AlignedDataMut<V> {
            AlignedDataMut(self.0.as_slice(), PhantomData)
        }

        unsafe fn as_unaligned_unchecked(self) -> Self {
            self
        }

        unsafe fn as_aligned_unchecked(self) -> Self {
            self
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for AlignedDataMut<'target, V> {
        #[inline(always)]
        unsafe fn split_at_unchecked(self, mid: usize, len: usize) -> (Self, Self) {
            let (left, right) = self.0.split_at_unchecked(mid, len);
            let wrap = |inner| Self(inner, PhantomData);
            (wrap(left), wrap(right))
        }
    }

    /// Read access to unaligned SIMD data
    //
    // --- Internal docs start here ---
    //
    // Base pointer of an unaligned `&[Vector]` slice, tagged with lifetime
    // information. Built out of a `&[Scalar]` slice. Usable length is the
    // number of complete SIMD vectors within the underlying scalar slice.
    #[derive(Copy, Clone)]
    pub struct UnalignedData<'target, V: VectorInfo>(NonNull<V::Array>, PhantomData<&'target [V]>);
    //
    impl<'target, V: VectorInfo> From<&'target [V::Scalar]> for UnalignedData<'target, V> {
        fn from(data: &'target [V::Scalar]) -> Self {
            unsafe { Self::from_data_ptr(NonNull::from(data)) }
        }
    }
    //
    impl<'target, V: VectorInfo> UnalignedData<'target, V> {
        /// Construct from raw pointer to a slice of data
        ///
        /// # Safety
        ///
        /// It must be valid to access the slice behind `ptr` for lifetime 'target
        unsafe fn from_data_ptr(ptr: NonNull<[V::Scalar]>) -> Self {
            Self(ptr.cast::<V::Array>(), PhantomData)
        }

        /// Base pointer used by get_unchecked(idx)
        ///
        /// # Safety
        ///
        /// `idx` must be in range for the surrounding slice
        #[inline(always)]
        unsafe fn get_ptr(&self, idx: usize) -> NonNull<V::Array> {
            unsafe { NonNull::new_unchecked(self.0.as_ptr().add(idx)) }
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for UnalignedData<'target, V> {
        type Element = V;
        type ElementRef<'result> = V where Self: 'result;
        type Slice<'result> = UnalignedData<'result, V> where Self: 'result;
        type Unaligned = Self;
        type Aligned = AlignedData<'target, V>;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for UnalignedData<'target, V> {
        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _is_last: bool) -> V {
            unsafe { *self.get_ptr(idx).as_ref() }.into()
        }

        #[inline(always)]
        fn as_slice(&mut self) -> UnalignedData<V> {
            UnalignedData(self.0, PhantomData)
        }

        unsafe fn as_unaligned_unchecked(self) -> Self {
            self
        }

        unsafe fn as_aligned_unchecked(self) -> AlignedData<'target, V> {
            V::assert_overaligned_array();
            debug_assert_eq!(
                self.0.as_ptr() as usize % core::mem::align_of::<V>(),
                0,
                "Asked to treat an unaligned pointer as aligned, which is Undefined Behavior"
            );
            AlignedData(self.0.cast::<V>(), PhantomData)
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for UnalignedData<'target, V> {
        #[inline(always)]
        unsafe fn split_at_unchecked(self, mid: usize, _len: usize) -> (Self, Self) {
            let wrap = |ptr| Self(ptr, PhantomData);
            (wrap(self.0), wrap(self.get_ptr(mid)))
        }
    }

    /// Write access to unaligned SIMD data
    //
    // --- Internal docs start here ---
    //
    // Base pointer of an unaligned `&mut [Vector]` slice, tagged with lifetime
    // information. Built out of a `&mut [Scalar]` slice. Usable length is
    // the number of complete SIMD vectors within the underlying scalar slice.
    pub struct UnalignedDataMut<'target, V: VectorInfo>(
        UnalignedData<'target, V>,
        PhantomData<&'target mut [V]>,
    );
    //
    impl<'target, V: VectorInfo> From<&'target mut [V::Scalar]> for UnalignedDataMut<'target, V> {
        fn from(data: &'target mut [V::Scalar]) -> Self {
            Self(
                unsafe { UnalignedData::from_data_ptr(NonNull::from(data)) },
                PhantomData,
            )
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for UnalignedDataMut<'target, V> {
        type Element = Self::ElementRef<'target>;
        type ElementRef<'result> = UnalignedMut<'result, V> where Self: 'result;
        type Slice<'result> = UnalignedDataMut<'result, V> where Self: 'result;
        type Unaligned = Self;
        type Aligned = AlignedDataMut<'target, V>;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for UnalignedDataMut<'target, V> {
        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _is_last: bool) -> UnalignedMut<V> {
            let target = self.0.get_ptr(idx).as_mut();
            let vector = V::from(*target);
            UnalignedMut { vector, target }
        }

        #[inline(always)]
        fn as_slice(&mut self) -> UnalignedDataMut<V> {
            UnalignedDataMut(self.0.as_slice(), PhantomData)
        }

        unsafe fn as_unaligned_unchecked(self) -> Self {
            self
        }

        unsafe fn as_aligned_unchecked(self) -> AlignedDataMut<'target, V> {
            AlignedDataMut(self.0.as_aligned_unchecked(), PhantomData)
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for UnalignedDataMut<'target, V> {
        #[inline(always)]
        unsafe fn split_at_unchecked(self, mid: usize, len: usize) -> (Self, Self) {
            let (left, right) = self.0.split_at_unchecked(mid, len);
            let wrap = |inner| Self(inner, PhantomData);
            (wrap(left), wrap(right))
        }
    }

    /// Vector mutation proxy for unaligned SIMD data
    ///
    /// For mutation from &mut [Scalar], even if the number of elements is a
    /// multiple of the number of vector lanes, we can't provide an &mut Vector
    /// as it could be misaligned. So we provide a proxy object that acts as
    /// closely to &mut Vector as possible.
    pub struct UnalignedMut<'target, V: VectorInfo> {
        vector: V,
        target: &'target mut V::Array,
    }
    //
    impl<V: VectorInfo> Borrow<V> for UnalignedMut<'_, V> {
        #[inline(always)]
        fn borrow(&self) -> &V {
            &self.vector
        }
    }
    //
    impl<V: VectorInfo> BorrowMut<V> for UnalignedMut<'_, V> {
        #[inline(always)]
        fn borrow_mut(&mut self) -> &mut V {
            &mut self.vector
        }
    }
    //
    impl<V: VectorInfo> Deref for UnalignedMut<'_, V> {
        type Target = V;

        #[inline(always)]
        fn deref(&self) -> &V {
            &self.vector
        }
    }
    //
    impl<V: VectorInfo> DerefMut for UnalignedMut<'_, V> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut V {
            &mut self.vector
        }
    }
    //
    impl<V: VectorInfo> Drop for UnalignedMut<'_, V> {
        #[inline(always)]
        fn drop(&mut self) {
            *self.target = self.vector.into()
        }
    }

    /// Read access to padded scalar data
    //
    // --- Internal docs start here ---
    //
    // Base pointer to an unaligned &[Vector] slice with a vector that will be
    // emitted in place of (possibly out-of-bounds) slice data when the last
    // element is requested.
    #[derive(Copy, Clone)]
    pub struct PaddedData<'target, V: VectorInfo> {
        vectors: UnalignedData<'target, V>,
        last_vector: MaybeUninit<V>,
    }
    //
    impl<'target, V: VectorInfo> PaddedData<'target, V> {
        /// Build from a scalar slice and padding data
        ///
        /// In addition to Self, the number of elements in the last vector that
        /// actually originated from the scalar slice is also returned.
        ///
        /// # Errors
        ///
        /// - `NeedsPadding` if padding was needed but not provided
        fn new(
            data: &'target [V::Scalar],
            padding: Option<V::Scalar>,
        ) -> Result<(Self, usize), VectorizeError> {
            // Start by treating most of the slice as unaligned SIMD data
            let mut vectors = UnalignedData::from(data);

            // Decide what the last vector of the simulated slice will be
            let tail_len = data.len() % V::LANES;
            let (last_vector, last_elems) = if tail_len == 0 {
                // If this slice does not actually need padding, last_vector is
                // just the last vector of slice elements, if any
                if data.is_empty() {
                    (MaybeUninit::uninit(), 0)
                } else {
                    (
                        MaybeUninit::new(unsafe {
                            vectors.get_unchecked(data.len() / V::LANES - 1, true)
                        }),
                        V::LANES,
                    )
                }
            } else {
                // Otherwise, last_vector is the last partial vector of slice
                // elements, completed with supplied padding data
                let Some(padding) = padding else { return Err(VectorizeError::NeedsPadding) };
                let last_start = data.len() - tail_len;
                let last_vector =
                    V::from_fn(|idx| data.get(last_start + idx).copied().unwrap_or(padding));
                (MaybeUninit::new(last_vector), tail_len)
            };
            Ok((
                Self {
                    vectors,
                    last_vector,
                },
                last_elems,
            ))
        }

        /// Make an empty slice
        #[inline]
        fn empty() -> Self {
            Self {
                vectors: UnalignedData::from(&[][..]),
                last_vector: MaybeUninit::uninit(),
            }
        }

        /// Base pointer used by get_unchecked(idx)
        ///
        /// # Safety
        ///
        /// `idx` must be in range for the surrounding slice
        #[inline(always)]
        unsafe fn get_ptr(&self, idx: usize) -> NonNull<V::Array> {
            unsafe { self.vectors.get_ptr(idx) }
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for PaddedData<'target, V> {
        type Element = V;
        type ElementRef<'result> = V where Self: 'result;
        type Slice<'result> = PaddedData<'result, V> where Self: 'result;
        type Unaligned = UnalignedData<'target, V>;
        type Aligned = AlignedData<'target, V>;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for PaddedData<'target, V> {
        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, is_last: bool) -> V {
            if is_last {
                unsafe { self.last_vector.assume_init() }
            } else {
                unsafe { self.vectors.get_unchecked(idx, false) }
            }
        }

        #[inline(always)]
        fn as_slice(&mut self) -> PaddedData<V> {
            PaddedData {
                vectors: self.vectors.as_slice(),
                last_vector: self.last_vector,
            }
        }

        unsafe fn as_unaligned_unchecked(self) -> UnalignedData<'target, V> {
            self.vectors
        }

        unsafe fn as_aligned_unchecked(self) -> AlignedData<'target, V> {
            unsafe { self.vectors.as_aligned_unchecked() }
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for PaddedData<'target, V> {
        #[inline(always)]
        unsafe fn split_at_unchecked(mut self, mid: usize, len: usize) -> (Self, Self) {
            if mid < len {
                let left_last_vector = self.vectors.get_unchecked(mid, mid == len - 1);
                let (left_vectors, right_vectors) = self.vectors.split_at_unchecked(mid, len);
                let wrap = |vectors, last_vector| Self {
                    vectors,
                    last_vector,
                };
                (
                    wrap(left_vectors, MaybeUninit::new(left_last_vector)),
                    wrap(right_vectors, self.last_vector),
                )
            } else {
                (self, Self::empty())
            }
        }
    }

    // NOTE: Can't implement support for [Scalar; SIZE] yet due to const
    //       generics limitations (Vectorized::Aligned should be
    //       [Vector; { SIZE / Vector::LANES }], but array lengths derived from
    //       generic parameters are not allowed yet).

    /// Write access to padded scalar data
    //
    // --- Internal docs start here ---
    //
    // Base pointer to an unaligned &mut [Vector] slice with a vector that will
    // be emitted in place of (possibly out-of-bounds) slice data when the last
    // element is requested.
    pub struct PaddedDataMut<'target, V: VectorInfo> {
        inner: PaddedData<'target, V>,
        num_last_elems: usize,
        lifetime: PhantomData<&'target mut [V]>,
    }
    //
    impl<'target, V: VectorInfo> PaddedDataMut<'target, V> {
        /// Build from a scalar slice and padding data
        ///
        /// # Errors
        ///
        /// - `NeedsPadding` if padding was needed but not provided
        fn new(
            data: &'target mut [V::Scalar],
            padding: Option<V::Scalar>,
        ) -> Result<Self, VectorizeError> {
            let (inner, num_last_elems) = PaddedData::new(data, padding)?;
            Ok(Self {
                inner,
                num_last_elems,
                lifetime: PhantomData,
            })
        }

        /// Make an empty slice
        #[inline]
        fn empty() -> Self {
            Self {
                inner: PaddedData::empty(),
                num_last_elems: 0,
                lifetime: PhantomData,
            }
        }

        /// Number of actually stored scalar elements for a vector whose index
        /// is in range, knowing if it's the last one or not
        #[inline(always)]
        fn num_elems(&self, is_last: bool) -> usize {
            if is_last {
                self.num_last_elems
            } else {
                V::LANES
            }
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for PaddedDataMut<'target, V> {
        type Element = PaddedMut<'target, V>;
        type ElementRef<'result> = PaddedMut<'result, V> where Self: 'result;
        type Slice<'result> = PaddedDataMut<'result, V> where Self: 'result;
        type Unaligned = UnalignedDataMut<'target, V>;
        type Aligned = AlignedDataMut<'target, V>;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for PaddedDataMut<'target, V> {
        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, is_last: bool) -> PaddedMut<V> {
            PaddedMut {
                vector: self.inner.get_unchecked(idx, is_last),
                target: core::slice::from_raw_parts_mut(
                    self.inner.get_ptr(idx).cast::<V::Scalar>().as_ptr(),
                    self.num_elems(is_last),
                ),
            }
        }

        #[inline(always)]
        fn as_slice(&mut self) -> PaddedDataMut<V> {
            PaddedDataMut {
                inner: self.inner.as_slice(),
                num_last_elems: self.num_last_elems,
                lifetime: PhantomData,
            }
        }

        unsafe fn as_unaligned_unchecked(self) -> UnalignedDataMut<'target, V> {
            unsafe { UnalignedDataMut(self.inner.as_unaligned_unchecked(), PhantomData) }
        }

        unsafe fn as_aligned_unchecked(self) -> AlignedDataMut<'target, V> {
            unsafe { AlignedDataMut(self.inner.as_aligned_unchecked(), PhantomData) }
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for PaddedDataMut<'target, V> {
        #[inline(always)]
        unsafe fn split_at_unchecked(self, mid: usize, len: usize) -> (Self, Self) {
            if mid < len {
                let left_num_last_elems = self.num_elems(mid == len - 1);
                let (left_inner, right_inner) = self.inner.split_at_unchecked(mid, len);
                let wrap = |inner, num_last_elems| Self {
                    inner,
                    num_last_elems,
                    lifetime: PhantomData,
                };
                (
                    wrap(left_inner, left_num_last_elems),
                    wrap(right_inner, self.num_last_elems),
                )
            } else {
                (self, Self::empty())
            }
        }
    }

    /// Vector mutation proxy for padded scalar slices
    ///
    /// For mutation from &mut [Scalar], we can't provide an &mut Vector as it
    /// could be misaligned and out of bounds. So we provide a proxy object
    /// that acts as closely to &mut Vector as possible.
    pub struct PaddedMut<'target, V: VectorInfo> {
        vector: V,
        target: &'target mut [V::Scalar],
    }
    //
    impl<V: VectorInfo> Borrow<V> for PaddedMut<'_, V> {
        #[inline(always)]
        fn borrow(&self) -> &V {
            &self.vector
        }
    }
    //
    impl<V: VectorInfo> BorrowMut<V> for PaddedMut<'_, V> {
        #[inline(always)]
        fn borrow_mut(&mut self) -> &mut V {
            &mut self.vector
        }
    }
    //
    impl<V: VectorInfo> Deref for PaddedMut<'_, V> {
        type Target = V;

        #[inline(always)]
        fn deref(&self) -> &V {
            &self.vector
        }
    }
    //
    impl<V: VectorInfo> DerefMut for PaddedMut<'_, V> {
        #[inline(always)]
        fn deref_mut(&mut self) -> &mut V {
            &mut self.vector
        }
    }
    //
    impl<V: VectorInfo> Drop for PaddedMut<'_, V> {
        #[inline(always)]
        fn drop(&mut self) {
            self.target
                .copy_from_slice(&self.vector.as_ref()[..self.target.len()]);
        }
    }

    /// Tuples of pointers yield tuples of deref results
    macro_rules! impl_vector_slice_base_for_tuple {
        (
            $($t:ident),*
        ) => {
            unsafe impl<
                'target,
                V: VectorInfo
                $(, $t: Vectorized<V> + 'target)*
            > Vectorized<V> for ($($t,)*) {
                type Element = ($($t::Element,)*);
                type ElementRef<'result> = ($($t::ElementRef<'result>,)*) where Self: 'result;
                type Slice<'result> = ($($t::Slice<'result>,)*) where Self: 'result;
                type Unaligned = ($($t::Unaligned,)*);
                type Aligned = ($($t::Aligned,)*);
            }

            #[allow(non_snake_case)]
            unsafe impl<
                'target,
                V: VectorInfo
                $(, $t: VectorizedImpl<V> + 'target)*
            > VectorizedImpl<V> for ($($t,)*) {
                #[inline(always)]
                unsafe fn get_unchecked(
                    &mut self,
                    idx: usize,
                    is_last: bool
                ) -> Self::ElementRef<'_> {
                    let ($($t,)*) = self;
                    unsafe { ($($t.get_unchecked(idx, is_last),)*) }
                }

                #[inline(always)]
                fn as_slice(&mut self) -> Self::Slice<'_> {
                    let ($($t,)*) = self;
                    ($($t.as_slice(),)*)
                }

                unsafe fn as_unaligned_unchecked(self) -> Self::Unaligned {
                    let ($($t,)*) = self;
                    unsafe { ($($t.as_unaligned_unchecked(),)*) }
                }

                unsafe fn as_aligned_unchecked(self) -> Self::Aligned {
                    let ($($t,)*) = self;
                    unsafe { ($($t.as_aligned_unchecked(),)*) }
                }
            }

            #[allow(non_snake_case)]
            unsafe impl<
                'target,
                V: VectorInfo
                $(, $t: VectorizedSliceImpl<V> + 'target)*
            > VectorizedSliceImpl<V> for ($($t,)*) {
                #[inline(always)]
                unsafe fn split_at_unchecked(
                    self,
                    mid: usize,
                    len: usize,
                ) -> (Self, Self) {
                    let ($($t,)*) = self;
                    let ($($t,)*) = unsafe { ($($t.split_at_unchecked(mid, len),)*) };
                    (($($t.0,)*), ($($t.1,)*))
                }
            }
        };
    }
    impl_vector_slice_base_for_tuple!(A);
    impl_vector_slice_base_for_tuple!(A, B);
    impl_vector_slice_base_for_tuple!(A, B, C);
    impl_vector_slice_base_for_tuple!(A, B, C, D);
    impl_vector_slice_base_for_tuple!(A, B, C, D, E);
    impl_vector_slice_base_for_tuple!(A, B, C, D, E, F);
    impl_vector_slice_base_for_tuple!(A, B, C, D, E, F, G);
    impl_vector_slice_base_for_tuple!(A, B, C, D, E, F, G, H);

    // === Step 2: Optimized vector data container ===

    /// Data that can be processed using SIMD
    ///
    /// This container is built using the `Vectorizable` trait.
    ///
    /// It behaves conceptually like an array of `Vector` or tuples thereof,
    /// with iteration and indexing operations yielding the type that is
    /// described in the documentation of `Vectorizable`.
    #[derive(Copy, Clone)]
    pub struct Vectors<V: VectorInfo, Data: VectorizedImpl<V>> {
        data: Data,
        len: usize,
        vectors: PhantomData<V>,
    }
    //
    impl<V: VectorInfo, Data: VectorizedImpl<V>> Vectors<V, Data> {
        /// Create a SIMD data container
        ///
        /// # Safety
        ///
        /// - It must be safe to call `data.get_unchecked(idx)` for any index in
        ///   range 0..len
        unsafe fn from_raw_parts(data: Data, len: usize) -> Self {
            Self {
                data,
                len,
                vectors: PhantomData,
            }
        }

        /// Returns the number of elements in the container
        #[inline]
        pub const fn len(&self) -> usize {
            self.len
        }

        /// Returns `true` if there are no elements in the container
        #[inline]
        pub const fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Returns the first element, or None if the container is empty
        #[inline]
        pub fn first(&mut self) -> Option<Data::ElementRef<'_>> {
            self.get(0)
        }

        /// Returns the first and all the rest of the elements of the container,
        /// or None if it is empty.
        #[inline(always)]
        pub fn split_first(&mut self) -> Option<(Data::ElementRef<'_>, Slice<V, Data>)> {
            (!self.is_empty()).then(move || {
                let (head, tail) = unsafe { self.as_slice().split_at_unchecked(1) };
                (head.into_iter().next().unwrap(), tail)
            })
        }

        /// Returns the last and all the rest of the elements of the container,
        /// or None if it is empty.
        #[inline(always)]
        pub fn split_last(&mut self) -> Option<(Data::ElementRef<'_>, Slice<V, Data>)> {
            (!self.is_empty()).then(move || {
                let last = self.last_idx();
                let (head, tail) = unsafe { self.as_slice().split_at_unchecked(last) };
                (tail.into_iter().next().unwrap(), head)
            })
        }

        /// Returns the last element, or None if the container is empty
        #[inline]
        pub fn last(&mut self) -> Option<Data::ElementRef<'_>> {
            self.get(self.last_idx())
        }

        /// Index of the last element
        #[inline(always)]
        fn last_idx(&self) -> usize {
            self.len - 1
        }

        /// Like get(), but panics if index is out of range
        ///
        /// # Panics
        ///
        /// If index is out of range
        //
        // NOTE: We can't implement the real Index trait because it requires
        //       returning a &V that we don't have for padded/unaligned data.
        #[inline(always)]
        pub fn index<I>(&mut self, index: I) -> <I as VectorIndex<V, Data>>::Output<'_>
        where
            I: VectorIndex<V, Data>,
        {
            self.get(index).expect("Index is out of range")
        }

        /// Returns the specified element(s) of the container
        ///
        /// This operation accepts either a single `usize` index or a range of
        /// `usize` indices:
        ///
        /// - Given a single index, it emits `Data::ElementRef<'_>`.
        /// - Given a range of indices, it emits `Data::Slice<'_>`.
        ///
        /// If one or more of the specified indices is out of range, None is
        /// returned.
        #[inline(always)]
        pub fn get<I>(&mut self, index: I) -> Option<<I as VectorIndex<V, Data>>::Output<'_>>
        where
            I: VectorIndex<V, Data>,
        {
            (index.is_valid_index(self)).then(move || unsafe { self.get_unchecked(index) })
        }

        /// Returns the specified element(s) of the container without bounds
        /// checking
        ///
        /// # Safety
        ///
        /// Indices covered by `index` must be in range `0..self.len()`
        #[inline(always)]
        pub unsafe fn get_unchecked<I>(
            &mut self,
            index: I,
        ) -> <I as VectorIndex<V, Data>>::Output<'_>
        where
            I: VectorIndex<V, Data>,
        {
            unsafe { index.get_unchecked(self) }
        }

        /// Returns an iterator over contained elements
        #[inline]
        pub fn iter(&mut self) -> Iter<V, Data> {
            <&mut Self>::into_iter(self)
        }

        // TODO: chunks(_exact)? : mark inline

        /// Returns an iterator over N elements at a time, starting at the
        /// beginning of the container
        // TODO: Make a dedicated Iterator so I can implement DoubleEnded + ExactSize + Fused
        //       and add a remainder
        #[inline]
        pub fn array_chunks<const N: usize>(
            &mut self,
        ) -> impl Iterator<Item = [Data::ElementRef<'_>; N]> {
            let mut iter = self.iter();
            core::iter::from_fn(move || {
                if iter.len() >= N {
                    Some(array_from_fn(|_| iter.next().unwrap()))
                } else {
                    None
                }
            })
        }

        // TODO: rchunks(_exact)? : mark inline

        /// Extract a slice containing the entire dataset
        ///
        /// Equivalent to `self.index(..)`
        #[inline]
        pub fn as_slice(&mut self) -> Slice<V, Data> {
            unsafe { Vectors::from_raw_parts(self.data.as_slice(), self.len) }
        }

        /// Divides a slice into two at an index
        ///
        /// The first will contain all indices from `[0, mid)` (excluding the
        /// index `mid` itself) and the second will contain all indices from
        /// `[mid, len)` (excluding the index `len` itself).
        ///
        /// # Panics
        ///
        /// Panics if mid > len
        #[inline]
        pub fn split_at(self, mid: usize) -> (Self, Self)
        where
            Data: VectorizedSliceImpl<V>,
        {
            assert!(mid <= self.len(), "Split point is out of range");
            unsafe { self.split_at_unchecked(mid) }
        }

        /// Divides a slice into two at an index, without doing bounds checking
        ///
        /// The first will contain all indices from `[0, mid)` (excluding the
        /// index `mid` itself) and the second will contain all indices from
        /// `[mid, len)` (excluding the index `len` itself).
        ///
        /// For a safe alternative, see [`split_at`](Vectors::split_at).
        ///
        /// # Safety
        ///
        /// Calling this method with an out-of-bounds index is undefined
        /// behavior even if the resulting reference is not used. The caller has
        /// to ensure that 0 <= mid <= self.len().
        #[inline(always)]
        pub unsafe fn split_at_unchecked(self, mid: usize) -> (Self, Self)
        where
            Data: VectorizedSliceImpl<V>,
        {
            let total_len = self.len();
            let (left_data, right_data) = unsafe { self.data.split_at_unchecked(mid, total_len) };
            let wrap = |data, len| unsafe { Vectors::from_raw_parts(data, len) };
            (wrap(left_data, mid), wrap(right_data, total_len - mid))
        }

        // TODO: Figure out inlining discipline for the following
        // TODO: r?split(_inclusive)?, r?splitn,
        //       contains, (starts|ends)_with,
        //       (sort|select_nth|binary_search)(_unstable)?_by((_cached)?_key)?,
        //       copy_from_slice (optimiser !), partition_point
    }

    /// A helper trait used for indexing operations
    ///
    /// Analogous to the standard (unstable)
    /// [`SliceIndex`](core::slice::SliceIndex) trait, but for `Vectors`.
    ///
    /// # Safety
    ///
    /// Unsafe code can rely on this trait being implemented correctly
    pub unsafe trait VectorIndex<V: VectorInfo, Data: VectorizedImpl<V>> {
        /// The output type returned by methods
        type Output<'out>
        where
            Self: 'out,
            Data: 'out;

        /// Truth that `self` is a valid index for `vectors`
        fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool;

        /// Perform unchecked indexing
        ///
        /// # Safety
        ///
        /// `self` must be a valid index for `vectors` (see is_valid_index)
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_>;
    }

    unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for usize {
        type Output<'out> = Data::ElementRef<'out> where Self: 'out, Data: 'out;

        #[inline(always)]
        fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
            *self < vectors.len()
        }

        #[inline(always)]
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_> {
            unsafe { vectors.data.get_unchecked(self, self == vectors.last_idx()) }
        }
    }

    unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for RangeFull {
        type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

        #[inline]
        fn is_valid_index(&self, _vectors: &Vectors<V, Data>) -> bool {
            true
        }

        #[inline]
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_> {
            vectors.as_slice()
        }
    }

    unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for RangeFrom<usize> {
        type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

        #[inline]
        fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
            self.start < vectors.len()
        }

        #[inline]
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_> {
            unsafe { vectors.as_slice().split_at_unchecked(self.start) }.1
        }
    }

    unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for RangeTo<usize> {
        type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

        #[inline]
        fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
            self.end <= vectors.len()
        }

        #[inline]
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_> {
            unsafe { vectors.as_slice().split_at_unchecked(self.end) }.0
        }
    }

    unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for Range<usize> {
        type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

        #[inline]
        fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
            if self.start < self.end {
                self.start < vectors.len() && self.end <= vectors.len()
            } else {
                true
            }
        }

        #[inline]
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_> {
            if self.start < self.end {
                let after_start = unsafe { vectors.as_slice().split_at_unchecked(self.start) }.1;
                unsafe { after_start.split_at_unchecked(self.end - self.start) }.0
            } else {
                unsafe { vectors.as_slice().split_at_unchecked(0) }.0
            }
        }
    }

    unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for RangeInclusive<usize> {
        type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

        #[inline]
        fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
            let (&start, &end) = (self.start(), self.end());
            if end == usize::MAX {
                false
            } else {
                (start..end + 1).is_valid_index(vectors)
            }
        }

        #[inline]
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_> {
            let (&start, &end) = (self.start(), self.end());
            if end == usize::MAX {
                core::hint::unreachable_unchecked()
            } else {
                (start..end + 1).get_unchecked(vectors)
            }
        }
    }

    unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data>
        for RangeToInclusive<usize>
    {
        type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

        #[inline]
        fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
            (0..=self.end).is_valid_index(vectors)
        }

        #[inline]
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_> {
            (0..=self.end).get_unchecked(vectors)
        }
    }

    unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data>
        for (Bound<usize>, Bound<usize>)
    {
        type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

        #[inline]
        fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
            let (lower_excluded, upper) = match (self.0, self.1) {
                (Bound::Included(s), Bound::Included(e)) => return (s..=e).is_valid_index(vectors),
                (Bound::Included(s), Bound::Excluded(e)) => return (s..e).is_valid_index(vectors),
                (Bound::Included(s), Bound::Unbounded) => return (s..).is_valid_index(vectors),
                (Bound::Unbounded, Bound::Included(e)) => return (..=e).is_valid_index(vectors),
                (Bound::Unbounded, Bound::Excluded(e)) => return (..e).is_valid_index(vectors),
                (Bound::Unbounded, Bound::Unbounded) => return (..).is_valid_index(vectors),
                (Bound::Excluded(s), upper) => (s, upper),
            };
            let lower_included = if lower_excluded == usize::MAX {
                return false;
            } else {
                lower_excluded + 1
            };
            (Bound::Included(lower_included), upper).is_valid_index(vectors)
        }

        #[inline]
        unsafe fn get_unchecked(self, vectors: &mut Vectors<V, Data>) -> Self::Output<'_> {
            let (lower_excluded, upper) = match (self.0, self.1) {
                (Bound::Included(s), Bound::Included(e)) => return (s..=e).get_unchecked(vectors),
                (Bound::Included(s), Bound::Excluded(e)) => return (s..e).get_unchecked(vectors),
                (Bound::Included(s), Bound::Unbounded) => return (s..).get_unchecked(vectors),
                (Bound::Unbounded, Bound::Included(e)) => return (..=e).get_unchecked(vectors),
                (Bound::Unbounded, Bound::Excluded(e)) => return (..e).get_unchecked(vectors),
                (Bound::Unbounded, Bound::Unbounded) => return (..).get_unchecked(vectors),
                (Bound::Excluded(s), upper) => (s, upper),
            };
            let lower_included = if lower_excluded == usize::MAX {
                core::hint::unreachable_unchecked()
            } else {
                lower_excluded + 1
            };
            (Bound::Included(lower_included), upper).get_unchecked(vectors)
        }
    }

    macro_rules! impl_iterator {
        (
            $(#[$attr:meta])*
            ($name:ident, Data::$elem:ident$(<$lifetime:lifetime>)?)
        ) => {
            impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V> $( + $lifetime)?> IntoIterator
                for $(&$lifetime mut)? Vectors<V, Data>
            {
                type Item = Data::$elem $(<$lifetime>)?;
                type IntoIter = $name<$($lifetime,)? V, Data>;

                #[inline]
                fn into_iter(self) -> Self::IntoIter {
                    let end = self.len;
                    Self::IntoIter {
                        vectors: self,
                        start: 0,
                        end,
                    }
                }
            }

            $(#[$attr])*
            pub struct $name<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> {
                vectors: $(&$lifetime mut)? Vectors<V, Data>,
                start: usize,
                end: usize,
            }
            //
            impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> $name<$($lifetime,)? V, Data> {
                /// Get the i-th element
                ///
                /// # Safety
                ///
                /// - This should only be called once per index, i.e. you must ensure
                ///   that the iterator will not visit this index again.
                /// - This should only be called for valid indices of the underlying Vectors.
                #[inline(always)]
                unsafe fn get_elem<'iter>(&'iter mut self, idx: usize) -> Data::$elem$(<$lifetime>)? {
                    debug_assert!(idx < self.vectors.len());
                    let result = unsafe { self.vectors.get_unchecked(idx) };
                    unsafe { core::mem::transmute_copy::<Data::ElementRef<'iter>, Data::$elem$(<$lifetime>)?>(&result) }
                }

                $(
                    /// Views the underlying data as a subslice of the original
                    /// data.
                    ///
                    /// To avoid creating &mut [T] references that alias, this
                    /// is forced to consume the iterator
                    #[inline]
                    pub fn into_slice(self) -> Slice<$lifetime, V, Data> {
                        unsafe { self.vectors.get_unchecked(self.start..self.end) }
                    }
                )?

                /// Views the underlying data as a subslice of the original data.
                ///
                /// To avoid creating &mut [T] references that alias, the
                /// returned slice borrows its lifetime from the iterator the
                /// method is applied on.
                #[inline]
                pub fn as_slice(&mut self) -> Slice<V, Data> {
                    unsafe { self.vectors.get_unchecked(self.start..self.end) }
                }
            }
            //
            impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> Iterator
                for $name<$($lifetime,)? V, Data>
            {
                type Item = Data::$elem$(<$lifetime>)?;

                #[inline(always)]
                fn next<'iter>(&'iter mut self) -> Option<Self::Item> {
                    if self.start < self.end {
                        self.start += 1;
                        Some(unsafe { self.get_elem(self.start - 1) })
                    } else {
                        None
                    }
                }

                #[inline]
                fn size_hint(&self) -> (usize, Option<usize>) {
                    (self.len(), Some(self.len()))
                }

                #[inline]
                fn count(self) -> usize {
                    self.len()
                }

                #[inline]
                fn nth(&mut self, n: usize) -> Option<Self::Item> {
                    if self.start.checked_add(n)? < self.end {
                        self.start += n - 1;
                        self.next()
                    } else {
                        None
                    }
                }
            }
            //
            impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> DoubleEndedIterator
                for $name<$($lifetime,)? V, Data>
            {
                #[inline(always)]
                fn next_back(&mut self) -> Option<Self::Item> {
                    if self.start < self.end {
                        self.end -= 1;
                        Some(unsafe { self.get_elem(self.end + 1) })
                    } else {
                        None
                    }
                }

                #[inline]
                fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
                    if self.start < self.end.checked_sub(n)? {
                        self.end -= n - 1;
                        self.next_back()
                    } else {
                        None
                    }
                }
            }
            //
            impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> ExactSizeIterator
                for $name<$($lifetime,)? V, Data>
            {
                #[inline]
                fn len(&self) -> usize {
                    self.end - self.start
                }
            }
            //
            impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> FusedIterator
                for $name<$($lifetime,)? V, Data>
            {
            }
            //
            #[cfg(feature = "iterator_ilp")]
            unsafe impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> iterator_ilp::TrustedLowerBound
                for $name<$($lifetime,)? V, Data>
            {
            }
        }
    }
    impl_iterator!(
        /// Borrowing iterator over Vectors' elements
        (Iter, Data::ElementRef<'vectors>)
    );
    impl_iterator!(
        /// Owned iterator over Vectors' elements
        (IntoIter, Data::Element)
    );

    /// Slice of a Vectors container
    pub type Slice<'a, V, Data> = Vectors<V, <Data as Vectorized<V>>::Slice<'a>>;

    /// Aligned SIMD data
    pub type AlignedVectors<V, Data> = Vectors<V, <Data as Vectorized<V>>::Aligned>;

    /// Unaligned SIMD data
    pub type UnalignedVectors<V, Data> = Vectors<V, <Data as Vectorized<V>>::Unaligned>;

    /// Padded scalar data treated as SIMD data
    pub type PaddedVectors<V, Data> = Vectors<V, Data>;

    // === Step 3: Translate from scalar slices and containers to vector slices ===

    /// Trait for data that can be processed using SIMD
    ///
    /// Implemented for slices and containers of vectors and scalars,
    /// as well as for tuples of these entities.
    ///
    /// Provides you with ways to create the `Vectors` collection, which
    /// behaves conceptually like a slice of `Vector` or tuples thereof, with
    /// iteration and indexing operations yielding the following types:
    ///
    /// - If built out of a read-only slice or owned container of vectors or
    ///   scalars, it yields owned `Vector`s of data.
    /// - If built out of `&mut [Vector]`, or `&mut [Scalar]` that is assumed
    ///   to be SIMD-aligned (see below), it yields `&mut Vector` references.
    /// - If built out of `&mut [Scalar]` that is not SIMD-aligned, it yields
    ///   a proxy type which can be used like an `&mut Vector` (but cannot
    ///   literally be `&mut Vector` for alignment and padding reasons)
    /// - If built out of a tuple of the above entities, it yields tuples of the
    ///   aforementioned elements.
    ///
    /// There are three ways to create `Vectors` using this trait depending on
    /// what kind of data you're starting from:
    ///
    /// - If starting out of arbitrary data, you can use the [`vectorize_pad()`]
    ///   method to get a SIMD view that does not make any assumption, but
    ///   tends to exhibit poor performance on scalar data as a result.
    /// - If you know that every scalar slice in your dataset has a number of
    ///   elements that is a multiple of the SIMD vector width, you can use the
    ///   [`vectorize()`] method to get a SIMD view that assumes this (or a
    ///   panic if this is not true), resulting in much better performance.
    /// - If, in addition to the above, you know that every scalar slice in your
    ///   dataset is SIMD-aligned, you can use the [`vectorize_aligned()`]
    ///   method to get a SIMD view that assumes this (or a panic if this is not
    ///   true), which may result in even better performance.
    ///
    /// Note that even on hardware architectures like x86 where SIMD alignment
    /// is not a prerequisite for good code generation (and hence you may not
    /// need to call `vectorize_aligned()` for optimal performance), it is
    /// always a hardware prerequisite for good computational performance, so
    /// you should aim for it whenever possible!
    ///
    /// # Safety
    ///
    /// Unsafe code may rely on the implementation being correct.
    ///
    /// [`vectorize()`]: Vectorizable::vectorize()
    /// [`vectorize_aligned()`]: Vectorizable::vectorize_aligned()
    /// [`vectorize_pad()`]: Vectorizable::vectorize_pad()
    pub unsafe trait Vectorizable<V: VectorInfo>: Sized {
        /// Vectorized representation of this data
        ///
        /// You can use the Vectorized trait to query at compile time which type
        /// of Vectors collections you are going to get and what kind of
        /// elements iterators and getters of this collection will emit.
        ///
        /// VectorizedImpl is an implementation detail of this crate.
        type Vectorized: Vectorized<V> + VectorizedImpl<V>;

        // Required methods

        /// Implementation of the `vectorize()` methods
        //
        // --- Internal docs starts here ---
        //
        // The returned building blocks are...
        //
        // - A pointer-like entity for treating the data as a slice of Vector
        //   (see VectorizedImpl for more information)
        // - The number of Vector elements that the emulated slice contains
        //
        // # Errors
        //
        // - NeedsPadding if padding was needed, but not provided
        // - InhomogeneousLength if input is (or contains) a tuple and not all
        //   tuple elements yield the same amount of SIMD vectors
        fn into_vectorized_parts(
            self,
            padding: Option<V::Scalar>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError>;

        // Provided methods

        /// Create a SIMD view of this data, asserting that its length is a
        /// multiple of the SIMD vector length
        ///
        /// # Panics
        ///
        /// - If called on a scalar slice whose length is not a multiple of the
        ///   number of SIMD vector lanes (you need `vectorize_pad()`)
        /// - If called on a tuple and not all tuple elements yield the same
        ///   amount of SIMD elements.
        fn vectorize(self) -> UnalignedVectors<V, Self::Vectorized> {
            let (base, len) = self.into_vectorized_parts(None).unwrap();
            unsafe { Vectors::from_raw_parts(base.as_unaligned_unchecked(), len) }
        }

        /// Create a SIMD view of this data, providing some padding
        ///
        /// Vector slices do not need padding and will ignore it.
        ///
        /// For scalar slices whose size is not a multiple of the number of SIMD
        /// vector lanes, padding will be inserted where incomplete Vectors
        /// would be produced, to fill in the missing vector lanes. One would
        /// normally set the padding to the neutral element of the computation
        /// being performed so that its presence doesn't affect results.
        ///
        /// The use of padding makes it harder for the compiler to optimize the
        /// code even if the padding ends up not being used, so using this
        /// option will generally result in lower runtime performance.
        ///
        /// # Panics
        ///
        /// - If called on a tuple and not all tuple elements yield the same
        ///   amount of SIMD elements.
        fn vectorize_pad(self, padding: V::Scalar) -> PaddedVectors<V, Self::Vectorized> {
            let (base, len) = self.into_vectorized_parts(Some(padding)).unwrap();
            unsafe { Vectors::from_raw_parts(base, len) }
        }

        /// Create a SIMD view of this data, assert it is (or can be moved to) a
        /// layout optimized for SIMD processing
        ///
        /// Vector data always passes this check, but scalar data only passes it
        /// if it meets two conditions:
        ///
        /// - The start of the data is aligned as `std::mem::align_of::<V>()`,
        ///   or the data can be moved around in memory to enforce this.
        /// - The number of inner scalar elements is a multiple of the number
        ///   of SIMD vector lanes of V.
        ///
        /// If this is true, the data is reinterpreted in a manner that will
        /// simplify the implementation, which should result in less edge cases
        /// where the compiler does not generate good code.
        ///
        /// Furthermore, note that the above are actually _hardware_
        /// requirements for good vectorization: even if the compiler does
        /// generate good code, the resulting binary will perform less well if
        /// the data does not have the above properties. So you should enforce
        /// them whenever possible!
        ///
        /// # Panics
        ///
        /// - If the data is not in a SIMD-optimized layout.
        /// - If called on a tuple and not all tuple elements yield the same
        ///   amount of SIMD elements.
        fn vectorize_aligned(self) -> AlignedVectors<V, Self::Vectorized> {
            let (base, len) = self.into_vectorized_parts(None).unwrap();
            unsafe { Vectors::from_raw_parts(base.as_aligned_unchecked(), len) }
        }
    }

    /// Error returned by `Vectorizable::into_vectorized_parts`
    #[doc(hidden)]
    #[derive(Debug)]
    pub enum VectorizeError {
        /// Padding data was needed, but not provided
        NeedsPadding,

        /// Input contains tuples of data with inhomogeneous SIMD length
        InhomogeneousLength,
    }

    // === Vectorize implementation is trivial for slices of vector data ===

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target [Vector<A, B, S>]
    {
        type Vectorized = AlignedData<'target, Vector<A, B, S>>;

        fn into_vectorized_parts(
            self,
            _padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            Ok((self.into(), self.len()))
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target mut [Vector<A, B, S>]
    {
        type Vectorized = AlignedDataMut<'target, Vector<A, B, S>>;

        fn into_vectorized_parts(
            self,
            _padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            let len = self.len();
            Ok((self.into(), len))
        }
    }

    unsafe impl<A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for [Vector<A, B, S>; ARRAY_SIZE]
    {
        type Vectorized = [Vector<A, B, S>; ARRAY_SIZE];

        fn into_vectorized_parts(
            self,
            _padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            Ok((self, ARRAY_SIZE))
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for &'target [Vector<A, B, S>; ARRAY_SIZE]
    {
        type Vectorized = AlignedData<'target, Vector<A, B, S>>;

        fn into_vectorized_parts(
            self,
            padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            self.as_slice().into_vectorized_parts(padding)
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for &'target mut [Vector<A, B, S>; ARRAY_SIZE]
    {
        type Vectorized = AlignedDataMut<'target, Vector<A, B, S>>;

        fn into_vectorized_parts(
            self,
            padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            self.as_mut_slice().into_vectorized_parts(padding)
        }
    }

    // === For scalar data, must cautiously handle padding and alignment ===

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target [B]
    {
        type Vectorized = PaddedData<'target, Vector<A, B, S>>;

        fn into_vectorized_parts(
            self,
            padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            Ok((
                PaddedData::new(self, padding)?.0,
                self.len() / S + (self.len() % S != 0) as usize,
            ))
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target mut [B]
    {
        type Vectorized = PaddedDataMut<'target, Vector<A, B, S>>;

        fn into_vectorized_parts(
            self,
            padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            let simd_len = self.len() / S + (self.len() % S != 0) as usize;
            Ok((PaddedDataMut::new(self, padding)?, simd_len))
        }
    }

    // NOTE: Cannot be implemented for owned scalar arrays yet due to const
    //       generics limitations around vectorized_aligned().

    unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for &'target [B; ARRAY_SIZE]
    {
        type Vectorized = PaddedData<'target, Vector<A, B, S>>;

        fn into_vectorized_parts(
            self,
            padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            self.as_slice().into_vectorized_parts(padding)
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for &'target mut [B; ARRAY_SIZE]
    {
        type Vectorized = PaddedDataMut<'target, Vector<A, B, S>>;

        fn into_vectorized_parts(
            self,
            padding: Option<B>,
        ) -> Result<(Self::Vectorized, usize), VectorizeError> {
            self.as_mut_slice().into_vectorized_parts(padding)
        }
    }

    // === Tuples must have homogeneous length and vector type ===

    macro_rules! impl_vectorize_for_tuple {
        (
            $($t:ident),*
        ) => {
            #[allow(non_snake_case)]
            unsafe impl<V: VectorInfo $(, $t: Vectorizable<V>)*> Vectorizable<V> for ($($t,)*) {
                type Vectorized = ($($t::Vectorized,)*);

                fn into_vectorized_parts(
                    self,
                    padding: Option<V::Scalar>,
                ) -> Result<(Self::Vectorized, usize), VectorizeError> {
                    // Pattern-match the tuple to variables named after inner types
                    let ($($t,)*) = self;

                    // Reinterpret tuple fields as SIMD vectors
                    let ($($t,)*) = ($($t.into_vectorized_parts(padding)?,)*);

                    // Analyze tuple field lengths and need for padding
                    let mut len = None;
                    $(
                        let (_, t_len) = $t;

                        // All tuple fields should have the same SIMD length
                        #[allow(unused_assignments)]
                        if let Some(len) = len {
                            if len != t_len {
                                return Err(VectorizeError::InhomogeneousLength);
                            }
                            assert_eq!(
                                t_len, len,
                                "Tuple elements do not produce the same amount of SIMD vectors"
                            );
                        } else {
                            len = Some(t_len);
                        }
                    )*

                    // All good, return Vectors building blocks
                    Ok((
                        ($($t.0,)*),
                        len.expect("This should not be implemented for zero-sized tuples"),
                    ))
                }
            }
        };
    }
    impl_vectorize_for_tuple!(A);
    impl_vectorize_for_tuple!(A, B);
    impl_vectorize_for_tuple!(A, B, C);
    impl_vectorize_for_tuple!(A, B, C, D);
    impl_vectorize_for_tuple!(A, B, C, D, E);
    impl_vectorize_for_tuple!(A, B, C, D, E, F);
    impl_vectorize_for_tuple!(A, B, C, D, E, F, G);
    impl_vectorize_for_tuple!(A, B, C, D, E, F, G, H);
}

// TODO: Deref to arrays, not slices
/// A proxy object for iterating over mutable slices.
///
/// For technical reasons (mostly alignment and padding), it's not possible to return a simple
/// reference. This type is returned instead and it can be used to both read and write the vectors
/// a slice is turned into.
///
/// Note that the data are written in the destructor. Usually, this should not matter, but if you
/// [`forget`][mem::forget], the changes will be lost (this is meant as a warning, not as a way to
/// implement poor-man's transactions).
#[derive(Debug)]
pub struct MutProxy<'a, B, V>
where
    V: AsRef<[B]>,
    B: Copy,
{
    data: V,
    restore: &'a mut [B],
}

impl<B, V> Deref for MutProxy<'_, B, V>
where
    V: AsRef<[B]>,
    B: Copy,
{
    type Target = V;
    #[inline]
    fn deref(&self) -> &V {
        &self.data
    }
}

impl<B, V> DerefMut for MutProxy<'_, B, V>
where
    V: AsRef<[B]>,
    B: Copy,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut V {
        &mut self.data
    }
}

impl<B, V> Drop for MutProxy<'_, B, V>
where
    V: AsRef<[B]>,
    B: Copy,
{
    #[inline]
    fn drop(&mut self) {
        self.restore
            .copy_from_slice(&self.data.as_ref()[..self.restore.len()]);
    }
}

#[doc(hidden)]
pub trait Partial<V> {
    fn take_partial(&mut self) -> Option<V>;
    fn size(&self) -> usize;
}

impl<V> Partial<V> for () {
    #[inline]
    fn take_partial(&mut self) -> Option<V> {
        None
    }
    #[inline]
    fn size(&self) -> usize {
        0
    }
}

impl<V> Partial<V> for Option<V> {
    #[inline]
    fn take_partial(&mut self) -> Option<V> {
        Option::take(self)
    }
    fn size(&self) -> usize {
        self.is_some() as usize
    }
}

#[doc(hidden)]
pub trait Vectorizer<R> {
    /// Get the nth vector.
    ///
    /// # Safety
    ///
    /// * idx must be in range (as declared on creation).
    /// * It may be called at most once per each index.
    unsafe fn get(&mut self, idx: usize) -> R;
}

/// The iterator returned by methods on [`Vectorizable`].
///
/// While it's unusual to need to *name* the type, this is the thing that is returned from
/// [`Vectorizable::vectorize`] and [`Vectorizable::vectorize_pad`]. It might be of interest to
/// know that it implements several iterator „extensions“ ([`DoubleEndedIterator`],
/// [`ExactSizeIterator`] and [`FusedIterator`]). Also, several methods are optimized ‒ for
/// example, the `count` is constant time operation, while the generic is linear.
#[derive(Copy, Clone, Debug)]
pub struct VectorizedIter<V, P, R> {
    partial: P,
    vectorizer: V,
    left: usize,
    right: usize,
    _result: PhantomData<R>,
}

impl<V, P, R> Iterator for VectorizedIter<V, P, R>
where
    V: Vectorizer<R>,
    P: Partial<R>,
{
    type Item = R;

    #[inline]
    fn next(&mut self) -> Option<R> {
        if self.left < self.right {
            let idx = self.left;
            self.left += 1;
            Some(unsafe { self.vectorizer.get(idx) })
        } else {
            self.partial.take_partial()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.right - self.left + self.partial.size();
        (len, Some(len))
    }

    // Overriden for performance… these things have no side effects, so we can avoid calling next

    #[inline]
    fn count(self) -> usize {
        self.size_hint().0
    }

    #[inline]
    fn last(mut self) -> Option<R> {
        self.next_back()
    }

    // TODO: This wants some tests
    #[inline]
    fn nth(&mut self, n: usize) -> Option<R> {
        let main_len = self.right - self.left;
        if main_len >= n {
            self.left += n;
            self.next()
        } else {
            self.left = self.right;
            self.partial.take_partial();
            None
        }
    }
}

impl<V, P, R> DoubleEndedIterator for VectorizedIter<V, P, R>
where
    V: Vectorizer<R>,
    P: Partial<R>,
{
    // TODO: Tests
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(partial) = self.partial.take_partial() {
            Some(partial)
        } else if self.left < self.right {
            self.right -= 1;
            Some(unsafe { self.vectorizer.get(self.right) })
        } else {
            None
        }
    }
}

impl<V, P, R> ExactSizeIterator for VectorizedIter<V, P, R>
where
    V: Vectorizer<R>,
    P: Partial<R>,
{
}

impl<V, P, R> FusedIterator for VectorizedIter<V, P, R>
where
    V: Vectorizer<R>,
    P: Partial<R>,
{
}

/// A trait describing things with direct support for splitting into vectors.
///
/// This supports vectorized iteration over shared and mutable slices as well as types composed of
/// them (tuples and short fixed-sized arrays).
///
/// Note that, unlike normal iterators, shared slices return owned values (vectors) and mutable
/// slices return [proxy objects][MutProxy] that allow writing the data back. It is not possible to
/// directly borrow from the slice because of alignment. The tuples and arrays return tuples and
/// arrays of the inner values.
///
/// Already pre-vectorized inputs are also supported (this is useful in combination with other not
/// vectorized inputs).
///
/// # Type hints
///
/// Oftentimes, the compiler can infer the type of the base type, but not the length of the vector.
/// It is therefore needed to provide a type hint.
///
/// Furthermore, for tuples and arrays, the inner type really needs to be the slice, not something
/// that can coerce into it (eg. vec or array).
///
/// Alternatively, you can use the free-standing functions [`vectorize`][crate::vectorize] and
/// [`vectorize_pad`][crate::vectorize_pad]. It allows using the turbofish to provide the hint.
///
/// # Examples
///
/// ```rust
/// # use slipstream::prelude::*;
/// let data = [1, 2, 3, 4];
/// let v = data.vectorize().collect::<Vec<u32x2>>();
/// assert_eq!(vec![u32x2::new([1, 2]), u32x2::new([3, 4])], v);
/// ```
///
/// ```rust
/// # use slipstream::prelude::*;
/// let data = [1, 2, 3, 4];
/// for v in data.vectorize() {
///     let v: u32x2 = v; // Type hint
///     println!("{:?}", v);
/// }
/// ```
///
/// ```rust
/// # use slipstream::prelude::*;
/// let input = [1, 2, 3, 4];
/// let mut output = [0; 4];
/// let mul = u32x2::splat(2);
/// // We have to force the coercion to slice by [..]
/// for (i, mut o) in (&input[..], &mut output[..]).vectorize() {
///     *o = mul * i;
/// }
/// assert_eq!(output, [2, 4, 6, 8]);
/// ```
///
/// ```rust
/// # use slipstream::prelude::*;
/// let vectorized = [u32x2::new([1, 2]), u32x2::new([3, 4])];
/// let not_vectorized = [1, 2, 3, 4];
/// for (v, n) in (&vectorized[..], &not_vectorized[..]).vectorize() {
///     assert_eq!(v, n);
/// }
/// ```
pub trait Vectorizable<V>: Sized {
    /// The input type provided by user to fill in the padding/uneven end.
    ///
    /// Note that this doesn't necessarily have to be the same type as the type returned by the
    /// resulting iterator. For example, in case of mutable slices, the input is the vector, while
    /// the output is [`MutProxy`].
    type Padding;

    /// An internal type managing the splitting into vectors.
    ///
    /// Not of direct interest of the users of this crate.
    type Vectorizer: Vectorizer<V>;

    /// Internal method to create the vectorizer and kick of the iteration.
    fn create(self, pad: Option<Self::Padding>) -> (Self::Vectorizer, usize, Option<V>);

    /// Vectorize a slice or composite of slices
    ///
    /// This variant assumes the input is divisible by the size of the vector. Prefer this if
    /// possible over [`vectorize_pad`][Vectorizable::vectorize_pad], as it is usually
    /// significantly faster.
    ///
    /// # Panics
    ///
    /// * If the slice length isn't divisible by the vector size.
    /// * If the parts of the composite produce different number of vectors. It is not mandated for
    ///   the slices to be of equal length, only to produce the same number of vectors.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use slipstream::prelude::*;
    /// let longer = [1, 2, 3, 4, 5, 6, 7, 8];
    /// let shorter = [1, 2, 3, 4];
    /// for i in (&shorter[..], &longer[..]).vectorize() {
    ///     let (s, l): (u32x2, u32x4) = i;
    ///     println!("s: {:?}, l: {:?})", s, l);
    /// }
    /// ```
    #[inline(always)]
    fn vectorize(self) -> VectorizedIter<Self::Vectorizer, (), V> {
        let (vectorizer, len, partial) = self.create(None);
        assert!(partial.is_none());
        VectorizedIter {
            partial: (),
            vectorizer,
            left: 0,
            right: len,
            _result: PhantomData,
        }
    }

    /// Vectorizes a slice or composite of slices, padding the odd end if needed.
    ///
    /// While the [`vectorize`][Vectorizable::vectorize] assumes the input can be split into
    /// vectors without leftover, this version deals with the uneven rest by producing a padding
    /// vector (if needed). The unused lanes are taken from the `pad` parameter. This is at the
    /// cost of some performance (TODO: figure out why it is so much slower).
    ///
    /// For mutable slices, padding is used as usual, but the added lanes are not stored anywhere.
    ///
    /// The padding is produced at the end.
    ///
    /// In case of composites, this still assumes they produce the same number of full vectors and
    /// that they all either do or don't need a padding.
    ///
    /// # Panics
    ///
    /// If the above assumption about number of vectors and same padding behaviour is violated.
    ///
    /// ```rust
    /// # use slipstream::prelude::*;
    /// let data = [1, 2, 3, 4, 5, 6];
    /// let v = data.vectorize_pad(i32x4::splat(-1)).collect::<Vec<_>>();
    /// assert_eq!(v, vec![i32x4::new([1, 2, 3, 4]), i32x4::new([5, 6, -1, -1])]);
    /// ```
    #[inline(always)]
    fn vectorize_pad(self, pad: Self::Padding) -> VectorizedIter<Self::Vectorizer, Option<V>, V> {
        let (vectorizer, len, partial) = self.create(Some(pad));
        VectorizedIter {
            partial,
            vectorizer,
            left: 0,
            right: len,
            _result: PhantomData,
        }
    }
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct ReadVectorizer<'a, A: Align, B: Repr, const S: usize> {
    start: *const B,
    _vector: PhantomData<Vector<A, B, S>>,
    _slice: PhantomData<&'a [B]>, // To hold the lifetime
}

// Note: The impls here assume V, B, P are Sync and Send, which they are. Nobody is able to create
// this directly and we do have the limits on Vector, the allowed implementations, etc.
unsafe impl<A: Align, B: Repr, const S: usize> Send for ReadVectorizer<'_, A, B, S> {}
unsafe impl<A: Align, B: Repr, const S: usize> Sync for ReadVectorizer<'_, A, B, S> {}

impl<A: Align, B: Repr, const S: usize> Vectorizer<Vector<A, B, S>>
    for ReadVectorizer<'_, A, B, S>
{
    #[inline(always)]
    unsafe fn get(&mut self, idx: usize) -> Vector<A, B, S> {
        Vector::new_unchecked(self.start.add(S * idx))
    }
}

impl<'a, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>> for &'a [B] {
    type Vectorizer = ReadVectorizer<'a, A, B, S>;
    type Padding = Vector<A, B, S>;
    #[inline]
    fn create(
        self,
        pad: Option<Vector<A, B, S>>,
    ) -> (Self::Vectorizer, usize, Option<Vector<A, B, S>>) {
        let len = self.len();
        assert!(
            len * mem::size_of::<B>() <= isize::MAX as usize,
            "Slice too huge"
        );
        let rest = len % S;
        let main = len - rest;
        let start = self.as_ptr();
        let partial = match (rest, pad) {
            (0, _) => None,
            (_, Some(mut pad)) => {
                pad[..rest].copy_from_slice(&self[main..]);
                Some(pad)
            }
            _ => panic!(
                "Data to vectorize not divisible by lanes ({} vs {})",
                S, len,
            ),
        };
        let me = ReadVectorizer {
            start,
            _vector: PhantomData,
            _slice: PhantomData,
        };
        (me, main / S, partial)
    }
}

#[doc(hidden)]
#[derive(Copy, Clone, Debug)]
pub struct WriteVectorizer<'a, A: Align, B: Repr, const S: usize> {
    start: *mut B,
    _vector: PhantomData<Vector<A, B, S>>,
    _slice: PhantomData<&'a mut [B]>, // To hold the lifetime
}

// Note: The impls here assume V, B, P are Sync and Send, which they are. Nobody is able to create
// this directly and we do have the limits on Vector, the allowed implementations, etc.
unsafe impl<A: Align, B: Repr, const S: usize> Send for WriteVectorizer<'_, A, B, S> {}
unsafe impl<A: Align, B: Repr, const S: usize> Sync for WriteVectorizer<'_, A, B, S> {}

impl<'a, A: Align, B: Repr, const S: usize> Vectorizer<MutProxy<'a, B, Vector<A, B, S>>>
    for WriteVectorizer<'a, A, B, S>
{
    #[inline(always)]
    unsafe fn get(&mut self, idx: usize) -> MutProxy<'a, B, Vector<A, B, S>> {
        // FIXME: Technically, we extend the lifetime in the from_raw_parts_mut beyond what rust
        // would allow us to normally do. But is this OK? As we are guaranteed never to give any
        // chunk twice, this should act similar to IterMut from slice or similar.
        let ptr = self.start.add(S * idx);
        MutProxy {
            data: Vector::new_unchecked(ptr),
            restore: slice::from_raw_parts_mut(ptr, S),
        }
    }
}

impl<'a, A: Align, B: Repr, const S: usize> Vectorizable<MutProxy<'a, B, Vector<A, B, S>>>
    for &'a mut [B]
{
    type Vectorizer = WriteVectorizer<'a, A, B, S>;
    type Padding = Vector<A, B, S>;
    #[inline]
    #[allow(clippy::type_complexity)]
    fn create(
        self,
        pad: Option<Vector<A, B, S>>,
    ) -> (
        Self::Vectorizer,
        usize,
        Option<MutProxy<'a, B, Vector<A, B, S>>>,
    ) {
        let len = self.len();
        assert!(
            len * mem::size_of::<B>() <= isize::MAX as usize,
            "Slice too huge"
        );
        let rest = len % S;
        let main = len - rest;
        let start = self.as_mut_ptr();
        let partial = match (rest, pad) {
            (0, _) => None,
            (_, Some(mut pad)) => {
                let restore = &mut self[main..];
                pad[..rest].copy_from_slice(restore);
                Some(MutProxy { data: pad, restore })
            }
            _ => panic!(
                "Data to vectorize not divisible by lanes ({} vs {})",
                S, len,
            ),
        };
        let me = WriteVectorizer {
            start,
            _vector: PhantomData,
            _slice: PhantomData,
        };
        (me, main / S, partial)
    }
}

macro_rules! vectorizable_tuple {
    ($(($X: ident, $XR: ident, $X0: tt)),*) => {
        impl<$($X, $XR),*> Vectorizer<($($XR),*)> for ($($X),*)
        where
            $($X: Vectorizer<$XR>,)*
        {
            #[inline(always)]
            unsafe fn get(&mut self, idx: usize) -> ($($XR),*) {
                ($(self.$X0.get(idx)),*)
            }
        }

        impl<$($X, $XR),*> Vectorizable<($($XR),*)> for ($($X),*)
        where
            $($X: Vectorizable<$XR>,)*
        {
            type Vectorizer = ($($X::Vectorizer),*);
            type Padding = ($($X::Padding),*);
            #[inline]
            #[allow(clippy::eq_op)]
            fn create(self, pad: Option<Self::Padding>)
                -> (Self::Vectorizer, usize, Option<($($XR),*)>)
            {
                let pad = match pad {
                    Some(pad) => ($(Some(pad.$X0)),*),
                    None => Default::default(), // Bunch of Nones in a tuple.. (None, None, None)...
                };
                let created = ($(self.$X0.create(pad.$X0)),*);
                $(
                    // TODO: We may want to support this in the padded mode eventually by
                    // creating more paddings
                    assert_eq!(
                        (created.0).1,
                        created.$X0.1,
                        "Vectorizing data of different lengths"
                    );
                    // TODO: We could also handle this in the padded mode by doing empty pads
                    assert_eq!(
                        (created.0).2.is_some(),
                        created.$X0.2.is_some(),
                        "Paddings are not the same for all vectorized data",
                    );
                )*
                let vectorizer = ($(created.$X0.0),*);
                let pad = if (created.0).2.is_some() {
                    Some(($(created.$X0.2.unwrap()),*))
                } else {
                    None
                };
                (vectorizer, (created.0).1, pad)
            }
        }
    }
}

vectorizable_tuple!((A, AR, 0), (B, BR, 1));
vectorizable_tuple!((A, AR, 0), (B, BR, 1), (C, CR, 2));
vectorizable_tuple!((A, AR, 0), (B, BR, 1), (C, CR, 2), (D, DR, 3));
vectorizable_tuple!((A, AR, 0), (B, BR, 1), (C, CR, 2), (D, DR, 3), (E, ER, 4));
vectorizable_tuple!(
    (A, AR, 0),
    (B, BR, 1),
    (C, CR, 2),
    (D, DR, 3),
    (E, ER, 4),
    (F, FR, 5)
);
vectorizable_tuple!(
    (A, AR, 0),
    (B, BR, 1),
    (C, CR, 2),
    (D, DR, 3),
    (E, ER, 4),
    (F, FR, 5),
    (G, GR, 6)
);
vectorizable_tuple!(
    (A, AR, 0),
    (B, BR, 1),
    (C, CR, 2),
    (D, DR, 3),
    (E, ER, 4),
    (F, FR, 5),
    (G, GR, 6),
    (H, HR, 7)
);

impl<T, TR, const S: usize> Vectorizer<[TR; S]> for [T; S]
where
    T: Vectorizer<TR>,
{
    #[inline(always)]
    unsafe fn get(&mut self, idx: usize) -> [TR; S] {
        let mut res = MaybeUninit::<[TR; S]>::uninit();
        for (i, v) in self.iter_mut().enumerate() {
            ptr::write(res.as_mut_ptr().cast::<TR>().add(i), v.get(idx));
        }
        res.assume_init()
    }
}

impl<T, TR, const S: usize> Vectorizable<[TR; S]> for [T; S]
where
    T: Vectorizable<TR> + Copy,
    T::Padding: Copy,
{
    type Vectorizer = [T::Vectorizer; S];
    type Padding = [T::Padding; S];
    #[inline]
    fn create(self, pad: Option<Self::Padding>) -> (Self::Vectorizer, usize, Option<[TR; S]>) {
        let mut vectorizer = MaybeUninit::<Self::Vectorizer>::uninit();
        let mut size = 0;
        let mut padding = MaybeUninit::<[TR; S]>::uninit();
        let mut seen_some_pad = false;
        let mut seen_none_pad = false;
        unsafe {
            for i in 0..S {
                let (v, s, p) = self[i].create(pad.map(|p| p[i]));
                ptr::write(vectorizer.as_mut_ptr().cast::<T::Vectorizer>().add(i), v);
                if i == 0 {
                    size = s;
                } else {
                    assert_eq!(size, s, "Vectorized lengths inconsistent across the array",);
                }
                match p {
                    Some(p) => {
                        seen_some_pad = true;
                        ptr::write(padding.as_mut_ptr().cast::<TR>().add(i), p);
                    }
                    None => seen_none_pad = true,
                }
            }
            assert!(
                !seen_some_pad || !seen_none_pad,
                "Paddings inconsistent across the array",
            );
            let padding = if seen_some_pad {
                Some(padding.assume_init())
            } else {
                None
            };
            (vectorizer.assume_init(), size, padding)
        }
    }
}

impl<'a, T> Vectorizer<T> for &'a [T]
where
    T: Copy,
{
    unsafe fn get(&mut self, idx: usize) -> T {
        *self.get_unchecked(idx)
    }
}

impl<'a, T> Vectorizer<&'a mut T> for &'a mut [T] {
    unsafe fn get(&mut self, idx: usize) -> &'a mut T {
        // FIXME: Why do we have to extend the lifetime here? Is it safe? Intuitively, it should,
        // because we hand out each chunk only once and this is what IterMut does too.
        let ptr = self.get_unchecked_mut(idx) as *mut T;
        &mut *ptr
    }
}

impl<'a, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'a [Vector<A, B, S>]
{
    type Padding = ();
    type Vectorizer = &'a [Vector<A, B, S>];
    fn create(self, _pad: Option<()>) -> (Self::Vectorizer, usize, Option<Vector<A, B, S>>) {
        (self, self.len(), None)
    }
}

impl<'a, A: Align, B: Repr, const S: usize> Vectorizable<&'a mut Vector<A, B, S>>
    for &'a mut [Vector<A, B, S>]
{
    type Padding = ();
    type Vectorizer = &'a mut [Vector<A, B, S>];
    fn create(
        self,
        _pad: Option<()>,
    ) -> (Self::Vectorizer, usize, Option<&'a mut Vector<A, B, S>>) {
        let len = self.len();
        (self, len, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn iter() {
        let data = (0..=10u16).collect::<Vec<_>>();
        let vtotal: u16x8 = data.vectorize_pad(u16x8::default()).sum();
        let total: u16 = vtotal.horizontal_sum();
        assert_eq!(total, 55);
    }

    #[test]
    fn iter_mut() {
        let data = (0..33u32).collect::<Vec<_>>();
        let mut dst = [0u32; 33];
        let ones = u32x4::splat(1);
        for (mut d, s) in
            (&mut dst[..], &data[..]).vectorize_pad((u32x4::default(), u32x4::default()))
        {
            *d = ones + s;
        }

        for (l, r) in data.iter().zip(dst.iter()) {
            assert_eq!(*l + 1, *r);
        }
    }

    // Here, one of the inputs is already vectorized
    #[test]
    fn iter_prevec() {
        let src = [0, 1, 2, 3, 4, 5, 6, 7];
        let mut dst = [u16x4::default(); 2];

        for (dst, src) in (&mut dst[..], &src[..]).vectorize() {
            *dst = src;
        }

        assert_eq!(dst, [u16x4::new([0, 1, 2, 3]), u16x4::new([4, 5, 6, 7])]);
    }
}
