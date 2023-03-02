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
pub mod experimental {
    use crate::{inner::Repr, vector::align::Align, Vector};
    use core::{
        iter::FusedIterator,
        marker::PhantomData,
        mem::MaybeUninit,
        ops::{Deref, DerefMut},
    };

    // === Step 1: Abstraction over SIMD data access ===

    /// Query the configuration of a Vector type
    ///
    /// # Safety
    ///
    /// Users of this trait may rely on the provided information to be correct
    /// for safety.
    //
    // TODO: Should probably be elsewhere in the crate
    pub unsafe trait VectorInfo: AsRef<[Self::Scalar]> + Copy + Sized + 'static {
        /// Inner scalar type (commonly called B in generics)
        type Scalar: Copy + Sized + 'static;

        /// Number of vector lanes (commonly called S in generics)
        const LANES: usize;

        /// Equivalent array type (will always be [Self::Scalar; Self::LANES],
        /// but Rust does not support asserting this at the moment)
        type Array: Copy + Sized + 'static;

        /// Build from an index -> element mapping
        fn from_fn(mapping: impl FnMut(usize) -> Self::Scalar) -> Self;
    }
    //
    unsafe impl<A: Align, B: Repr, const S: usize> VectorInfo for Vector<A, B, S> {
        type Scalar = B;
        const LANES: usize = S;
        type Array = [B; S];

        // FIXME: Can't use array::from_fn in perf-sensitive code as it is
        //        not marked inline and may actually not be inlined... But this
        //        will go away during padding reform anyhow
        fn from_fn(mapping: impl FnMut(usize) -> Self::Scalar) -> Self {
            core::array::from_fn(mapping).into()
        }
    }

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
        /// Borrowed element of the output `Vectors` collection
        type ElementRef<'result>: Sized
        where
            Self: 'result;

        /// Owned element of the output `Vectors` collection
        type Element: Sized;

        // TODO: Add SliceIndex-like GAT infrastructure

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
    // FIXME: Remove this paragraph after padding cleanup
    /// Unlike the normal `[T]::get_unchecked` operation, however, the
    /// `get_unchecked` operation of this type accounts for the possibility that
    /// some slice values may not be present in the underlying storage, and may
    /// need to be replaced by a placeholder scalar padding value, typically set
    /// to the neutral element of the computation being performed. This is
    /// needed when wrapping slices of scalars, whose length may not be a
    /// multiple of the SIMD vector width.
    ///
    /// # Safety
    ///
    /// Unsafe code may rely on the correctness of implementations of this trait
    /// and the higher-level `Vectorized` trait as part of their safety proofs.
    ///
    /// The safety preconditions on `Vectorized` are that `OwnedElement` should
    /// not outlive `Self`, and that it should be safe to transmute `ElementRef`
    /// to `Element` in scenarios where either `Element` is `Copy` or the
    /// transmute is abstracted in such a way that the user cannot abuse it to
    /// get two copies of the same element.
    ///
    /// Furthermore, a `Vectorized` impl is only allowed to implement `Copy` if
    /// the underlying element type is `Copy`.
    #[doc(hidden)]
    pub unsafe trait VectorizedImpl<V: VectorInfo>: Vectorized<V> + Sized {
        /// Truth that this data may safely reinterpreted as a slice or
        /// collection of `Vector` using `as_aligned_unchecked()`
        fn is_aligned(&self) -> bool;

        /// Unsafely cast this data to the equivalent slice or collection of Vector.
        unsafe fn as_aligned_unchecked(self) -> Self::Aligned;

        /// Attempt to cast this to the equivalent slice or collection of Vector,
        /// return None if it is not safe to do so.
        #[inline(always)]
        fn as_aligned(self) -> Option<Self::Aligned> {
            self.is_aligned()
                .then(|| unsafe { self.as_aligned_unchecked() })
        }

        // FIXME: Clean up during padding cleanup
        /// Access the underlying slice at vector index `idx` without bounds
        /// checking, adding scalar padding values if data is missing at the
        /// end of the target vector.
        ///
        /// # Safety
        ///
        /// - Index `idx` must be in within the bounds of the underlying slice.
        /// - If padding is needed, then `padding` must contain a valid scalar.
        unsafe fn get_unchecked(
            &mut self,
            idx: usize,
            padding: MaybeUninit<V::Scalar>,
        ) -> Self::ElementRef<'_>;
    }

    /// Read access to aligned SIMD data
    //
    // --- Internal docs start here ---
    //
    // Base pointer of an `&[Vector]` slice, tagged with lifetime information
    #[derive(Copy, Clone)]
    pub struct AlignedVectors<'target, T>(*const T, PhantomData<&'target [T]>);
    //
    impl<T> AlignedVectors<'_, T> {
        /// Tag a base pointer with lifetime information
        ///
        /// # Safety
        ///
        /// The pointer's target must be valid for this lifetime
        #[inline(always)]
        unsafe fn new(inner: *const T) -> Self {
            Self(inner, PhantomData)
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for AlignedVectors<'target, V> {
        type Aligned = Self;
        type ElementRef<'result> = V where Self: 'result;
        type Element = V;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for AlignedVectors<'target, V> {
        #[inline(always)]
        fn is_aligned(&self) -> bool {
            true
        }

        #[inline(always)]
        unsafe fn as_aligned_unchecked(self) -> Self {
            self
        }

        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _padding: MaybeUninit<V::Scalar>) -> V {
            unsafe { *self.0.add(idx) }
        }
    }

    // Owned arrays of Vector must be stored as-is in the Vectors collection,
    // but otherwise behave like &[Vector]
    unsafe impl<V: VectorInfo, const SIZE: usize> Vectorized<V> for [V; SIZE] {
        type Aligned = Self;
        type ElementRef<'result> = V;
        type Element = V;
    }
    //
    unsafe impl<V: VectorInfo, const SIZE: usize> VectorizedImpl<V> for [V; SIZE] {
        #[inline(always)]
        fn is_aligned(&self) -> bool {
            true
        }

        #[inline(always)]
        unsafe fn as_aligned_unchecked(self) -> Self {
            self
        }

        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _padding: MaybeUninit<V::Scalar>) -> V {
            unsafe { *<[V]>::get_unchecked(&self[..], idx) }
        }
    }

    /// Write access to aligned SIMD data
    //
    // --- Internal docs start here ---
    //
    // Base pointer of an `&mut [Vector]` slice, tagged with lifetime information
    pub struct AlignedVectorsMut<'target, T>(*mut T, PhantomData<&'target mut [T]>);
    //
    impl<T> AlignedVectorsMut<'_, T> {
        /// Tag a base pointer with lifetime information
        ///
        /// # Safety
        ///
        /// The pointer's target must be valid for this lifetime
        #[inline(always)]
        unsafe fn new(inner: *mut T) -> Self {
            Self(inner, PhantomData)
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for AlignedVectorsMut<'target, V> {
        type Aligned = Self;
        type ElementRef<'result> = &'result mut V where Self: 'result;
        type Element = &'target mut V;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for AlignedVectorsMut<'target, V> {
        #[inline(always)]
        fn is_aligned(&self) -> bool {
            true
        }

        #[inline(always)]
        unsafe fn as_aligned_unchecked(self) -> Self {
            self
        }

        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _padding: MaybeUninit<V::Scalar>) -> &mut V {
            unsafe { &mut *self.0.add(idx) }
        }
    }

    /// Read access to padded scalar data
    //
    // --- Internal docs start here ---
    //
    // Start and end pointers of an `&[Scalar]` slice, tagged with lifetime
    // information (FIXME: Adjust this during padding reform).
    #[derive(Copy, Clone)]
    pub struct PaddedScalars<'target, V: VectorInfo> {
        start: *const V::Scalar,
        end: *const V::Scalar,
        _vector: PhantomData<&'target [V]>,
    }
    //
    impl<V: VectorInfo> PaddedScalars<'_, V> {
        /// Build from scalar slice raw parts
        ///
        /// # Safety
        ///
        /// - `data` must be valid for the lifetime of this tagged pointer.
        /// - `len` must not go past the end of `data`'s allocation.
        #[inline(always)]
        unsafe fn new(data: *const V::Scalar, len: usize) -> Self {
            Self {
                start: data,
                end: unsafe { data.add(len) },
                _vector: PhantomData,
            }
        }

        /// Implementation of get_unchecked that gives back the base pointer
        #[inline(always)]
        unsafe fn get_unchecked_impl(
            &mut self,
            idx: usize,
            padding: MaybeUninit<V::Scalar>,
        ) -> (V, *const V::Scalar) {
            let base_ptr = self.start.add(idx * V::LANES);
            let value = V::from_fn(|offset| {
                let scalar_ptr = base_ptr.wrapping_add(offset);
                if scalar_ptr < self.end {
                    unsafe { *scalar_ptr }
                } else {
                    unsafe { padding.assume_init() }
                }
            });
            (value, base_ptr)
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for PaddedScalars<'target, V> {
        type Aligned = AlignedVectors<'target, V>;
        type ElementRef<'result> = V where Self: 'result;
        type Element = V;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for PaddedScalars<'target, V> {
        #[inline(always)]
        fn is_aligned(&self) -> bool {
            let is_aligned = |ptr: *const V::Scalar| ptr as usize % core::mem::align_of::<V>() == 0;
            // TODO: Move this to as_unaligned later on
            assert_eq!(
                core::mem::size_of::<V>(),
                core::mem::size_of::<V::Array>(),
                "Should always be true and will be elided at compile time, but \
                not actually guaranteed for non-repr(transparent) types"
            );
            is_aligned(self.start) && is_aligned(self.end)
        }

        #[inline(always)]
        unsafe fn as_aligned_unchecked(self) -> Self::Aligned {
            unsafe { AlignedVectors::new(self.start.cast()) }
        }

        #[inline(always)]
        unsafe fn get_unchecked(
            &mut self,
            idx: usize,
            padding: MaybeUninit<V::Scalar>,
        ) -> Self::ElementRef<'_> {
            self.get_unchecked_impl(idx, padding).0
        }
    }

    // NOTE: Can't implement support for [Scalar; SIZE] yet due to const
    //       generics limitations around AsVectorsResult (AsVectors should be
    //       [Vector; { SIZE / Vector::LANES }], but array lengths derived from
    //       generic parameters are not allowed yet.

    /// Write access to padded scalar data
    //
    // --- Internal docs start here ---
    //
    // Start and end pointers of an `&mut [Scalar]` slice, tagged with lifetime
    // information (FIXME: Adjust this during padding reform).
    pub struct PaddedScalarsMut<'target, V: VectorInfo> {
        inner: PaddedScalars<'target, V>,
        _vector: PhantomData<&'target mut [V]>,
    }
    //
    impl<V: VectorInfo> PaddedScalarsMut<'_, V> {
        /// Build from scalar slice raw parts
        ///
        /// # Safety
        ///
        /// - `data` must be valid for the lifetime of this tagged pointer.
        /// - `data.add(len)` must not go past the end of `data`'s allocation.
        #[inline(always)]
        unsafe fn new(data: *mut V::Scalar, len: usize) -> Self {
            Self {
                inner: PaddedScalars::new(data, len),
                _vector: PhantomData,
            }
        }
    }
    //
    unsafe impl<'target, V: VectorInfo> Vectorized<V> for PaddedScalarsMut<'target, V> {
        type Aligned = AlignedVectorsMut<'target, V>;
        type ElementRef<'result> = PaddedVectorMut<'result, V> where Self: 'result;
        type Element = PaddedVectorMut<'target, V>;
    }
    //
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for PaddedScalarsMut<'target, V> {
        #[inline(always)]
        fn is_aligned(&self) -> bool {
            self.inner.is_aligned()
        }

        #[inline(always)]
        unsafe fn as_aligned_unchecked(self) -> Self::Aligned {
            unsafe { AlignedVectorsMut::new(self.inner.start.cast_mut().cast()) }
        }

        #[inline(always)]
        unsafe fn get_unchecked(
            &mut self,
            idx: usize,
            padding: MaybeUninit<V::Scalar>,
        ) -> Self::ElementRef<'_> {
            let (vector, base_ptr) = self.inner.get_unchecked_impl(idx, padding);
            let base_ptr = base_ptr.cast_mut();
            PaddedVectorMut {
                vector,
                target: core::slice::from_raw_parts_mut(base_ptr, unsafe {
                    self.inner.end.offset_from(base_ptr)
                } as usize),
            }
        }
    }

    /// Vector mutation proxy for padded scalar slices
    ///
    /// For mutation from &mut [Scalar], we can't provide an &mut Vector as it
    /// could be misaligned and out of bounds. So we provide a proxy object
    /// that acts as closely to &mut Vector as possible.
    pub struct PaddedVectorMut<'target, V: VectorInfo> {
        vector: V,
        target: &'target mut [V::Scalar],
    }
    //
    impl<V: VectorInfo> Deref for PaddedVectorMut<'_, V> {
        type Target = V;

        fn deref(&self) -> &V {
            &self.vector
        }
    }
    //
    impl<V: VectorInfo> DerefMut for PaddedVectorMut<'_, V> {
        fn deref_mut(&mut self) -> &mut V {
            &mut self.vector
        }
    }
    //
    impl<V: VectorInfo> Drop for PaddedVectorMut<'_, V> {
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
                type Aligned = ($($t::Aligned,)*);
                type ElementRef<'result> = ($($t::ElementRef<'result>,)*) where Self: 'result;
                type Element = ($($t::Element,)*);
            }

            #[allow(non_snake_case)]
            unsafe impl<
                'target,
                V: VectorInfo
                $(, $t: VectorizedImpl<V> + 'target)*
            > VectorizedImpl<V> for ($($t,)*) {
                #[inline(always)]
                fn is_aligned(&self) -> bool {
                    let ($($t,)*) = self;
                    $(
                        if !$t.is_aligned() {
                            return false;
                        }
                    )*
                    true
                }

                #[inline(always)]
                unsafe fn as_aligned_unchecked(self) -> Self::Aligned {
                    let ($($t,)*) = self;
                    unsafe { ($($t.as_aligned_unchecked(),)*) }
                }

                #[inline(always)]
                unsafe fn get_unchecked(
                    &mut self,
                    idx: usize,
                    padding: MaybeUninit<V::Scalar>
                ) -> Self::ElementRef<'_> {
                    let ($($t,)*) = self;
                    unsafe { ($($t.get_unchecked(idx, padding),)*)  }
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
    pub struct Vectors<V: VectorInfo, Data: Vectorized<V> + VectorizedImpl<V>> {
        data: Data,
        len: usize,
        last: usize,
        padding: MaybeUninit<V::Scalar>,
    }
    //
    impl<V: VectorInfo, Data: VectorizedImpl<V>> Vectors<V, Data> {
        /// Create a SIMD data container
        ///
        /// # Safety
        ///
        /// - It must be safe to dereference `data` for any index in 0..len
        /// - If any of the inner pointers require padding, then padding must
        ///   be initialized to a valid padding value.
        #[inline(always)]
        unsafe fn new(data: Data, len: usize, padding: MaybeUninit<V::Scalar>) -> Self {
            Self {
                data,
                len,
                last: len - 1,
                padding,
            }
        }

        /// Returns the number of elements in the container
        #[inline(always)]
        pub const fn len(&self) -> usize {
            self.len
        }

        /// Returns `true` if there are no elements in the container
        #[inline(always)]
        pub const fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Returns the first element, or None if the container is empty
        #[inline(always)]
        pub fn first(&mut self) -> Option<Data::ElementRef<'_>> {
            self.get(0)
        }

        // TODO: split_(first|last)

        /// Returns the last element, or None if the container is empty
        #[inline(always)]
        pub fn last(&mut self) -> Option<Data::ElementRef<'_>> {
            self.get(self.last)
        }

        /// Returns the N-th element of the container
        // TODO: Generalize to subslices, but without using SliceIndex since
        //       that's not yet in stable Rust.
        #[inline(always)]
        pub fn get(&mut self, idx: usize) -> Option<Data::ElementRef<'_>> {
            if (0..self.len).contains(&idx) {
                Some(unsafe { self.get_unchecked(idx) })
            } else {
                None
            }
        }

        /// Returns the N-th element of the container without bounds checking
        ///
        /// # Safety
        ///
        /// `idx` must be in range `0..self.len()`
        //
        // TODO: Generalize to subslices, but without using SliceIndex since
        //       that's not yet in stable Rust.
        #[inline(always)]
        pub unsafe fn get_unchecked(&mut self, idx: usize) -> Data::ElementRef<'_> {
            unsafe { self.data.get_unchecked(idx, self.padding) }
        }

        /// Returns an iterator over contained elements
        #[inline(always)]
        pub fn iter(&mut self) -> VectorsIter<V, Data> {
            <&mut Self>::into_iter(self)
        }

        // TODO: chunks(_exact)?,

        /// Returns an iterator over N elements at a time, starting at the
        /// beginning of the container
        // TODO: Make a dedicated Iterator so I can implement DoubleEnded + ExactSize + Fused
        //       and add a remainder
        #[inline(always)]
        pub fn array_chunks<const N: usize>(
            &mut self,
        ) -> impl Iterator<Item = [Data::ElementRef<'_>; N]> {
            let mut iter = self.iter();
            core::iter::from_fn(move || {
                if iter.len() >= N {
                    // FIXME: Can't use array::from_fn in perf-sensitive code as
                    //        it may not be inlined...
                    Some(core::array::from_fn(|_| iter.next().unwrap()))
                } else {
                    None
                }
            })
        }

        // TODO: rchunks(_exact)?, split_at, r?split_array, split(_inclusive)?,
        //       rsplit(_inclusive)?, r?splitn
        // TODO: Index by anything that get accepts
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

                #[inline(always)]
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

                #[inline(always)]
                fn size_hint(&self) -> (usize, Option<usize>) {
                    (self.len(), Some(self.len()))
                }

                #[inline(always)]
                fn count(self) -> usize {
                    self.len()
                }

                #[inline(always)]
                fn last(mut self) -> Option<Self::Item> {
                    self.next_back()
                }

                #[inline(always)]
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

                #[inline(always)]
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
                #[inline(always)]
                fn len(&self) -> usize {
                    self.end - self.start
                }
            }
            //
            impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> FusedIterator
                for $name<$($lifetime,)? V, Data>
            {
            }
        }
    }
    impl_iterator!(
        /// Borrowing iterator over Vectors' elements
        (VectorsIter, Data::ElementRef<'vectors>)
    );
    impl_iterator!(
        /// Owned iterator over Vectors' elements
        (VectorsIntoIter, Data::Element)
    );

    // === Step 3: Translate from scalar slices and containers to vector slices ===

    /// Trait for data that can be processed using SIMD
    ///
    /// Implemented for slices and containers of vectors and scalars,
    /// as well as for tuples of these entities.
    ///
    /// Provides you with ways to create the `Vectors` collection, which
    /// behaves conceptually like an array of `Vector` or tuples thereof, with
    /// iteration and indexing operations yielding the following types:
    ///
    /// - If built out of a read-only slice or owned container of vectors or
    ///   scalars, yield owned `Vector`s of data.
    /// - If built out of `&mut [Vector]`, or `&mut [Scalar]` that is assumed
    ///   to be SIMD-aligned (see below), it yields `&mut Vector` references.
    /// - If built out of `&mut [Scalar]` that is not SIMD-aligned, it yields
    ///   a proxy type which can be used like an `&mut Vector` (but cannot
    ///   literally be `&mut Vector`)
    /// - If built out of a tuple of the above entities, it yields tuples of the
    ///   aforementioned elements.
    ///
    /// There are three ways to create `Vectors` using this trait depending on
    /// what kind of data you're starting from:
    ///
    /// - If starting out of arbitrary data, you can use the [`vectorize_pad()`]
    ///   method to get a SIMD view that does not make any assumption, but
    ///   tends to exhibit poor performance on scalar data as a result.
    /// - If you know that every scalar array in your dataset has a number of
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

        /// Internal method preparing for the `vectorize()` methods
        //
        // --- Internal docs starts here ---
        //
        // The returned building blocks are...
        //
        // - A pointer-like entity for treating the data as a slice of Vector
        //   (see VectorizedImpl for more information)
        // - The number of Vector elements that the emulated slice contains
        // - The truth that this data needs padding
        //
        // This panics if called on a tuple of slices/containers who would not
        // produce the same amount of Vector elements.
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool);

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
        #[inline(always)]
        fn vectorize(self) -> Vectors<V, Self::Vectorized> {
            let (base, len, needs_padding) = self.into_vectorized_parts();
            assert!(
                !needs_padding,
                "Scalar data requires padding, but padding was not provided"
            );
            unsafe { Vectors::new(base, len, MaybeUninit::uninit()) }
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
        #[inline(always)]
        fn vectorize_pad(self, padding: V::Scalar) -> Vectors<V, Self::Vectorized> {
            let (base, len, _needs_padding) = self.into_vectorized_parts();
            unsafe { Vectors::new(base, len, MaybeUninit::new(padding)) }
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
        #[inline(always)]
        fn vectorized_aligned(self) -> Vectors<V, <Self::Vectorized as Vectorized<V>>::Aligned> {
            let (base, len, needs_padding) = self.into_vectorized_parts();
            let base = base
                .as_aligned()
                .expect("Data is not in a SIMD-friendly layout");
            debug_assert!(!needs_padding, "SIMD-friendly data should not need padding");
            unsafe { Vectors::new(base, len, MaybeUninit::uninit()) }
        }
    }

    // === Vectorize implementation is trivial for slices of vector data ===

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target [Vector<A, B, S>]
    {
        type Vectorized = AlignedVectors<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            (
                unsafe { AlignedVectors::new(self.as_ptr()) },
                self.len(),
                false,
            )
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target mut [Vector<A, B, S>]
    {
        type Vectorized = AlignedVectorsMut<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            (
                unsafe { AlignedVectorsMut::new(self.as_mut_ptr()) },
                self.len(),
                false,
            )
        }
    }

    unsafe impl<A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for [Vector<A, B, S>; ARRAY_SIZE]
    {
        type Vectorized = [Vector<A, B, S>; ARRAY_SIZE];

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            (self, ARRAY_SIZE, false)
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for &'target [Vector<A, B, S>; ARRAY_SIZE]
    {
        type Vectorized = AlignedVectors<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            self.as_slice().into_vectorized_parts()
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for &'target mut [Vector<A, B, S>; ARRAY_SIZE]
    {
        type Vectorized = AlignedVectorsMut<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            self.as_mut_slice().into_vectorized_parts()
        }
    }

    // === For scalar data, must cautiously handle padding and alignment ===

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target [B]
    {
        type Vectorized = PaddedScalars<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            let needs_padding = self.len() % S != 0;
            (
                unsafe { PaddedScalars::new(self.as_ptr(), self.len()) },
                self.len() / S + needs_padding as usize,
                needs_padding,
            )
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target mut [B]
    {
        type Vectorized = PaddedScalarsMut<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            let needs_padding = self.len() % S != 0;
            (
                unsafe { PaddedScalarsMut::new(self.as_mut_ptr(), self.len()) },
                self.len() / S + needs_padding as usize,
                needs_padding,
            )
        }
    }

    // NOTE: Cannot be implemented for owned scalar arrays yet due to const
    //       generics limitations around vectorized_aligned().

    unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for &'target [B; ARRAY_SIZE]
    {
        type Vectorized = PaddedScalars<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            self.as_slice().into_vectorized_parts()
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
        Vectorizable<Vector<A, B, S>> for &'target mut [B; ARRAY_SIZE]
    {
        type Vectorized = PaddedScalarsMut<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn into_vectorized_parts(self) -> (Self::Vectorized, usize, bool) {
            self.as_mut_slice().into_vectorized_parts()
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

                #[inline(always)]
                fn into_vectorized_parts(
                    self,
                ) -> (Self::Vectorized, usize, bool) {
                    // Pattern-match the tuple to variables named after inner types
                    let ($($t,)*) = self;

                    // Reinterpret tuple fields as SIMD vectors
                    let ($($t,)*) = ($($t.into_vectorized_parts(),)*);

                    // Analyze tuple field lengths and need for padding
                    let mut len = None;
                    let mut needs_padding = false;
                    $(
                        let (_, t_len, t_needs_padding) = $t;

                        // All tuple fields should have the same SIMD length
                        #[allow(unused_assignments)]
                        if let Some(len) = len {
                            assert_eq!(
                                t_len, len,
                                "Tuple elements do not produce the same amount of SIMD vectors"
                            );
                        } else {
                            len = Some(t_len);
                        }

                        // If at least one field needs padding, the tuple does
                        if t_needs_padding {
                            needs_padding = true;
                        }
                    )*

                    // All good, return Vectors building blocks
                    (
                        ($($t.0,)*),
                        len.expect("This should not be implemented for zero-sized tuples"),
                        needs_padding
                    )
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
