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
        marker::PhantomData,
        mem::MaybeUninit,
        ops::{Deref, DerefMut},
    };

    // === Step 1: Abstraction over SIMD data access ===

    /// Query the configuration of a Vector type
    pub trait VectorInfo: AsRef<[Self::Scalar]> + Copy + Sized + 'static {
        /// Inner scalar type (commonly called B in generics)
        type Scalar: Copy + Sized + 'static;

        /// Number of vector lanes (commonly called S in generics)
        const LANES: usize;
    }
    //
    impl<A: Align, B: Repr, const S: usize> VectorInfo for Vector<A, B, S> {
        type Scalar = B;
        const LANES: usize = S;
    }

    /// Entity that can be treated as the base pointer of an &[Vector] or
    /// &mut [Vector] slice
    ///
    /// Implementors of this trait operate in the context of an underlying real
    /// or simulated slice of SIMD vectors, or of a tuple of several slices of
    /// equal length that is made to behave like a slice of tuples.
    ///
    /// The length of the underlying slice is not known by this type, it is
    /// stored as part of the higher-level `Vectors` abstraction that this type
    /// is used to implement.
    ///
    /// Instead, implementors of this type behave like the pointer that
    /// `[Vector]::as_ptr()` would return, and their main purpose is to
    /// implement the `[Vector]::get_unchecked(idx)` operation of the slice,
    /// like `*ptr.add(idx)` would in a real slice.
    ///
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
    /// - Users of this trait may rely all method implementations to be correct
    ///   for safety.
    #[doc(hidden)]
    pub unsafe trait VectorSliceBase<V: VectorInfo>: Copy + Sized {
        /// Truth that this data may safely reinterpreted as a collection of
        /// Vectors using `as_vectors_unchecked()`
        fn is_vectors(self) -> bool;

        /// Result of calling `as_vectors_unchecked()`
        ///
        /// - Self for slices and collections of Vector
        /// - Corresponding slice or collection of Vector for scalar storage
        type AsVectors: VectorSliceBase<V>;

        /// Cast this to the equivalent slice or collection of Vector
        unsafe fn as_vectors_unchecked(self) -> Self::AsVectors;

        /// Attempt to cast this to the equivalent slice or collection of Vector
        #[inline(always)]
        fn as_vectors(self) -> Option<Self::AsVectors> {
            self.is_vectors()
                .then(|| unsafe { self.as_vectors_unchecked() })
        }

        /// Result of calling `get_unchecked()`
        ///
        /// - Vector for &[Vector], &[Scalar] and owned data
        /// - &mut Vector for &mut [Vector]
        /// - VectorMutProxy that behaves like &mut Vector for &mut [Scalar]
        type Element<'result>
        where
            Self: 'result;

        /// Access the underlying slice at vector index `idx` without bound or
        /// lifetime checking, adding scalar padding values if data is missing
        /// for the beginning or the end of the target vector.
        ///
        /// # Safety
        ///
        /// - The underlying slice must be valid for lifetime `'result`.
        /// - Index `idx` must be in within the bounds of the underlying slice.
        /// - If padding is needed, then `padding` must contain a valid scalar.
        unsafe fn get_unchecked<'result>(
            self,
            idx: usize,
            padding: MaybeUninit<V::Scalar>,
        ) -> Self::Element<'result>
        where
            Self: 'result;
    }

    /// *const T tagged with a lifetime
    #[derive(Copy, Clone)]
    #[doc(hidden)]
    pub struct VectorPtr<'target, T>(*const T, PhantomData<&'target [T]>);
    //
    impl<T> VectorPtr<'_, T> {
        /// Tag a *const T with lifetime information
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
    // *const Vector yields Vector values
    unsafe impl<'target, V: VectorInfo> VectorSliceBase<V> for VectorPtr<'target, V> {
        #[inline(always)]
        fn is_vectors(self) -> bool {
            true
        }

        type AsVectors = Self;

        #[inline(always)]
        unsafe fn as_vectors_unchecked(self) -> Self {
            self
        }

        type Element<'result> = V where Self: 'result;

        #[inline(always)]
        unsafe fn get_unchecked<'result>(
            self,
            idx: usize,
            _padding: MaybeUninit<V::Scalar>,
        ) -> Self::Element<'result>
        where
            Self: 'result,
        {
            unsafe { *self.0.add(idx) }
        }
    }

    // [Vector; SIZE] yields Vector values
    unsafe impl<V: VectorInfo, const SIZE: usize> VectorSliceBase<V> for [V; SIZE] {
        #[inline(always)]
        fn is_vectors(self) -> bool {
            true
        }

        type AsVectors = Self;

        #[inline(always)]
        unsafe fn as_vectors_unchecked(self) -> Self {
            self
        }

        type Element<'result> = V;

        #[inline(always)]
        unsafe fn get_unchecked<'result>(
            self,
            idx: usize,
            _padding: MaybeUninit<V::Scalar>,
        ) -> Self::Element<'result>
        where
            Self: 'result,
        {
            unsafe { *<[V]>::get_unchecked(&self[..], idx) }
        }
    }

    /// *mut T tagged with a lifetime
    #[derive(Copy, Clone)]
    #[doc(hidden)]
    pub struct VectorPtrMut<'target, T>(*mut T, PhantomData<&'target mut [T]>);
    //
    impl<T> VectorPtrMut<'_, T> {
        /// Tag a *mut T with lifetime information
        ///
        /// # Safety
        ///
        /// The pointer's target must be valid for this type's lifetime
        #[inline(always)]
        unsafe fn new(inner: *mut T) -> Self {
            Self(inner, PhantomData)
        }
    }
    //
    // *mut Vector yields &mut Vector
    unsafe impl<'target, V: VectorInfo> VectorSliceBase<V> for VectorPtrMut<'target, V> {
        #[inline(always)]
        fn is_vectors(self) -> bool {
            true
        }

        type AsVectors = Self;

        #[inline(always)]
        unsafe fn as_vectors_unchecked(self) -> Self {
            self
        }

        type Element<'result> = &'result mut V where Self: 'result;

        #[inline(always)]
        unsafe fn get_unchecked<'result>(
            self,
            idx: usize,
            _padding: MaybeUninit<V::Scalar>,
        ) -> Self::Element<'result>
        where
            Self: 'result,
        {
            unsafe { &mut *self.0.add(idx) }
        }
    }

    /// *const Vector approximation for access from &[Scalar]
    #[derive(Copy, Clone)]
    #[doc(hidden)]
    pub struct ScalarPtr<'target, V: VectorInfo> {
        start: *const V::Scalar,
        end: *const V::Scalar,
        _vector: PhantomData<&'target [V]>,
    }
    //
    impl<V: VectorInfo> ScalarPtr<'_, V> {
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
    }
    //
    // NOTE: Can't use V: VectorInfo bound here yet due to const generics limitations
    unsafe impl<'target, A: Align, B: Repr, const S: usize> VectorSliceBase<Vector<A, B, S>>
        for ScalarPtr<'target, Vector<A, B, S>>
    {
        #[inline(always)]
        fn is_vectors(self) -> bool {
            let is_aligned =
                |ptr: *const B| ptr as usize % core::mem::align_of::<Vector<A, B, S>>() == 0;
            is_aligned(self.start) && is_aligned(self.end)
        }

        type AsVectors = VectorPtr<'target, Vector<A, B, S>>;

        #[inline(always)]
        unsafe fn as_vectors_unchecked(self) -> Self::AsVectors {
            unsafe { VectorPtr::new(self.start.cast()) }
        }

        type Element<'result> = Vector<A, B, S> where Self: 'result;

        #[inline(always)]
        unsafe fn get_unchecked<'result>(
            self,
            idx: usize,
            padding: MaybeUninit<B>,
        ) -> Self::Element<'result>
        where
            Self: 'result,
        {
            let base_ptr = self.start.add(idx * S);
            core::array::from_fn(|offset| {
                let scalar_ptr = base_ptr.add(offset);
                if scalar_ptr < self.end {
                    unsafe { *scalar_ptr }
                } else {
                    unsafe { padding.assume_init() }
                }
            })
            .into()
        }
    }

    // NOTE: Can't implement support for [Scalar; SIZE] yet due to const
    //       generics limitations around AsVectorsResult (AsVectors should be
    //       [Vector; { SIZE / Vector::LANES }], but array lengths derived from
    //       generic parameters are not allowed yet.

    /// *mut Vector approximation for access from &mut [Scalar]
    #[derive(Copy, Clone)]
    #[doc(hidden)]
    pub struct ScalarPtrMut<'target, V: VectorInfo> {
        start: *mut V::Scalar,
        end: *mut V::Scalar,
        _vector: PhantomData<&'target mut [V]>,
    }
    //
    impl<V: VectorInfo> ScalarPtrMut<'_, V> {
        /// Build from scalar slice raw parts
        ///
        /// # Safety
        ///
        /// - `data` must be valid for the lifetime of this tagged pointer.
        /// - `len` must not go past the end of `data`'s allocation.
        #[inline(always)]
        unsafe fn new(data: *mut V::Scalar, len: usize) -> Self {
            Self {
                start: data,
                end: unsafe { data.add(len) },
                _vector: PhantomData,
            }
        }
    }
    //
    // NOTE: Can't use V: VectorInfo bound here yet due to const generics limitations
    unsafe impl<'target, A: Align, B: Repr, const S: usize> VectorSliceBase<Vector<A, B, S>>
        for ScalarPtrMut<'target, Vector<A, B, S>>
    {
        #[inline(always)]
        fn is_vectors(self) -> bool {
            let is_aligned =
                |ptr: *mut B| ptr as usize % core::mem::align_of::<Vector<A, B, S>>() == 0;
            is_aligned(self.start) && is_aligned(self.end)
        }

        type AsVectors = VectorPtrMut<'target, Vector<A, B, S>>;

        #[inline(always)]
        unsafe fn as_vectors_unchecked(self) -> Self::AsVectors {
            unsafe { VectorPtrMut::new(self.start.cast()) }
        }

        type Element<'result> = VectorMutProxy<'result, Vector<A, B, S>> where Self: 'result;

        #[inline(always)]
        unsafe fn get_unchecked<'result>(
            self,
            idx: usize,
            padding: MaybeUninit<B>,
        ) -> Self::Element<'result>
        where
            Self: 'result,
        {
            let base_ptr = self.start.add(idx * S);
            VectorMutProxy {
                vector: core::array::from_fn(|offset| {
                    let scalar_ptr = base_ptr.add(offset);
                    if scalar_ptr < self.end {
                        unsafe { *scalar_ptr }
                    } else {
                        unsafe { padding.assume_init() }
                    }
                })
                .into(),
                target: core::slice::from_raw_parts_mut(base_ptr, unsafe {
                    self.end.offset_from(base_ptr)
                } as usize),
            }
        }
    }

    /// Vector mutation proxy for scalar slices
    ///
    /// For mutation from &mut [Scalar], we can't provide an &mut Vector as it
    /// could be misaligned and out of bounds. So we provide a proxy object
    /// that acts as closely to &mut Vector as possible.
    pub struct VectorMutProxy<'target, V: VectorInfo> {
        vector: V,
        target: &'target mut [V::Scalar],
    }
    //
    impl<V: VectorInfo> Deref for VectorMutProxy<'_, V> {
        type Target = V;

        fn deref(&self) -> &V {
            &self.vector
        }
    }
    //
    impl<V: VectorInfo> DerefMut for VectorMutProxy<'_, V> {
        fn deref_mut(&mut self) -> &mut V {
            &mut self.vector
        }
    }
    //
    impl<V: VectorInfo> Drop for VectorMutProxy<'_, V> {
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
            #[allow(non_snake_case)]
            unsafe impl<
                'target,
                V: VectorInfo
                $(, $t: VectorSliceBase<V> + 'target)*
            > VectorSliceBase<V> for ($($t,)*) {
                #[inline(always)]
                fn is_vectors(self) -> bool {
                    let ($($t,)*) = self;
                    $(
                        if !$t.is_vectors() {
                            return false;
                        }
                    )*
                    true
                }

                type AsVectors = ($($t::AsVectors,)*);

                #[inline(always)]
                unsafe fn as_vectors_unchecked(self) -> Self::AsVectors {
                    let ($($t,)*) = self;
                    unsafe { ($($t.as_vectors_unchecked(),)*) }
                }

                type Element<'result> = ($($t::Element<'result>,)*) where Self: 'result;

                #[inline(always)]
                unsafe fn get_unchecked<'result>(
                    self,
                    idx: usize,
                    padding: MaybeUninit<V::Scalar>
                ) -> Self::Element<'result>
                    where Self: 'result
                {
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
    /// with iteration and indexing operations yielding the following types:
    ///
    /// - If built out of a read-only slice or owned container of vectors or
    ///   scalars, yield owned `Vector`s of data.
    /// - If built out of `&mut [Vector]`, yield `&mut Vector` references.
    /// - If built out of `&mut [Scalar]`, yield `VectorMutProxy`, a proxy type
    ///   which can be used like an `&mut Vector` (but cannot be literally
    ///   `&mut Vector` because scalars are not vectors)
    /// - If built out of a tuple of the above entities, yield tuples of the
    ///   aforementioned elements.
    pub struct Vectors<V: VectorInfo, Base: VectorSliceBase<V>> {
        base: Base,
        len: usize,
        padding: MaybeUninit<V::Scalar>,
    }
    //
    impl<V: VectorInfo, Base: VectorSliceBase<V>> Vectors<V, Base> {
        /// Create a SIMD data container
        ///
        /// # Safety
        ///
        /// - It must be safe to dereference ptr for any index in 0..len
        ///   during the lifetime 'source.
        /// - If any of the inner pointers require padding, then padding must
        ///   be initialized to a valid padding value.
        #[inline(always)]
        unsafe fn new(base: Base, len: usize, padding: MaybeUninit<V::Scalar>) -> Self {
            Self { base, len, padding }
        }

        /// Access the N-th element of the container
        ///
        /// See [the top-level type description](`Vectors`) to know what type
        /// of element this operation yields.
        ///
        /// # Safety
        ///
        /// `idx` must be in range `0..self.len()`
        #[inline(always)]
        pub unsafe fn get_unchecked(&mut self, idx: usize) -> Base::Element<'_> {
            unsafe { self.base.get_unchecked(idx, self.padding) }
        }

        /// Get the number of elements stored in this container
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.len
        }

        // TODO: Implement iter, IntoIterator, r?(array_)?chunks(_exact)?,
        //       first, get, is_empty, last, r?splitn?, split_at(_unchecked)?
        //       split_(first|last), split_inclusive, r?split_array, windows, Index
    }

    // === Step 3: Translate from scalar slices and containers to vector slices ===

    /// Trait for data that can be processed using SIMD
    ///
    /// Implemented for slices and containers of vectors and scalars,
    /// as well as for tuples of these entities.
    ///
    /// Provides you with ways to create the `Vectors` proxy type, which
    /// behaves conceptually like an array of `Vector` or tuples thereof, with
    /// iteration and indexing operations yielding the following types:
    ///
    /// - If built out of a read-only slice or owned container of vectors or
    ///   scalars, yield owned `Vector`s of data.
    /// - If built out of `&mut [Vector]`, yield `&mut Vector` references.
    /// - If built out of `&mut [Scalar]`, yield `VectorMutProxy`, a proxy type
    ///   which can be used like an `&mut Vector` (but cannot be literally
    ///   `&mut Vector` because scalars are not vectors)
    /// - If built out of a tuple of the above entities, yield tuples of the
    ///   aforementioned elements.
    ///
    /// There are two ways to create `Vectors` using this trait depending on
    /// what kind of data you're starting from:
    ///
    /// - If starting out of arbitrary data, you can use [`vectorize()`] or
    ///   [`vectorize_pad()`] function to get to a SIMD view of that data that
    ///   is optimal in absence of further assumptions.
    /// - If you are starting out with scalar data that you know to be actually
    ///   prepared in a fashion that is optimal for SIMD processing, then you
    ///   can use [`as_vectors()`] to make this fact known to the implementation
    ///   so that it can produce a more optimized `Vectors` container.
    ///
    /// # Safety
    ///
    /// Unsafe code may rely on the implementation being correct.
    ///
    /// [`vectorize()`]: Vectorizable::vectorize()
    /// [`vectorize_pad()`]: Vectorizable::vectorize_pad()
    /// [`as_vectors()`]: Vectorizable::as_vectors()
    pub unsafe trait Vectorizable<V: VectorInfo>: Sized {
        /// Internal mechanism used to treat this data as a slice of vectors
        type VectorSliceBase: VectorSliceBase<V>;

        // Required methods

        /// Internal method to prepare treating this data as a slice of vectors
        //
        // --- Internal docs starts here ---
        //
        // The returned building blocks are...
        //
        // - A pointer-like entity for treating the data as a slice of Vector
        //   (see VectorSliceBase for more information)
        // - The number of Vector elements that this data contains
        // - The truth that this data needs padding
        //
        // This panics if called on a tuple of slices/containers who would not
        // produce the same amount of Vector elements.
        fn prepare_vectors(self) -> (Self::VectorSliceBase, usize, bool);

        // Provided methods

        /// Create a SIMD view of this data, asserting it doesn't need padding
        ///
        /// # Panics
        ///
        /// - If called on a scalar slice whose length is not a multiple of the
        ///   number of SIMD vector lanes (you need `vectorize_pad()`)
        /// - If called on a tuple and not all tuple elements yield the same
        ///   amount of SIMD elements.
        #[inline(always)]
        fn vectorize(self) -> Vectors<V, Self::VectorSliceBase> {
            let (base, len, needs_padding) = self.prepare_vectors();
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
        /// For scalar sizes whose size is not a multiple of the number of SIMD
        /// vector lanes, padding will be inserted where incomplete Vectors
        /// would be produced, to fill in the missing vector lanes.
        ///
        /// # Panics
        ///
        /// - If called on a tuple and not all tuple elements yield the same
        ///   amount of SIMD elements.
        fn vectorize_pad(self, padding: V::Scalar) -> Vectors<V, Self::VectorSliceBase> {
            let (base, len, _needs_padding) = self.prepare_vectors();
            unsafe { Vectors::new(base, len, MaybeUninit::new(padding)) }
        }

        /// Assert that this data is in a layout optimized for SIMD processing
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
        /// allow the implementation to perform more optimizations, and thus
        /// better run-time performance can be achieved.
        ///
        /// # Panics
        ///
        /// - If the data is not in a SIMD-optimized layout.
        /// - If called on a tuple and not all tuple elements yield the same
        ///   amount of SIMD elements.
        #[inline(always)]
        fn as_vectors(
            self,
        ) -> Vectors<V, <Self::VectorSliceBase as VectorSliceBase<V>>::AsVectors> {
            let (base, len, needs_padding) = self.prepare_vectors();
            let base = base
                .as_vectors()
                .expect("Data is not in a SIMD-friendly layout");
            debug_assert!(!needs_padding, "SIMD-friendly data should not need padding");
            unsafe { Vectors::new(base, len, MaybeUninit::uninit()) }
        }
    }

    // === Vectorize implementation is trivial for slices of vector data ===

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target [Vector<A, B, S>]
    {
        type VectorSliceBase = VectorPtr<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn prepare_vectors(self) -> (Self::VectorSliceBase, usize, bool) {
            (unsafe { VectorPtr::new(self.as_ptr()) }, self.len(), false)
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target mut [Vector<A, B, S>]
    {
        type VectorSliceBase = VectorPtrMut<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn prepare_vectors(self) -> (Self::VectorSliceBase, usize, bool) {
            (
                unsafe { VectorPtrMut::new(self.as_mut_ptr()) },
                self.len(),
                false,
            )
        }
    }

    unsafe impl<V: VectorInfo, const SIZE: usize> Vectorizable<V> for [V; SIZE] {
        type VectorSliceBase = [V; SIZE];

        #[inline(always)]
        fn prepare_vectors(self) -> ([V; SIZE], usize, bool) {
            (self, SIZE, false)
        }
    }

    // === For scalar data, must cautiously handle padding and alignment ===

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target [B]
    {
        type VectorSliceBase = ScalarPtr<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn prepare_vectors(self) -> (Self::VectorSliceBase, usize, bool) {
            let needs_padding = self.len() % S != 0;
            (
                unsafe { ScalarPtr::new(self.as_ptr(), self.len()) },
                self.len() / S + needs_padding as usize,
                needs_padding,
            )
        }
    }

    unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
        for &'target mut [B]
    {
        type VectorSliceBase = ScalarPtrMut<'target, Vector<A, B, S>>;

        #[inline(always)]
        fn prepare_vectors(self) -> (Self::VectorSliceBase, usize, bool) {
            let needs_padding = self.len() % S != 0;
            (
                unsafe { ScalarPtrMut::new(self.as_mut_ptr(), self.len()) },
                self.len() / S + needs_padding as usize,
                needs_padding,
            )
        }
    }

    // NOTE: Cannot be implemented for scalar arrays yet due to const generics
    //       limitations around as_vectors().

    // === Tuples must have homogeneous length and vector type ===

    macro_rules! impl_vectorize_for_tuple {
        (
            $($t:ident),*
        ) => {
            #[allow(non_snake_case)]
            unsafe impl<V: VectorInfo $(, $t: Vectorizable<V>)*> Vectorizable<V> for ($($t,)*) {
                type VectorSliceBase = ($($t::VectorSliceBase,)*);

                #[inline(always)]
                fn prepare_vectors(
                    self,
                ) -> (Self::VectorSliceBase, usize, bool) {
                    // Pattern-match the tuple to variables named after inner types
                    let ($($t,)*) = self;

                    // Reinterpret tuple fields as SIMD vectors
                    let ($($t,)*) = ($($t.prepare_vectors(),)*);

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
                        if !t_needs_padding {
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
