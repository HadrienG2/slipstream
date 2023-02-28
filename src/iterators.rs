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

mod experiment {
    use crate::{inner::Repr, vector::align::Align, Vector};
    use core::{
        marker::PhantomData,
        mem::MaybeUninit,
        ops::{Deref, DerefMut},
    };

    // === Step 1: Abstraction over SIMD data access ===

    /// Query some of the configuration of a SIMD type knowing only the type
    pub trait SIMD: AsRef<[Self::Scalar]> + Copy + Sized + 'static {
        type Scalar: Copy + Sized + 'static;
        const LANES: usize;
    }
    //
    impl<A: Align, B: Repr, const S: usize> SIMD for Vector<A, B, S> {
        type Scalar = B;
        const LANES: usize = S;
    }

    /// Stuff that can be treated as a pointer to SIMD data
    #[doc(hidden)]
    pub trait VectorPtr<V: SIMD>: Copy + Sized + 'static {
        /// Result of dereferencing this pointer
        type DerefResult<'target>
        where
            Self: 'target;

        /// Shift the pointer forward by `idx` SIMD vectors then dereference it,
        /// adding scalar padding values if data is missing at the end
        ///
        /// # Safety
        ///
        /// If padding is needed, then `padding` must contain a valid scalar.
        unsafe fn deref_at<'target>(
            self,
            idx: usize,
            padding: MaybeUninit<V::Scalar>,
        ) -> Self::DerefResult<'target>;
    }

    // *const Vector yields Vector values
    impl<V: SIMD> VectorPtr<V> for *const V {
        type DerefResult<'target> = V;

        #[inline(always)]
        unsafe fn deref_at<'target>(
            self,
            idx: usize,
            _padding: MaybeUninit<V::Scalar>,
        ) -> Self::DerefResult<'target> {
            unsafe { *self.add(idx) }
        }
    }

    // *mut Vector yields &mut Vector
    impl<V: SIMD> VectorPtr<V> for *mut V {
        type DerefResult<'target> = &'target mut V;

        #[inline(always)]
        unsafe fn deref_at<'target>(
            self,
            idx: usize,
            _padding: MaybeUninit<V::Scalar>,
        ) -> Self::DerefResult<'target> {
            unsafe { &mut *self.add(idx) }
        }
    }

    /// *const Vector approximation for access from &[Scalar]
    #[derive(Copy, Clone)]
    #[doc(hidden)]
    pub struct ScalarPtr<V: SIMD> {
        data: *const V::Scalar,
        len: usize,
        _vector: PhantomData<V>,
    }
    //
    // NOTE: Can't use V: SIMD bound here yet due to const generics limitations
    impl<A: Align, B: Repr, const S: usize> VectorPtr<Vector<A, B, S>> for ScalarPtr<Vector<A, B, S>> {
        type DerefResult<'target> = Vector<A, B, S>;

        #[inline(always)]
        unsafe fn deref_at<'target>(
            self,
            idx: usize,
            padding: MaybeUninit<B>,
        ) -> Self::DerefResult<'target> {
            let base_idx = idx * S;
            let base_ptr = self.data.add(base_idx);
            core::array::from_fn(|offset| {
                if base_idx + offset < self.len {
                    unsafe { *base_ptr.add(offset) }
                } else {
                    unsafe { padding.assume_init() }
                }
            })
            .into()
        }
    }

    /// *mut Vector approximation for access from &mut [Scalar]
    #[derive(Copy, Clone)]
    #[doc(hidden)]
    pub struct ScalarPtrMut<V: SIMD> {
        data: *mut V::Scalar,
        len: usize,
        _vector: PhantomData<V>,
    }
    //
    // NOTE: Can't use V: SIMD bound here yet due to const generics limitations
    impl<A: Align, B: Repr, const S: usize> VectorPtr<Vector<A, B, S>>
        for ScalarPtrMut<Vector<A, B, S>>
    {
        type DerefResult<'target> = ScalarMutProxy<'target, Vector<A, B, S>>;

        #[inline(always)]
        unsafe fn deref_at<'target>(
            self,
            idx: usize,
            padding: MaybeUninit<B>,
        ) -> Self::DerefResult<'target> {
            let base_idx = idx * S;
            let base_ptr = self.data.add(base_idx);
            ScalarMutProxy {
                vector: core::array::from_fn(|offset| {
                    if base_idx + offset < self.len {
                        unsafe { *base_ptr.add(offset) }
                    } else {
                        unsafe { padding.assume_init() }
                    }
                })
                .into(),
                target: core::slice::from_raw_parts_mut(base_ptr, self.len - base_idx),
            }
        }
    }

    /// SIMD mutation proxy for scalar slices
    ///
    /// For mutation from &mut [Scalar], we can't provide an &mut Vector as it
    /// would be misaligned and out of bounds, so we provide a proxy object
    /// that mostly acts like &mut Vector instead.
    pub struct ScalarMutProxy<'target, V: SIMD> {
        vector: V,
        target: &'target mut [V::Scalar],
    }
    //
    impl<V: SIMD> Deref for ScalarMutProxy<'_, V> {
        type Target = V;

        fn deref(&self) -> &V {
            &self.vector
        }
    }
    //
    impl<V: SIMD> DerefMut for ScalarMutProxy<'_, V> {
        fn deref_mut(&mut self) -> &mut V {
            &mut self.vector
        }
    }
    //
    impl<V: SIMD> Drop for ScalarMutProxy<'_, V> {
        fn drop(&mut self) {
            self.target
                .copy_from_slice(&self.vector.as_ref()[..self.target.len()]);
        }
    }

    /// Tuples of pointers yield tuples of deref results
    macro_rules! impl_vectorptr_for_tuple {
        (
            $($t:ident),*
        ) => {
            #[allow(non_snake_case)]
            impl<V: SIMD $(, $t: VectorPtr<V>)*> VectorPtr<V> for ($($t,)*) {
                type DerefResult<'target> = ($($t::DerefResult<'target>,)*);

                #[inline(always)]
                unsafe fn deref_at<'target>(self, idx: usize, padding: MaybeUninit<V::Scalar>) -> Self::DerefResult<'target> {
                    let ($($t,)*) = self;
                    unsafe { ($($t.deref_at(idx, padding),)*)  }
                }
            }
        };
    }
    impl_vectorptr_for_tuple!(A);
    impl_vectorptr_for_tuple!(A, B);
    impl_vectorptr_for_tuple!(A, B, C);
    impl_vectorptr_for_tuple!(A, B, C, D);
    impl_vectorptr_for_tuple!(A, B, C, D, E);
    impl_vectorptr_for_tuple!(A, B, C, D, E, F);
    impl_vectorptr_for_tuple!(A, B, C, D, E, F, G);
    impl_vectorptr_for_tuple!(A, B, C, D, E, F, G, H);

    // === Step 2: Optimized tuple of SIMD data slices ===

    /// Tuple of vector data that behaves like a slice of tuples of SIMD vectors
    ///
    /// Can be built from a tuple of scalar and/or SIMD data by using the
    /// Vectorize trait.
    pub struct Vectors<'source, V: SIMD, Ptr: VectorPtr<V>> {
        ptr: Ptr,
        len: usize,
        padding: MaybeUninit<V::Scalar>,
        _lifetime: PhantomData<&'source mut V>,
    }
    //
    impl<V: SIMD, Ptr: VectorPtr<V>> Vectors<'_, V, Ptr> {
        /// Create a tuple
        ///
        /// # Safety
        ///
        /// - It must be safe to dereference ptr for any index in 0..len
        ///   during the lifetime 'source.
        /// - If any of the inner pointers require padding, then padding must
        ///   be set to Some with a valid padding value.
        unsafe fn new(ptr: Ptr, len: usize, padding: Option<V::Scalar>) -> Self {
            Self {
                ptr,
                len,
                padding: if let Some(padding) = padding {
                    MaybeUninit::new(padding)
                } else {
                    MaybeUninit::uninit()
                },
                _lifetime: PhantomData,
            }
        }

        /// Cast to a different lifetime
        ///
        /// # Safety
        ///
        /// This operation is only meant for delegating from one Vectorize
        /// implementation to another, and can trivially lead to use-after-free
        /// undefined behavior if used in another context.
        unsafe fn extend_lifetime<'any>(self) -> Vectors<'any, V, Ptr> {
            unsafe { core::mem::transmute(self) }
        }

        /// Get a raw pointer to the underlying SIMD buffers
        fn as_ptr(&mut self) -> Ptr {
            self.ptr
        }

        /// Access the N-th elements of the underlying SIMD buffers
        ///
        /// # Safety
        ///
        /// `idx` must be in range `0..self.len()`
        #[inline(always)]
        pub unsafe fn get_unchecked(&mut self, idx: usize) -> Ptr::DerefResult<'_> {
            unsafe { self.ptr.deref_at(idx, self.padding) }
        }

        /// Get the length of the underlying data in SIMD vectors
        #[inline(always)]
        pub fn len(&self) -> usize {
            self.len
        }

        // TODO: Implement iter, IntoIterator, r?chunks(_exact)?,
        //       first, get, is_empty, last, r?splitn?, split_at(_unchecked)?
        //       split_(first|last), split_inclusive, windows, Index
    }

    // === Step 3: Translate from scalar slices and containers to vector slices ===

    /// Trait for data that can be treated as a slice of SIMD vectors
    ///
    /// Implemented for slices and containers of vectors and scalars,
    /// and for tuples of these entities.
    ///
    /// # Safety
    ///
    /// - `is_aligned()` must yield correct results, or the default
    ///   implementation of `assume_aligned()` will trigger undefined behavior.
    // TODO: More docs
    pub unsafe trait Vectorize<V: SIMD> {
        /// Pointer to this data reinterpreted as SIMD
        ///
        /// This is an implementation detail that you should not need to
        /// interact with as a user of this crate.
        type VectorPtr: VectorPtr<V>;

        /// Aligned view of this data
        ///
        /// This type is returned by `assume_aligned()` and can be used to tell
        /// the vectorizer that scalar data is SIMD aligned so that it can take
        /// better SIMD code generation decisions.
        type AlignedView<'view>: Vectorize<V>
        where
            Self: 'view;

        // Required methods

        /// Create a SIMD view of this data, with optional padding
        ///
        /// End users of the library will most likely want to use the
        /// `vectorize()` and `vectorize_pad()` shortcuts.
        ///
        /// # Panics
        ///
        /// - If scalar slice padding is needed and not provided.
        /// - If called on a tuple of slices of inhomogeneous SIMD length
        fn create(&mut self, padding: Option<V::Scalar>) -> Vectors<V, Self::VectorPtr>;

        /// Truth that this data is correctly aligned for SIMD vector type `V`
        /// and has a number of scalar elements that is divisible by `V::LANES`.
        fn is_aligned(&self) -> bool;

        /// Returns a view of the data that also implements Vectorize, but in a
        /// manner that lets the vectorizer assume that `is_aligned()` is true.
        ///
        /// This enables the vectorizer to generate simpler and faster code.
        ///
        /// In most cases, you will want to use the safe `as_aligned()`
        /// alternative instead.
        ///
        /// # Safety
        ///
        /// is_aligned() must be true or Undefined Behavior will ensue.
        unsafe fn as_aligned_unchecked(&mut self) -> Self::AlignedView<'_>;

        // Provided methods

        /// Create a SIMD view of this data, asserting it doesn't need padding
        ///
        /// # Panics
        ///
        /// - If scalar slice padding is needed.
        /// - If called on a tuple of slices of inhomogeneous SIMD length
        fn vectorize(&mut self) -> Vectors<V, Self::VectorPtr> {
            self.create(None)
        }

        /// Create a SIMD view of this data, providing some padding
        ///
        /// Vector slices and scalar slices whose size is a multiple of the
        /// number of SIMD vector lanes, do not need padding and will ignore it.
        ///
        /// # Panics
        ///
        /// - If called on a tuple of slices of inhomogeneous SIMD length
        fn vectorize_pad(&mut self, padding: V::Scalar) -> Vectors<V, Self::VectorPtr> {
            self.create(Some(padding))
        }

        /// Assert that `is_aligned()` is true and returns a view of the data
        /// that makes it known to the vectorizer.
        ///
        /// This enables the vectorizer to generate simpler and faster code.
        ///
        /// # Panics
        ///
        /// - If `is_aligned()` is false.
        #[inline(always)]
        fn as_aligned(&mut self) -> Self::AlignedView<'_> {
            assert!(self.is_aligned(), "Data is not SIMD aligned");
            unsafe { self.as_aligned_unchecked() }
        }
    }

    // === Vectorize implementation is trivial for slices of vector data ===

    unsafe impl<A: Align, B: Repr, const S: usize> Vectorize<Vector<A, B, S>> for &[Vector<A, B, S>] {
        type VectorPtr = *const Vector<A, B, S>;

        #[inline(always)]
        fn create(&mut self, padding: Option<B>) -> Vectors<Vector<A, B, S>, Self::VectorPtr> {
            unsafe { Vectors::new(self.as_ptr(), self.len(), padding) }
        }

        #[inline(always)]
        fn is_aligned(&self) -> bool {
            true
        }

        type AlignedView<'view> = &'view [Vector<A, B, S>] where Self: 'view;

        #[inline(always)]
        unsafe fn as_aligned_unchecked(&mut self) -> Self::AlignedView<'_> {
            self
        }
    }

    unsafe impl<A: Align, B: Repr, const S: usize> Vectorize<Vector<A, B, S>>
        for &mut [Vector<A, B, S>]
    {
        type VectorPtr = *mut Vector<A, B, S>;

        #[inline(always)]
        fn create(&mut self, padding: Option<B>) -> Vectors<Vector<A, B, S>, Self::VectorPtr> {
            unsafe { Vectors::new(self.as_mut_ptr(), self.len(), padding) }
        }

        #[inline(always)]
        fn is_aligned(&self) -> bool {
            true
        }

        type AlignedView<'view> = &'view mut [Vector<A, B, S>] where Self: 'view;

        #[inline(always)]
        unsafe fn as_aligned_unchecked(&mut self) -> Self::AlignedView<'_> {
            self
        }
    }

    unsafe impl<const ARRAY_SIZE: usize, A: Align, B: Repr, const S: usize>
        Vectorize<Vector<A, B, S>> for [Vector<A, B, S>; ARRAY_SIZE]
    {
        type VectorPtr = *const Vector<A, B, S>;

        #[inline(always)]
        fn create(&mut self, padding: Option<B>) -> Vectors<Vector<A, B, S>, Self::VectorPtr> {
            unsafe { (&self[..]).create(padding).extend_lifetime() }
        }

        #[inline(always)]
        fn is_aligned(&self) -> bool {
            true
        }

        type AlignedView<'view> = &'view [Vector<A, B, S>] where Self: 'view;

        #[inline(always)]
        unsafe fn as_aligned_unchecked(&mut self) -> Self::AlignedView<'_> {
            &self[..]
        }
    }

    // === For scalar data, must cautiously handle padding and alignment ===

    unsafe impl<A: Align, B: Repr, const S: usize> Vectorize<Vector<A, B, S>> for &[B] {
        type VectorPtr = ScalarPtr<Vector<A, B, S>>;

        #[inline(always)]
        fn create(&mut self, padding: Option<B>) -> Vectors<Vector<A, B, S>, Self::VectorPtr> {
            assert!(
                self.len() % S == 0 || padding.is_some(),
                "Padding must be provided for this slice"
            );
            unsafe {
                Vectors::new(
                    ScalarPtr {
                        data: self.as_ptr(),
                        len: self.len(),
                        _vector: PhantomData,
                    },
                    self.len() / S + (self.len() % S != 0) as usize,
                    padding,
                )
            }
        }

        #[inline(always)]
        fn is_aligned(&self) -> bool {
            let misalignment = (self.as_ptr() as usize) % core::mem::align_of::<Vector<A, B, S>>();
            (misalignment == 0) && (self.len() % S == 0)
        }

        type AlignedView<'view> = &'view [Vector<A, B, S>] where Self: 'view;

        #[inline(always)]
        unsafe fn as_aligned_unchecked(&mut self) -> Self::AlignedView<'_> {
            unsafe {
                core::slice::from_raw_parts(self.as_ptr() as *const Vector<A, B, S>, self.len() / S)
            }
        }
    }

    unsafe impl<A: Align, B: Repr, const S: usize> Vectorize<Vector<A, B, S>> for &mut [B] {
        type VectorPtr = ScalarPtrMut<Vector<A, B, S>>;

        #[inline(always)]
        fn create(&mut self, padding: Option<B>) -> Vectors<Vector<A, B, S>, Self::VectorPtr> {
            assert!(
                self.len() % S == 0 || padding.is_some(),
                "Padding must be provided for this slice"
            );
            unsafe {
                Vectors::new(
                    ScalarPtrMut {
                        data: self.as_mut_ptr(),
                        len: self.len(),
                        _vector: PhantomData,
                    },
                    self.len() / S + (self.len() % S != 0) as usize,
                    padding,
                )
            }
        }

        #[inline(always)]
        fn is_aligned(&self) -> bool {
            <&[B] as Vectorize<Vector<A, B, S>>>::is_aligned(&&**self)
        }

        type AlignedView<'view> = &'view mut [Vector<A, B, S>] where Self: 'view;

        #[inline(always)]
        unsafe fn as_aligned_unchecked(&mut self) -> Self::AlignedView<'_> {
            unsafe {
                core::slice::from_raw_parts_mut(
                    self.as_mut_ptr() as *mut Vector<A, B, S>,
                    self.len() / S,
                )
            }
        }
    }

    unsafe impl<const ARRAY_SIZE: usize, A: Align, B: Repr, const S: usize>
        Vectorize<Vector<A, B, S>> for [B; ARRAY_SIZE]
    {
        type VectorPtr = ScalarPtr<Vector<A, B, S>>;

        #[inline(always)]
        fn create(&mut self, padding: Option<B>) -> Vectors<Vector<A, B, S>, Self::VectorPtr> {
            unsafe { (&self[..]).create(padding).extend_lifetime() }
        }

        #[inline(always)]
        fn is_aligned(&self) -> bool {
            <&[B] as Vectorize<Vector<A, B, S>>>::is_aligned(&&self[..])
        }

        type AlignedView<'view> = &'view [Vector<A, B, S>] where Self: 'view;

        #[inline(always)]
        unsafe fn as_aligned_unchecked(&mut self) -> Self::AlignedView<'_> {
            unsafe { extend_slice_lifetime((&self[..]).as_aligned_unchecked()) }
        }
    }

    /// Cast slice to a different lifetime
    ///
    /// # Safety
    ///
    /// This operation is only meant for delegating from one Vectorize
    /// implementation to another, and can trivially lead to use-after-free
    /// undefined behavior if used in another context.
    unsafe fn extend_slice_lifetime<'input, 'output, T>(s: &'input [T]) -> &'output [T] {
        unsafe { core::mem::transmute(s) }
    }

    // === Tuples must have homogeneous length and vector type ===

    macro_rules! impl_vectorize_for_tuple {
        (
            $($t:ident),*
        ) => {
            #[allow(non_snake_case)]
            unsafe impl<V: SIMD $(, $t: Vectorize<V>)*> Vectorize<V> for ($($t,)*) {
                type VectorPtr = ($($t::VectorPtr,)*);

                #[inline(always)]
                fn create(&mut self, padding: Option<V::Scalar>) -> Vectors<V, Self::VectorPtr> {
                    // Pattern-match the tuple to variables named after inner types
                    let ($($t,)*) = self;

                    // Reinterpret tuple fields as SIMD vectors
                    let ($(mut $t,)*) = ($($t.create(padding),)*);

                    // Check that tuple field length is homogeneous
                    let mut len = None;
                    $(
                        #[allow(unused_assignments)]
                        if let Some(len) = len {
                            assert_eq!($t.len(), len, "Slice lengths are not homogeneous");
                        } else {
                            len = Some($t.len());
                        }
                    )*
                    let len = len.expect("Cannot implement this for empty tuples");

                    // Return optimized tuple of vector slices
                    unsafe { Vectors::new(($($t.as_ptr(),)*), len, padding) }
                }

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

                type AlignedView<'view> = ($($t::AlignedView<'view>,)*) where Self: 'view;

                #[inline(always)]
                unsafe fn as_aligned_unchecked(&mut self) -> Self::AlignedView<'_> {
                    let ($($t,)*) = self;
                    unsafe { (
                        $($t.as_aligned_unchecked(),)*
                    ) }
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
