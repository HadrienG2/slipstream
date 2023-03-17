//! SIMD data access
//!
//! This module implements the VectorizedData trait family, whose purpose is to
//! let you treat slices and containers of `Vector`s and scalar elements as if
//! they were slices of `Vector`s.
//!
//! It is focused on the low-level unsafe plumbing needed to perform this data
//! reinterpretation. The high-level API that users invoke to manipulate the
//! output slice is mostly defined in the `vectors` sibling module.

mod mutation;

use super::{VectorInfo, VectorizeError};
#[cfg(doc)]
use crate::{
    vectorize::{Vectorizable, Vectorized},
    Vector,
};
use core::{
    borrow::Borrow,
    fmt::{self, Debug, Pointer},
    hash::Hash,
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::NonNull,
};

pub use mutation::{PaddedMut, UnalignedMut};

/// Outcome of reinterpreting [`Vectorizable`] data as [`Vectorized`]
///
/// This trait tells you what types you can expect out of the [`Vectorized`]
/// collection that is emitted by the [`Vectorizable`] trait.
///
/// To recapitulate the basic rules concerning [`Element`](Self::Element) are:
///
/// - For all supported types T, a [`Vectorized`] collection built out of `&[T]`
///   or a collection of T has iterators and getters that emit owned [`Vector`]
///   values.
/// - If built out of `&mut [Vector]`, or out of `&mut [Scalar]` with an
///   assertion that the data has optimal SIMD layout
///   (see [`Vectorizable::vectorize_aligned()`]), it emits `&mut Vector`.
/// - If built out of `&mut [Scalar]` without asserting optimal SIMD layout,
///   it emits a proxy type that emulates `&mut Vector`.
/// - If build out of a tuple of the above, it emits tuples of the above
///   element types.
///
/// All [`Vectorized`] getters that take `&self` emit owned [`Vector`] values or
/// tuples thereof, called [`ElementCopy`](Self::ElementCopy), while those that
/// take `&mut self` emit shorter-lived versions of the `Element` type described
/// above, called [`ElementRef`](Self::ElementRef).
///
/// # Safety
///
/// Unsafe code may rely on the correctness of implementations of this trait
/// as part of their safety proofs. The definition of "correct" is an
/// implementation detail of this crate, therefore this trait should not
/// be implemented outside of this crate.
pub unsafe trait VectorizedData<V: VectorInfo>: Sized {
    /// Owned element of the output [`Vectorized`] collection
    ///
    /// Yielded by the `IntoIterator` impl that consumes the collection.
    type Element: Sized;

    /// Borrowed element of the output [`Vectorized`] collection
    ///
    /// Returned by methods that take `&mut Vectorized` and yield individual
    /// [`Vector`] or `&mut Vector` elements.
    ///
    /// Will always be [`Self::Element`] with a reduced lifetime, but this
    /// constraint cannot be expressed in current Rust.
    type ElementRef<'result>: Sized
    where
        Self: 'result;

    /// Copy of an element of the output [`Vectorized`] collection
    ///
    /// Will either be a single [`Vector`] value (for slice-like data) or a
    /// tuple of [`Vector`] values whose length match that of the `Element` tuple.
    type ElementCopy: Copy + Sized;

    /// Mutably borrowed slice of this dataset
    ///
    /// Returned by methods that borrow a subset of `&mut Vectorized`.
    type RefSlice<'result>: VectorizedData<V, Element = Self::ElementRef<'result>, ElementCopy = Self::ElementCopy>
        + VectorizedSliceImpl<V>
        + Debug
    where
        Self: 'result;

    /// Read-only slice of this dataset
    ///
    /// Returned by methods that borrow a subset of `&Vectorized`.
    type CopySlice<'result>: VectorizedData<V, Element = Self::ElementCopy, ElementCopy = Self::ElementCopy>
        + VectorizedSliceImpl<V>
        + Copy
        + Debug
    where
        Self: 'result;
}

/// Entity that can be treated as the base pointer of an &[Vector] or
/// &mut [Vector] slice
///
/// Implementors of this trait operate in the context of an underlying real
/// or simulated slice of SIMD vectors, or of a tuple of several slices of
/// equal length that is made to behave like a slice of tuples.
///
/// The length of the underlying slice is not known by this type, it is
/// stored as part of the higher-level `Vectorized` collection that this type
/// is used to implement.
///
/// Instead, implementors of this type behave like the pointer that
/// `[Vector]::as_ptr()` would return, and their main purpose is to
/// implement the `[Vector]::get_unchecked(idx)` operations of the slice,
/// like `*ptr.add(idx)` would in a real slice.
///
/// # Safety
///
/// Unsafe code may rely on the correctness of implementations of this trait
/// and the higher-level `VectorizedData` trait as part of their safety proofs.
///
/// The safety preconditions on `VectorizedData` are that `Element` should
/// not outlive `Self`, and that it should be safe to transmute `ElementMut`
/// to `Element` in scenarios where either `Element` is `Copy` or the
/// transmute is abstracted in such a way that the user cannot abuse it to
/// get two copies of the same element. In other words, `Element` should be
/// the maximal-lifetime version of `ElementMut`.
///
/// Further, `Slice::ElementMut` should be pretty much the same GAT as
/// `Self::ElementMut`, with just a different `Self` lifetime bound.
///
/// Finally, a `VectorizedData` impl is only allowed to implement `Copy` if
/// the source container type is `Copy` (i.e. not &mut container).
#[doc(hidden)]
pub unsafe trait VectorizedDataImpl<V: VectorInfo>: VectorizedData<V> + Sized {
    /// Get a copy of the slice element at index `idx` without bounds
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
    unsafe fn get_unchecked(&self, idx: usize, is_last: bool) -> Self::ElementCopy;

    /// Like `get_unchecked`, but allows in-place mutation of the underlying
    /// dataset if possible (underlying slice or collection is &mut).
    ///
    /// # Safety
    ///
    /// - Index `idx` must be in within the bounds of the underlying slice.
    /// - `is_last` must be true if and only if the last element of the
    ///   slice is being accessed.
    unsafe fn get_unchecked_ref(&mut self, idx: usize, is_last: bool) -> Self::ElementRef<'_>;

    /// Turn this data into the equivalent read-only slice
    ///
    /// Lifetime-shrinking no-op if Self is already a read-only slice, but turns
    /// mutable data into read-only data and owned data into slices.
    fn as_slice(&self) -> Self::CopySlice<'_>;

    /// Turn this data into the equivalent slice, allowing in-place access
    ///
    /// Lifetime-shrinking no-op if Self is already a slice, but turns
    /// owned data into slices.
    fn as_ref_slice(&mut self) -> Self::RefSlice<'_>;

    /// Reinterpretation of this data as SIMD data that may not be aligned
    type Unaligned: VectorizedData<V> + VectorizedDataImpl<V>;

    /// Unsafely cast this data to the equivalent slice or collection of
    /// unaligned `Vector`s
    ///
    /// # Safety
    ///
    /// The underlying scalar data must have a number of elements that is
    /// a multiple of `V::LANES`.
    unsafe fn into_unaligned_unchecked(self) -> Self::Unaligned;

    /// Reinterpretation of this data as SIMD data with optimal layout
    type Aligned: VectorizedData<V> + VectorizedDataImpl<V>;

    /// Unsafely cast this data to the equivalent slice or collection of Vector.
    ///
    /// # Safety
    ///
    /// The underlying scalar data must be aligned like V and have a number
    /// of elements that is a multiple of `V::LANES`.
    unsafe fn into_aligned_unchecked(self) -> Self::Aligned;
}

/// `VectorizedDataImpl` that is a true slice, i.e. does not own its elements
/// and can be split
#[doc(hidden)]
pub unsafe trait VectorizedSliceImpl<V: VectorInfo>:
    VectorizedDataImpl<V> + Sized + Debug
{
    /// Construct an empty slice
    fn empty() -> Self;

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
    /// - `len` must be the length of the underlying `Vectorized` slice.
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
impl<V: VectorInfo> Borrow<NonNull<V>> for AlignedData<'_, V> {
    fn borrow(&self) -> &NonNull<V> {
        &self.0
    }
}
//
impl<V: VectorInfo> Debug for AlignedData<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <NonNull<V> as Debug>::fmt(&self.0, f)
    }
}
//
impl<V: VectorInfo> Eq for AlignedData<'_, V> {}
//
impl<V: VectorInfo> Hash for AlignedData<'_, V> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
//
impl<V: VectorInfo> Ord for AlignedData<'_, V> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
//
impl<V: VectorInfo, Ptr: Borrow<NonNull<V>>> PartialEq<Ptr> for AlignedData<'_, V> {
    fn eq(&self, other: &Ptr) -> bool {
        self.0 == *other.borrow()
    }
}
//
impl<V: VectorInfo, Ptr: Borrow<NonNull<V>>> PartialOrd<Ptr> for AlignedData<'_, V> {
    fn partial_cmp(&self, other: &Ptr) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(other.borrow())
    }
}
//
impl<V: VectorInfo> Pointer for AlignedData<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <NonNull<V> as Pointer>::fmt(&self.0, f)
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedData<V> for AlignedData<'target, V> {
    type Element = V;
    type ElementRef<'result> = V where Self: 'result;
    type ElementCopy = V;
    type RefSlice<'result> = AlignedData<'result, V> where Self: 'result;
    type CopySlice<'result> = AlignedData<'result, V> where Self: 'result;
}
//
unsafe impl<'target, V: VectorInfo> VectorizedDataImpl<V> for AlignedData<'target, V> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize, _is_last: bool) -> V {
        unsafe { *self.get_ptr(idx).as_ref() }
    }

    #[inline(always)]
    unsafe fn get_unchecked_ref(&mut self, idx: usize, is_last: bool) -> V {
        unsafe { self.get_unchecked(idx, is_last) }
    }

    #[inline(always)]
    fn as_slice(&self) -> AlignedData<V> {
        AlignedData(self.0, PhantomData)
    }

    #[inline(always)]
    fn as_ref_slice(&mut self) -> AlignedData<V> {
        self.as_slice()
    }

    type Unaligned = Self;

    unsafe fn into_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = Self;

    unsafe fn into_aligned_unchecked(self) -> Self {
        self
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for AlignedData<'target, V> {
    #[inline(always)]
    fn empty() -> Self {
        Self(NonNull::dangling(), PhantomData)
    }

    #[inline(always)]
    unsafe fn split_at_unchecked(self, mid: usize, _len: usize) -> (Self, Self) {
        let wrap = |ptr| Self(ptr, PhantomData);
        (wrap(self.0), wrap(self.get_ptr(mid)))
    }
}

// Owned arrays of Vector must be stored as-is in the Vectorized collection,
// but otherwise behave like &[Vector]
unsafe impl<V: VectorInfo, const SIZE: usize> VectorizedData<V> for [V; SIZE] {
    type Element = V;
    type ElementRef<'result> = V where Self: 'result;
    type ElementCopy = V;
    type RefSlice<'result> = AlignedData<'result, V> where Self: 'result;
    type CopySlice<'result> = AlignedData<'result, V> where Self: 'result;
}
//
unsafe impl<V: VectorInfo, const SIZE: usize> VectorizedDataImpl<V> for [V; SIZE] {
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize, _is_last: bool) -> V {
        unsafe { *<[V]>::get_unchecked(&self[..], idx) }
    }

    #[inline(always)]
    unsafe fn get_unchecked_ref(&mut self, idx: usize, is_last: bool) -> V {
        unsafe { self.get_unchecked(idx, is_last) }
    }

    #[inline(always)]
    fn as_slice(&self) -> AlignedData<V> {
        AlignedData::from(&self[..])
    }

    #[inline(always)]
    fn as_ref_slice(&mut self) -> AlignedData<V> {
        AlignedData::from(&self[..])
    }

    type Unaligned = Self;

    unsafe fn into_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = Self;

    unsafe fn into_aligned_unchecked(self) -> Self {
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
impl<V: VectorInfo> Borrow<NonNull<V>> for AlignedDataMut<'_, V> {
    fn borrow(&self) -> &NonNull<V> {
        self.0.borrow()
    }
}
//
impl<'target, V: VectorInfo> Debug for AlignedDataMut<'target, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <AlignedData<'target, V> as Debug>::fmt(&self.0, f)
    }
}
//
impl<V: VectorInfo> Eq for AlignedDataMut<'_, V> {}
//
impl<V: VectorInfo> Hash for AlignedDataMut<'_, V> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
//
impl<V: VectorInfo> Ord for AlignedDataMut<'_, V> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
//
impl<V: VectorInfo, Ptr: Borrow<NonNull<V>>> PartialEq<Ptr> for AlignedDataMut<'_, V> {
    fn eq(&self, other: &Ptr) -> bool {
        self.0 == *other.borrow()
    }
}
//
impl<V: VectorInfo, Ptr: Borrow<NonNull<V>>> PartialOrd<Ptr> for AlignedDataMut<'_, V> {
    fn partial_cmp(&self, other: &Ptr) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(other.borrow())
    }
}
//
impl<'target, V: VectorInfo> Pointer for AlignedDataMut<'target, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <AlignedData<'target, V> as Pointer>::fmt(&self.0, f)
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedData<V> for AlignedDataMut<'target, V> {
    type Element = &'target mut V;
    type ElementRef<'result> = &'result mut V where Self: 'result;
    type ElementCopy = V;
    type RefSlice<'result> = AlignedDataMut<'result, V> where Self: 'result;
    type CopySlice<'result> = AlignedData<'result, V> where Self: 'result;
}
//
unsafe impl<'target, V: VectorInfo> VectorizedDataImpl<V> for AlignedDataMut<'target, V> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize, is_last: bool) -> V {
        unsafe { self.0.get_unchecked(idx, is_last) }
    }

    #[inline(always)]
    unsafe fn get_unchecked_ref(&mut self, idx: usize, _is_last: bool) -> &mut V {
        unsafe { self.0.get_ptr(idx).as_mut() }
    }

    #[inline(always)]
    fn as_slice(&self) -> AlignedData<V> {
        self.0.as_slice()
    }

    #[inline(always)]
    fn as_ref_slice(&mut self) -> AlignedDataMut<V> {
        AlignedDataMut(self.0.as_ref_slice(), PhantomData)
    }

    type Unaligned = Self;

    unsafe fn into_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = Self;

    unsafe fn into_aligned_unchecked(self) -> Self {
        self
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for AlignedDataMut<'target, V> {
    #[inline(always)]
    fn empty() -> Self {
        Self(AlignedData::empty(), PhantomData)
    }

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
impl<V: VectorInfo> Borrow<NonNull<V::Array>> for UnalignedData<'_, V> {
    fn borrow(&self) -> &NonNull<V::Array> {
        &self.0
    }
}
//
impl<V: VectorInfo> Debug for UnalignedData<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <NonNull<V::Array> as Debug>::fmt(&self.0, f)
    }
}
//
impl<V: VectorInfo> Eq for UnalignedData<'_, V> {}
//
impl<V: VectorInfo> Hash for UnalignedData<'_, V> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
//
impl<V: VectorInfo> Ord for UnalignedData<'_, V> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
//
impl<V: VectorInfo, Ptr: Borrow<NonNull<V::Array>>> PartialEq<Ptr> for UnalignedData<'_, V> {
    fn eq(&self, other: &Ptr) -> bool {
        self.0 == *other.borrow()
    }
}
//
impl<V: VectorInfo, Ptr: Borrow<NonNull<V::Array>>> PartialOrd<Ptr> for UnalignedData<'_, V> {
    fn partial_cmp(&self, other: &Ptr) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(other.borrow())
    }
}
//
impl<V: VectorInfo> Pointer for UnalignedData<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <NonNull<V::Array> as Pointer>::fmt(&self.0, f)
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedData<V> for UnalignedData<'target, V> {
    type Element = V;
    type ElementRef<'result> = V where Self: 'result;
    type ElementCopy = V;
    type RefSlice<'result> = UnalignedData<'result, V> where Self: 'result;
    type CopySlice<'result> = UnalignedData<'result, V> where Self: 'result;
}
//
unsafe impl<'target, V: VectorInfo> VectorizedDataImpl<V> for UnalignedData<'target, V> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize, _is_last: bool) -> V {
        unsafe { *self.get_ptr(idx).as_ref() }.into()
    }

    #[inline(always)]
    unsafe fn get_unchecked_ref(&mut self, idx: usize, is_last: bool) -> V {
        unsafe { self.get_unchecked(idx, is_last) }
    }

    #[inline(always)]
    fn as_slice(&self) -> UnalignedData<V> {
        UnalignedData(self.0, PhantomData)
    }

    #[inline(always)]
    fn as_ref_slice(&mut self) -> UnalignedData<V> {
        self.as_slice()
    }

    type Unaligned = Self;

    unsafe fn into_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = AlignedData<'target, V>;

    unsafe fn into_aligned_unchecked(self) -> AlignedData<'target, V> {
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
    fn empty() -> Self {
        Self(NonNull::dangling(), PhantomData)
    }

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
// the number of **complete** SIMD vectors within the underlying scalar slice.
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
impl<V: VectorInfo> Borrow<NonNull<V::Array>> for UnalignedDataMut<'_, V> {
    fn borrow(&self) -> &NonNull<V::Array> {
        self.0.borrow()
    }
}
//
impl<'target, V: VectorInfo> Debug for UnalignedDataMut<'target, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <UnalignedData<'target, V> as Debug>::fmt(&self.0, f)
    }
}
//
impl<V: VectorInfo> Eq for UnalignedDataMut<'_, V> {}
//
impl<V: VectorInfo> Hash for UnalignedDataMut<'_, V> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
//
impl<V: VectorInfo> Ord for UnalignedDataMut<'_, V> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}
//
impl<V: VectorInfo, Ptr: Borrow<NonNull<V::Array>>> PartialEq<Ptr> for UnalignedDataMut<'_, V> {
    fn eq(&self, other: &Ptr) -> bool {
        self.0 == *other.borrow()
    }
}
//
impl<V: VectorInfo, Ptr: Borrow<NonNull<V::Array>>> PartialOrd<Ptr> for UnalignedDataMut<'_, V> {
    fn partial_cmp(&self, other: &Ptr) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(other.borrow())
    }
}
//
impl<'target, V: VectorInfo> Pointer for UnalignedDataMut<'target, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <UnalignedData<'target, V> as Pointer>::fmt(&self.0, f)
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedData<V> for UnalignedDataMut<'target, V> {
    type Element = Self::ElementRef<'target>;
    type ElementRef<'result> = UnalignedMut<'result, V> where Self: 'result;
    type ElementCopy = V;
    type RefSlice<'result> = UnalignedDataMut<'result, V> where Self: 'result;
    type CopySlice<'result> = UnalignedData<'result, V> where Self: 'result;
}
//
unsafe impl<'target, V: VectorInfo> VectorizedDataImpl<V> for UnalignedDataMut<'target, V> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize, is_last: bool) -> V {
        unsafe { self.0.get_unchecked(idx, is_last) }
    }

    #[inline(always)]
    unsafe fn get_unchecked_ref(&mut self, idx: usize, _is_last: bool) -> UnalignedMut<V> {
        let target = self.0.get_ptr(idx).as_mut();
        let vector = V::from(*target);
        UnalignedMut::new(vector, target)
    }

    #[inline(always)]
    fn as_slice(&self) -> UnalignedData<V> {
        self.0.as_slice()
    }

    #[inline(always)]
    fn as_ref_slice(&mut self) -> UnalignedDataMut<V> {
        UnalignedDataMut(self.0.as_ref_slice(), PhantomData)
    }

    type Unaligned = Self;

    unsafe fn into_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = AlignedDataMut<'target, V>;

    unsafe fn into_aligned_unchecked(self) -> AlignedDataMut<'target, V> {
        AlignedDataMut(self.0.into_aligned_unchecked(), PhantomData)
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for UnalignedDataMut<'target, V> {
    #[inline(always)]
    fn empty() -> Self {
        Self(UnalignedData::empty(), PhantomData)
    }

    #[inline(always)]
    unsafe fn split_at_unchecked(self, mid: usize, len: usize) -> (Self, Self) {
        let (left, right) = self.0.split_at_unchecked(mid, len);
        let wrap = |inner| Self(inner, PhantomData);
        (wrap(left), wrap(right))
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
    pub(crate) fn new(
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
                        vectors.get_unchecked_ref(data.len() / V::LANES - 1, true)
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
impl<V: VectorInfo> Borrow<NonNull<V::Array>> for PaddedData<'_, V> {
    fn borrow(&self) -> &NonNull<V::Array> {
        self.vectors.borrow()
    }
}
//
impl<'target, V: VectorInfo> Debug for PaddedData<'target, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PaddedData")
            .field("vectors", &self.vectors)
            .field("last_vector", &self.last_vector)
            .finish()
    }
}
//
impl<'target, V: VectorInfo> Pointer for PaddedData<'target, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <UnalignedData<'target, V> as Pointer>::fmt(&self.vectors, f)
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedData<V> for PaddedData<'target, V> {
    type Element = V;
    type ElementRef<'result> = V where Self: 'result;
    type ElementCopy = V;
    type RefSlice<'result> = PaddedData<'result, V> where Self: 'result;
    type CopySlice<'result> = PaddedData<'result, V> where Self: 'result;
}
//
unsafe impl<'target, V: VectorInfo> VectorizedDataImpl<V> for PaddedData<'target, V> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize, is_last: bool) -> V {
        if is_last {
            unsafe { self.last_vector.assume_init() }
        } else {
            unsafe { self.vectors.get_unchecked(idx, false) }
        }
    }

    #[inline(always)]
    unsafe fn get_unchecked_ref(&mut self, idx: usize, is_last: bool) -> V {
        unsafe { self.get_unchecked(idx, is_last) }
    }

    #[inline(always)]
    fn as_slice(&self) -> PaddedData<V> {
        PaddedData {
            vectors: self.vectors.as_slice(),
            last_vector: self.last_vector,
        }
    }

    #[inline(always)]
    fn as_ref_slice(&mut self) -> PaddedData<V> {
        self.as_slice()
    }

    type Unaligned = UnalignedData<'target, V>;

    unsafe fn into_unaligned_unchecked(self) -> UnalignedData<'target, V> {
        self.vectors
    }

    type Aligned = AlignedData<'target, V>;

    unsafe fn into_aligned_unchecked(self) -> AlignedData<'target, V> {
        unsafe { self.vectors.into_aligned_unchecked() }
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for PaddedData<'target, V> {
    #[inline(always)]
    fn empty() -> Self {
        Self {
            vectors: UnalignedData::empty(),
            last_vector: MaybeUninit::uninit(),
        }
    }

    #[inline(always)]
    unsafe fn split_at_unchecked(mut self, mid: usize, len: usize) -> (Self, Self) {
        if mid == 0 {
            (Self::empty(), self)
        } else if mid == len {
            (self, Self::empty())
        } else {
            let left_last_vector = self.vectors.get_unchecked_ref(mid - 1, mid == len);
            let (left_vectors, right_vectors) = self.vectors.split_at_unchecked(mid, len);
            let wrap = |vectors, last_vector| Self {
                vectors,
                last_vector,
            };
            (
                wrap(left_vectors, MaybeUninit::new(left_last_vector)),
                wrap(right_vectors, self.last_vector),
            )
        }
    }
}

// NOTE: Can't implement support for [Scalar; SIZE] yet due to const
//       generics limitations (VectorizedDataImpl::Aligned should be
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
    pub(crate) fn new(
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
impl<V: VectorInfo> Borrow<NonNull<V::Array>> for PaddedDataMut<'_, V> {
    fn borrow(&self) -> &NonNull<V::Array> {
        self.inner.borrow()
    }
}
//
impl<'target, V: VectorInfo> Debug for PaddedDataMut<'target, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PaddedDataMut")
            .field("inner", &self.inner)
            .field("num_last_elems", &self.num_last_elems)
            .finish()
    }
}
//
impl<'target, V: VectorInfo> Pointer for PaddedDataMut<'target, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <PaddedData<'target, V> as Pointer>::fmt(&self.inner, f)
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedData<V> for PaddedDataMut<'target, V> {
    type Element = PaddedMut<'target, V>;
    type ElementRef<'result> = PaddedMut<'result, V> where Self: 'result;
    type ElementCopy = V;
    type RefSlice<'result> = PaddedDataMut<'result, V> where Self: 'result;
    type CopySlice<'result> = PaddedData<'result, V> where Self: 'result;
}
//
unsafe impl<'target, V: VectorInfo> VectorizedDataImpl<V> for PaddedDataMut<'target, V> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize, is_last: bool) -> V {
        unsafe { self.inner.get_unchecked(idx, is_last) }
    }

    #[inline(always)]
    unsafe fn get_unchecked_ref(&mut self, idx: usize, is_last: bool) -> PaddedMut<V> {
        PaddedMut::new(
            self.get_unchecked(idx, is_last),
            core::slice::from_raw_parts_mut(
                self.inner.get_ptr(idx).cast::<V::Scalar>().as_ptr(),
                self.num_elems(is_last),
            ),
        )
    }

    #[inline(always)]
    fn as_slice(&self) -> PaddedData<V> {
        self.inner.as_slice()
    }

    #[inline(always)]
    fn as_ref_slice(&mut self) -> PaddedDataMut<V> {
        PaddedDataMut {
            inner: self.inner.as_ref_slice(),
            num_last_elems: self.num_last_elems,
            lifetime: PhantomData,
        }
    }

    type Unaligned = UnalignedDataMut<'target, V>;

    unsafe fn into_unaligned_unchecked(self) -> UnalignedDataMut<'target, V> {
        unsafe { UnalignedDataMut(self.inner.into_unaligned_unchecked(), PhantomData) }
    }

    type Aligned = AlignedDataMut<'target, V>;

    unsafe fn into_aligned_unchecked(self) -> AlignedDataMut<'target, V> {
        unsafe { AlignedDataMut(self.inner.into_aligned_unchecked(), PhantomData) }
    }
}
//
unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for PaddedDataMut<'target, V> {
    #[inline(always)]
    fn empty() -> Self {
        Self {
            inner: PaddedData::empty(),
            num_last_elems: 0,
            lifetime: PhantomData,
        }
    }

    #[inline(always)]
    unsafe fn split_at_unchecked(self, mid: usize, len: usize) -> (Self, Self) {
        if mid == 0 {
            (Self::empty(), self)
        } else if mid == len {
            (self, Self::empty())
        } else {
            let left_num_last_elems = self.num_elems(mid == len);
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
        }
    }
}

/// Tuples of pointers yield tuples of deref results
macro_rules! impl_vectorized_for_tuple {
    (
        $($t:ident),*
    ) => {
        unsafe impl<
            'target,
            V: VectorInfo
            $(, $t: VectorizedData<V> + 'target)*
        > VectorizedData<V> for ($($t,)*) {
            type Element = ($($t::Element,)*);
            type ElementRef<'result> = ($($t::ElementRef<'result>,)*) where Self: 'result;
            type ElementCopy = ($($t::ElementCopy,)*);
            type RefSlice<'result> = ($($t::RefSlice<'result>,)*) where Self: 'result;
            type CopySlice<'result> = ($($t::CopySlice<'result>,)*) where Self: 'result;
        }

        #[allow(non_snake_case)]
        unsafe impl<
            'target,
            V: VectorInfo
            $(, $t: VectorizedDataImpl<V> + 'target)*
        > VectorizedDataImpl<V> for ($($t,)*) {
            #[inline(always)]
            unsafe fn get_unchecked(
                &self,
                idx: usize,
                is_last: bool
            ) -> Self::ElementCopy {
                let ($($t,)*) = self;
                unsafe { ($($t.get_unchecked(idx, is_last),)*) }
            }

            #[inline(always)]
            unsafe fn get_unchecked_ref(
                &mut self,
                idx: usize,
                is_last: bool
            ) -> Self::ElementRef<'_> {
                let ($($t,)*) = self;
                unsafe { ($($t.get_unchecked_ref(idx, is_last),)*) }
            }

            #[inline(always)]
            fn as_slice(&self) -> Self::CopySlice<'_> {
                let ($($t,)*) = self;
                ($($t.as_slice(),)*)
            }

            #[inline(always)]
            fn as_ref_slice(&mut self) -> Self::RefSlice<'_> {
                let ($($t,)*) = self;
                ($($t.as_ref_slice(),)*)
            }

            type Unaligned = ($($t::Unaligned,)*);

            unsafe fn into_unaligned_unchecked(self) -> Self::Unaligned {
                let ($($t,)*) = self;
                unsafe { ($($t.into_unaligned_unchecked(),)*) }
            }

            type Aligned = ($($t::Aligned,)*);

            unsafe fn into_aligned_unchecked(self) -> Self::Aligned {
                let ($($t,)*) = self;
                unsafe { ($($t.into_aligned_unchecked(),)*) }
            }
        }

        #[allow(non_snake_case)]
        unsafe impl<
            'target,
            V: VectorInfo
            $(, $t: VectorizedSliceImpl<V> + 'target)*
        > VectorizedSliceImpl<V> for ($($t,)*) {
            #[inline(always)]
            fn empty() -> Self {
                ($($t::empty(),)*)
            }

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
impl_vectorized_for_tuple!(A);
impl_vectorized_for_tuple!(A, B);
impl_vectorized_for_tuple!(A, B, C);
impl_vectorized_for_tuple!(A, B, C, D);
impl_vectorized_for_tuple!(A, B, C, D, E);
impl_vectorized_for_tuple!(A, B, C, D, E, F);
impl_vectorized_for_tuple!(A, B, C, D, E, F, G);
impl_vectorized_for_tuple!(A, B, C, D, E, F, G, H);

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::vectorize::tests::{any_v, VArray, VScalar, V};
    use proptest::{
        array::{uniform2, uniform3},
        prelude::*,
    };
    use std::{collections::hash_map::DefaultHasher, hash::Hasher};

    // === COMMON TEST HARNESS ===

    /// Maximum length (in SIMD vector elements) that we need to test in order
    /// to be sure to observe all interesting effects
    ///
    /// Should be >= 3 to test that loops stay correct beyond the first
    /// repetition, add 1 due to truncation of trailing scalars in UnalignedData
    const MAX_SIMD_LEN: usize = 4;

    /// Generate the building blocks to initialize AlignedData(Mut)?
    pub(crate) fn aligned_init_input(allow_empty: bool) -> impl Strategy<Value = Vec<V>> {
        prop::collection::vec(any_v(), ((!allow_empty) as usize)..=MAX_SIMD_LEN)
    }

    /// Arbitrary array of aligned data
    pub(crate) type AlignedArray = [V; MAX_SIMD_LEN];
    pub(crate) fn any_aligned_array() -> impl Strategy<Value = AlignedArray> {
        prop::array::uniform4(any_v())
    }

    /// Generate the building blocks to initialize UnalignedData(Mut)?
    pub(crate) fn unaligned_init_input(min_scalars: usize) -> impl Strategy<Value = Vec<VScalar>> {
        prop::collection::vec(any::<VScalar>(), min_scalars..=MAX_SIMD_LEN * V::LANES)
    }

    /// Generate the building blocks to initialize PaddedData(Mut)?
    pub(crate) fn padded_init_input(
        allow_empty: bool,
    ) -> impl Strategy<Value = (Vec<VScalar>, Option<VScalar>)> {
        (
            unaligned_init_input(!allow_empty as usize),
            if allow_empty {
                any::<Option<VScalar>>().boxed()
            } else {
                any::<VScalar>().prop_map(Some).boxed()
            },
        )
    }

    // V-based typedefs to reduce inference issues
    type AlignedV<'a> = AlignedData<'a, V>;
    type AlignedVMut<'a> = AlignedDataMut<'a, V>;
    type UnalignedV<'a> = UnalignedData<'a, V>;
    type UnalignedVMut<'a> = UnalignedDataMut<'a, V>;
    type PaddedV<'a> = PaddedData<'a, V>;
    type PaddedVMut<'a> = PaddedDataMut<'a, V>;

    /// Tuple of most supported entities, for exhaustive tests
    pub(crate) type TupleData<'a> = (
        AlignedV<'a>,
        AlignedVMut<'a>,
        UnalignedV<'a>,
        UnalignedVMut<'a>,
        PaddedV<'a>,
        PaddedVMut<'a>,
    );

    /// Building blocks needed to initialize TupleData
    ///
    /// Satisfies the extra invariant that all inner containers must have the
    /// same SIMD length, denoted `simd_len` in the following.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct TupleInitInput {
        // Length == `simd_len`
        aligned: Vec<V>,
        aligned_mut: Vec<V>,
        // Length == `simd_len * V::LANES`
        unaligned: Vec<VScalar>,
        unaligned_mut: Vec<VScalar>,
        // `(simd_len - 1) * V::LANES` < Length < `simd_len * V::LANES` or Length == 0
        padded: Vec<VScalar>,
        padded_mut: Vec<VScalar>,
        padding: VScalar,
    }
    //
    impl TupleInitInput {
        /// Construct TupleData from this
        pub fn as_tuple_data(&mut self) -> TupleData {
            (
                AlignedV::from(self.aligned.as_slice()),
                AlignedVMut::from(self.aligned_mut.as_mut_slice()),
                UnalignedV::from(self.unaligned.as_slice()),
                UnalignedVMut::from(self.unaligned_mut.as_mut_slice()),
                PaddedV::new(self.padded.as_slice(), Some(self.padding))
                    .unwrap()
                    .0,
                PaddedVMut::new(self.padded_mut.as_mut_slice(), Some(self.padding)).unwrap(),
            )
        }
    }

    /// Generate the building blocks to initialize TupleData
    pub(crate) fn tuple_init_input(allow_empty: bool) -> impl Strategy<Value = TupleInitInput> {
        (((!allow_empty) as usize)..=MAX_SIMD_LEN)
            .prop_flat_map(|simd_len| {
                let aligned = || prop::collection::vec(any_v(), simd_len);
                let scalar_len = simd_len * V::LANES;
                let unaligned = || prop::collection::vec(any::<VScalar>(), scalar_len);
                let padded = || {
                    prop::collection::vec(
                        any::<VScalar>(),
                        scalar_len.saturating_sub(V::LANES - 1)..=scalar_len.saturating_sub(1),
                    )
                };
                (
                    aligned(),
                    aligned(),
                    unaligned(),
                    unaligned(),
                    padded(),
                    padded(),
                    any::<VScalar>(),
                )
            })
            .prop_map(
                |(aligned, aligned_mut, unaligned, unaligned_mut, padded, padded_mut, padding)| {
                    TupleInitInput {
                        aligned,
                        aligned_mut,
                        unaligned,
                        unaligned_mut,
                        padded,
                        padded_mut,
                        padding,
                    }
                },
            )
    }

    /// Query some properties of SIMD data (as built above)
    pub(crate) trait SimdData: Clone + Debug {
        // Base pointer (can be used to check if two things target the same allocation
        type BasePtr: Copy + Debug + Eq + Hash;
        fn base_ptr(&self) -> Self::BasePtr;

        // Number of SIMD vectors that can be read using get_unchecked
        fn simd_len(&self) -> usize;

        // Truth that one element is the last one
        fn is_last(&self, idx: usize) -> bool {
            idx == self.simd_len() - 1
        }

        // Read the N-th SIMD vector (or tuple of vectors)
        type Element;
        fn simd_element(&self, idx: usize) -> Self::Element;
    }
    //
    impl SimdData for Vec<V> {
        type BasePtr = NonNull<V>;
        fn base_ptr(&self) -> NonNull<V> {
            NonNull::from(self.as_slice()).cast::<V>()
        }

        fn simd_len(&self) -> usize {
            self.len()
        }

        type Element = V;
        fn simd_element(&self, idx: usize) -> V {
            self[idx]
        }
    }
    //
    impl<const N: usize> SimdData for [V; N] {
        type BasePtr = NonNull<V>;
        fn base_ptr(&self) -> NonNull<V> {
            NonNull::from(self.as_slice()).cast::<V>()
        }

        fn simd_len(&self) -> usize {
            self.len()
        }

        type Element = V;
        fn simd_element(&self, idx: usize) -> V {
            self[idx]
        }
    }
    //
    impl SimdData for Vec<VScalar> {
        type BasePtr = NonNull<VArray>;
        fn base_ptr(&self) -> NonNull<VArray> {
            NonNull::from(self.as_slice()).cast::<VArray>()
        }

        fn simd_len(&self) -> usize {
            self.len() / V::LANES
        }

        type Element = V;
        fn simd_element(&self, idx: usize) -> V {
            let base = idx * V::LANES;
            V::new(&self[base..base + V::LANES])
        }
    }
    //
    impl SimdData for (&Vec<VScalar>, Option<VScalar>) {
        type BasePtr = NonNull<VArray>;

        fn base_ptr(&self) -> NonNull<VArray> {
            self.0.base_ptr()
        }

        fn simd_len(&self) -> usize {
            let mut result = self.0.simd_len();
            if self.0.len() % V::LANES != 0 {
                assert_ne!(self.1, None);
                result += 1;
            }
            result
        }

        type Element = V;
        fn simd_element(&self, idx: usize) -> V {
            let base = idx * V::LANES;
            V::from_fn(|offset| {
                let idx = base + offset;
                if idx < self.0.len() {
                    self.0[idx]
                } else {
                    self.1.unwrap()
                }
            })
        }
    }
    //
    impl SimdData for (Vec<VScalar>, Option<VScalar>) {
        type BasePtr = NonNull<VArray>;
        fn base_ptr(&self) -> NonNull<VArray> {
            (&self.0, self.1).base_ptr()
        }

        fn simd_len(&self) -> usize {
            (&self.0, self.1).simd_len()
        }

        type Element = V;
        fn simd_element(&self, idx: usize) -> V {
            (&self.0, self.1).simd_element(idx)
        }
    }
    //
    impl SimdData for TupleInitInput {
        type BasePtr = (
            NonNull<V>,
            NonNull<V>,
            NonNull<VArray>,
            NonNull<VArray>,
            NonNull<VArray>,
            NonNull<VArray>,
        );

        fn base_ptr(&self) -> Self::BasePtr {
            (
                self.aligned.base_ptr(),
                self.aligned_mut.base_ptr(),
                self.unaligned.base_ptr(),
                self.unaligned_mut.base_ptr(),
                self.padded.base_ptr(),
                self.padded_mut.base_ptr(),
            )
        }

        fn simd_len(&self) -> usize {
            self.aligned.len()
        }

        type Element = TupleElem;
        fn simd_element(&self, idx: usize) -> Self::Element {
            (
                self.aligned.simd_element(idx),
                self.aligned_mut.simd_element(idx),
                self.unaligned.simd_element(idx),
                self.unaligned_mut.simd_element(idx),
                (&self.padded, Some(self.padding)).simd_element(idx),
                (&self.padded_mut, Some(self.padding)).simd_element(idx),
            )
        }
    }

    /// Complement an existing SIMD dataset generation Strategy with an index
    /// that's in range (use with prop_flat_map).
    ///
    /// Use with variants of the input generation strategies that _don't_ allow
    /// empty or invalid datasets to be generated.
    pub(crate) fn with_data_index<Data: SimdData>(
        data: Data,
    ) -> impl Strategy<Value = (Data, usize)> {
        let simd_len = data.simd_len();
        assert_ne!(
            simd_len, 0,
            "Don't set allow_empty to true when you want to index"
        );
        (Just(data), 0..simd_len)
    }

    /// Complement an existing SIMD dataset generation Strategy with an index
    /// that's suitable for splitting (in range + 1-past-the-end)
    pub(crate) fn with_split_index<Data: SimdData>(
        data: Data,
    ) -> impl Strategy<Value = (Data, usize)> {
        let simd_len = data.simd_len();
        (Just(data), 0..=simd_len)
    }

    /// Output of get_unchecked on TupleData
    pub(crate) type TupleElem = (V, V, V, V, V, V);
    //
    /// Output of get_unchecked_ref on TupleData
    pub(crate) type TupleRef<'a> = (V, &'a mut V, V, UnalignedMut<'a, V>, V, PaddedMut<'a, V>);
    //
    /// Read all data from TupleRef and discard it
    pub(crate) fn read_tuple(tuple: TupleRef) -> TupleElem {
        read_from_tuple(&tuple)
    }
    //
    /// Read all data from a borrowed TupleRef
    pub(crate) fn read_from_tuple(
        (aligned, aligned_mut, unaligned, unaligned_mut, padded, padded_mut): &TupleRef,
    ) -> TupleElem {
        (
            *aligned,
            **aligned_mut,
            *unaligned,
            **unaligned_mut,
            *padded,
            **padded_mut,
        )
    }
    //
    /// Data needed to fully change all mutable state behind TupleRef
    pub(crate) type TupleWrite = [V; 3];
    //
    /// Random generator for TupleWrite data
    pub(crate) fn any_tuple_write() -> impl Strategy<Value = TupleWrite> {
        uniform3(any_v())
    }
    //
    /// Commit TupleWrite data into a TupleRef
    pub(crate) fn write_tuple(tuple: &mut TupleRef, data: TupleWrite) {
        *tuple.1 = data[0];
        *tuple.3 = data[1];
        *tuple.5 = data[2];
    }
    //
    /// Check that a write went well
    pub(crate) fn check_tuple_write(
        data: &TupleInitInput,
        idx: usize,
        old: TupleElem,
        write: TupleWrite,
    ) {
        let new = data.simd_element(idx);
        assert_eq!(new.0, old.0);
        assert_eq!(new.1, write[0]);
        assert_eq!(new.2, old.2);
        assert_eq!(new.3, write[1]);
        assert_eq!(new.4, old.4);
        if idx < data.simd_len() - 1 {
            assert_eq!(new.5, write[2]);
        } else {
            // Padded pattern at end is complicated to predict due to padding,
            // and we're already testing setting of padded data elsewhere, so
            // here we're only testing that it's different if it should be.
            assert_eq!(new.5 == old.5, old.5 == write[2]);
        }
    }

    // === TESTS FOR THIS MODULE ===

    /// Hash a value
    fn hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    // 'static version of most types for empty slice testing
    type AlignedStatic = AlignedV<'static>;
    type AlignedStaticMut = AlignedVMut<'static>;
    type UnalignedStatic = UnalignedV<'static>;
    type UnalignedStaticMut = UnalignedVMut<'static>;
    type PaddedStatic = PaddedV<'static>;
    type PaddedStaticMut = PaddedVMut<'static>;

    #[test]
    fn empty_aligned() {
        assert_eq!(AlignedStatic::empty(), AlignedStatic::empty());
        assert_eq!(AlignedStatic::empty(), AlignedStaticMut::empty());
        assert_eq!(AlignedStaticMut::empty(), AlignedStaticMut::empty());
    }

    #[test]
    fn empty_unaligned() {
        assert_eq!(UnalignedStatic::empty(), UnalignedStatic::empty());
        assert_eq!(UnalignedStatic::empty(), UnalignedStaticMut::empty());
        assert_eq!(UnalignedStaticMut::empty(), UnalignedStaticMut::empty());
    }

    #[test]
    fn empty_padded() {
        assert_eq!(PaddedStatic::empty().vectors, PaddedStatic::empty().vectors);
        assert_eq!(
            PaddedStatic::empty().vectors,
            PaddedStaticMut::empty().inner.vectors
        );
        assert_eq!(
            PaddedStaticMut::empty().inner.vectors,
            PaddedStaticMut::empty().inner.vectors,
        );
        assert_eq!(
            PaddedStaticMut::empty().num_last_elems,
            PaddedStaticMut::empty().num_last_elems,
        );
    }

    #[test]
    fn empty_tuple() {
        type ComparableTupleStatic = (AlignedStatic, UnalignedStatic);
        type ComparableTupleStaticMut = (AlignedStaticMut, UnalignedStaticMut);
        assert_eq!(
            ComparableTupleStatic::empty(),
            ComparableTupleStatic::empty()
        );
        assert_eq!(
            ComparableTupleStaticMut::empty(),
            ComparableTupleStaticMut::empty()
        );
    }

    /// Test properties of a freshly initialized data pointer
    fn test_init<
        Target,
        DataFromRaw: Borrow<NonNull<Target>> + Debug,
        StaticData: Borrow<NonNull<Target>> + Debug,
        StaticDataMut: Borrow<NonNull<Target>> + Debug,
        Data: Borrow<NonNull<Target>>
            + Debug
            + Hash
            + PartialEq
            + PartialEq<DataFromRaw>
            + PartialEq<NonNull<Target>>
            + PartialEq<StaticData>
            + PartialEq<StaticDataMut>
            + Pointer
            + VectorizedDataImpl<V>,
    >(
        slice_ptr: NonNull<[Target]>,
        data: Data,
        data_from_raw: DataFromRaw,
        empty: StaticData,
        empty_mut: StaticDataMut,
    ) {
        let base_ptr = slice_ptr.cast::<Target>();
        let base_ptr_debug = format!("{base_ptr:?}");
        let base_ptr_hash = hash(&base_ptr);
        let base_ptr_pointer = format!("{base_ptr:p}");

        assert!(data.eq(&data));
        assert_eq!(data, base_ptr);
        assert_eq!(data, data_from_raw);

        if slice_ptr.len() > 0 {
            assert_ne!(data, empty);
            assert_ne!(data, empty_mut);
        }

        assert_eq!(format!("{data:?}"), base_ptr_debug);
        assert_eq!(hash(&data), base_ptr_hash);
        assert_eq!(format!("{data:p}"), base_ptr_pointer);
    }

    /// Extract the aligned subset of an unaligned scalar slice
    #[allow(clippy::redundant_slicing)]
    fn extract_aligned(data: &mut [VScalar]) -> (NonNull<V>, &mut [VScalar]) {
        let (_, aligned, _) = unsafe { data.align_to_mut::<V>() };
        let base_ptr = NonNull::from(&aligned[..]).cast::<V>();
        let aligned_scalars = unsafe {
            std::slice::from_raw_parts_mut(
                aligned.as_mut_ptr().cast::<VScalar>(),
                aligned.len() * V::LANES,
            )
        };
        (base_ptr, aligned_scalars)
    }

    /// Extract the unaligned vectors subset of a scalar slice
    #[allow(clippy::redundant_slicing)]
    fn extract_unaligned(data: &mut [VScalar]) -> (NonNull<VArray>, &mut [VScalar]) {
        let len_vecs = (data.len() / V::LANES) * V::LANES;
        let unaligned = &mut data[..len_vecs];
        let base_ptr = NonNull::from(&unaligned[..]).cast::<VArray>();
        (base_ptr, unaligned)
    }

    proptest! {
        /// Test freshly initialized AlignedData(Mut)?
        #[test]
        fn init_aligned(mut data in aligned_init_input(true)) {
            let initial_data = data.clone();
            let slice_ptr = NonNull::from(data.as_slice());
            let aligned_raw = unsafe { AlignedData::from_data_ptr(slice_ptr) };

            {
                let mut aligned = AlignedData::from(data.as_slice());
                test_init(
                    slice_ptr,
                    aligned,
                    aligned_raw,
                    AlignedStatic::empty(),
                    AlignedStaticMut::empty(),
                );
                assert_eq!(aligned.as_slice(), aligned_raw);
                assert_eq!(aligned.as_ref_slice(), aligned_raw);
                unsafe {
                    assert_eq!(aligned.into_aligned_unchecked(), aligned_raw);
                    assert_eq!(aligned.into_unaligned_unchecked(), aligned_raw);
                }
            }
            assert_eq!(data, initial_data);

            {
                test_init(
                    slice_ptr,
                    AlignedDataMut::from(data.as_mut_slice()),
                    aligned_raw,
                    AlignedStatic::empty(),
                    AlignedStaticMut::empty(),
                );
                let mut aligned_mut = AlignedDataMut::from(data.as_mut_slice());
                assert_eq!(aligned_mut.as_slice(), aligned_raw);
                assert_eq!(aligned_mut.as_ref_slice(), aligned_raw);
                unsafe {
                    assert_eq!(aligned_mut.into_aligned_unchecked(), aligned_raw);
                    assert_eq!(
                        AlignedDataMut::from(data.as_mut_slice()).into_unaligned_unchecked(),
                        aligned_raw
                    );
                }
            }
            assert_eq!(data, initial_data);
        }

        /// Test treating arrays as shared slices
        #[test]
        fn init_array(mut data in any_aligned_array()) {
            let data_ptr = |data: &AlignedArray| NonNull::from(data).cast::<V>();
            let slice = <AlignedArray as VectorizedDataImpl<V>>::as_ref_slice;

            let mut aligned = unsafe { data.into_aligned_unchecked() };
            let mut unaligned = unsafe { data.into_unaligned_unchecked() };

            let mut ptr = data_ptr(&data);
            assert_eq!(slice(&mut data), ptr);

            ptr = data_ptr(&aligned);
            assert_eq!(slice(&mut aligned), ptr);

            ptr = data_ptr(&unaligned);
            assert_eq!(slice(&mut unaligned), ptr);
        }

        /// Test freshly initialized UnalignedData(Mut)?
        #[test]
        fn init_unaligned(mut data in unaligned_init_input(0)) {
            let initial_data = data.clone();
            let slice_ptr = NonNull::from(data.as_slice());
            let unaligned_raw = unsafe { UnalignedV::from_data_ptr(slice_ptr) };
            let array_slice = unsafe { std::slice::from_raw_parts(
                slice_ptr.cast::<VArray>().as_ptr(),
                data.len() / V::LANES
            ) };
            let array_slice_ptr = NonNull::from(array_slice);

            {
                let mut unaligned = UnalignedV::from(data.as_slice());
                test_init(
                    array_slice_ptr,
                    unaligned,
                    unaligned_raw,
                    UnalignedStatic::empty(),
                    UnalignedStaticMut::empty(),
                );
                assert_eq!(unaligned.as_slice(), unaligned_raw);
                assert_eq!(unaligned.as_ref_slice(), unaligned_raw);
                #[allow(clippy::redundant_slicing)]
                unsafe {
                    assert_eq!(unaligned.into_unaligned_unchecked(), unaligned_raw);
                    let (aligned_base, aligned) = extract_aligned(data.as_mut_slice());
                    assert_eq!(UnalignedV::from(&aligned[..]).into_aligned_unchecked(), aligned_base);
                }
            }
            assert_eq!(data, initial_data);

            {
                let mut unaligned_mut = UnalignedVMut::from(data.as_mut_slice());
                test_init(
                    array_slice_ptr,
                    unaligned_mut,
                    unaligned_raw,
                    UnalignedStatic::empty(),
                    UnalignedStaticMut::empty(),
                );
                unaligned_mut = UnalignedVMut::from(data.as_mut_slice());
                assert_eq!(unaligned_mut.as_slice(), unaligned_raw);
                assert_eq!(unaligned_mut.as_ref_slice(), unaligned_raw);
                unsafe {
                    assert_eq!(UnalignedVMut::from(data.as_mut_slice()).into_unaligned_unchecked(), unaligned_raw);
                    let (aligned_base, aligned) = extract_aligned(data.as_mut_slice());
                    assert_eq!(UnalignedVMut::from(aligned).into_aligned_unchecked(), aligned_base);
                }
            }
            assert_eq!(data, initial_data);
        }

        /// Test freshly initialized PaddedData(Mut)?
        #[test]
        #[allow(clippy::redundant_slicing)]
        fn init_padded((mut data, padding) in padded_init_input(true)) {
            let initial_data = data.clone();

            // Padding is required if data length is not divisible by V::LANES.
            // In that case, constructor should error out if it's not present.
            if (data.len() % V::LANES != 0) && padding.is_none() {
                assert!(matches!(PaddedV::new(data.as_slice(), padding),
                                 Err(VectorizeError::NeedsPadding)));
                assert!(matches!(PaddedVMut::new(data.as_mut_slice(), padding),
                                 Err(VectorizeError::NeedsPadding)));
                assert_eq!(data, initial_data);
                return Ok(());
            }

            // Common setup
            let slice_ptr = NonNull::from(data.as_slice());
            let unaligned_raw = unsafe { UnalignedV::from_data_ptr(slice_ptr) };
            let array_slice = unsafe { std::slice::from_raw_parts(
                slice_ptr.cast::<VArray>().as_ptr(),
                data.len() / V::LANES
            ) };
            let array_slice_ptr = NonNull::from(array_slice);

            let (last_elems, last_vector) = {
                // Check output of constructing PaddedData
                let (mut padded, last_elems) = PaddedV::new(data.as_slice(), padding).unwrap();
                test_init(
                    array_slice_ptr,
                    padded.vectors,
                    unaligned_raw,
                    PaddedStatic::empty(),
                    PaddedStaticMut::empty(),
                );
                let expected_last_elems = match data.len() {
                    0 => 0,
                    len if len % V::LANES == 0 => V::LANES,
                    other_len => other_len % V::LANES,
                };
                assert_eq!(last_elems, expected_last_elems);
                let last_vector = (last_elems > 0).then(|| {
                    let last_vector = unsafe { padded.last_vector.assume_init() };
                    let offset = (array_slice.len() - (last_elems == V::LANES) as usize) * V::LANES;
                    let expected_last_vector = V::from_fn(|idx| if idx < last_elems {
                        data[offset + idx]
                    } else {
                        padding.unwrap()
                    });
                    assert_eq!(last_vector, expected_last_vector);
                    last_vector
                });

                // Check output of reinterpreting as a slice
                {
                    let padded_slice = padded.as_slice();
                    assert_eq!(padded_slice.vectors, unaligned_raw);
                    if let Some(last_vector) = last_vector {
                        let last_vector_slice = unsafe { padded_slice.last_vector.assume_init() };
                        assert_eq!(last_vector, last_vector_slice);
                    }
                }
                {
                    let padded_slice_mut = padded.as_ref_slice();
                    assert_eq!(padded_slice_mut.vectors, unaligned_raw);
                    if let Some(last_vector) = last_vector {
                        let last_vector_slice = unsafe { padded_slice_mut.last_vector.assume_init() };
                        assert_eq!(last_vector, last_vector_slice);
                    }
                }
                (last_elems, last_vector)
            };
            assert_eq!(data, initial_data);

            // Check output of reinterpreting as unaligned data
            {
                let (unaligned_base, unaligned_data) = extract_unaligned(data.as_mut_slice());
                let (padded, last_elems) = PaddedV::new(&unaligned_data[..], None).unwrap();
                assert_eq!(unsafe { padded.into_unaligned_unchecked() }, unaligned_base);
                assert_eq!(last_elems, V::LANES * (!unaligned_data.is_empty()) as usize);
            }
            assert_eq!(data, initial_data);

            // Check output of reinterpreting as aligned data
            {
                let (aligned_base, aligned_data) = extract_aligned(data.as_mut_slice());
                let (padded, last_elems) = PaddedV::new(&aligned_data[..], None).unwrap();
                assert_eq!(unsafe { padded.into_aligned_unchecked() }, aligned_base);
                assert_eq!(last_elems, V::LANES * (!aligned_data.is_empty()) as usize);
            }
            assert_eq!(data, initial_data);

            {
                // Check outcome of constructing PaddedDataMut
                let mut padded_mut = PaddedVMut::new(data.as_mut_slice(), padding).unwrap();
                if let Some(last_vector) = last_vector {
                    let last_vector_mut = unsafe { padded_mut.inner.last_vector.assume_init() };
                    assert_eq!(last_vector, last_vector_mut);
                }
                assert_eq!(padded_mut.num_last_elems, last_elems);
                test_init(
                    array_slice_ptr,
                    padded_mut.inner.vectors,
                    unaligned_raw,
                    PaddedStatic::empty(),
                    PaddedStaticMut::empty(),
                );

                // Check outcome of reinterpreting as a slice
                {
                    padded_mut = PaddedVMut::new(data.as_mut_slice(), padding).unwrap();
                    let padded_slice = padded_mut.as_slice();
                    assert_eq!(padded_slice.vectors, unaligned_raw);
                    if let Some(last_vector) = last_vector {
                        let last_vector_slice = unsafe { padded_slice.last_vector.assume_init() };
                        assert_eq!(last_vector, last_vector_slice);
                    }
                }
                {
                    padded_mut = PaddedVMut::new(data.as_mut_slice(), padding).unwrap();
                    let padded_slice_mut = padded_mut.as_ref_slice();
                    assert_eq!(padded_slice_mut.inner.vectors, unaligned_raw);
                    if let Some(last_vector) = last_vector {
                        let last_vector_slice = unsafe { padded_slice_mut.inner.last_vector.assume_init() };
                        assert_eq!(last_vector, last_vector_slice);
                    }
                    assert_eq!(padded_slice_mut.num_last_elems, last_elems);
                }
            }
            assert_eq!(data, initial_data);

            // Check output of reinterpreting as unaligned data
            {
                let (unaligned_base, unaligned_data) = extract_unaligned(data.as_mut_slice());
                let padded_mut = PaddedVMut::new(unaligned_data, None).unwrap();
                assert_eq!(unsafe { padded_mut.into_unaligned_unchecked() }, unaligned_base);
            }
            assert_eq!(data, initial_data);

            // Check output of reinterpreting as aligned data
            {
                let (aligned_base, aligned_data) = extract_aligned(data.as_mut_slice());
                let padded_mut = PaddedVMut::new(aligned_data, None).unwrap();
                assert_eq!(unsafe { padded_mut.into_aligned_unchecked() }, aligned_base);
            }
            assert_eq!(data, initial_data);
        }

        /// Test freshly initialized TupleData
        #[test]
        fn init_tuple(mut data in tuple_init_input(true)) {
            let initial_data = data.clone();
            {
                let base = data.base_ptr();
                let mut tuple = data.as_tuple_data();

                {
                    let slice = tuple.as_slice();
                    assert_eq!(slice.0, base.0);
                    assert_eq!(slice.1, base.1);
                    assert_eq!(slice.2, base.2);
                    assert_eq!(slice.3, base.3);
                    assert_eq!(slice.4.vectors, base.4);
                    assert_eq!(slice.5.vectors, base.5);
                }

                {
                    let slice = tuple.as_ref_slice();
                    assert_eq!(slice.0, base.0);
                    assert_eq!(slice.1, base.1);
                    assert_eq!(slice.2, base.2);
                    assert_eq!(slice.3, base.3);
                    assert_eq!(slice.4.vectors, base.4);
                    assert_eq!(slice.5.inner.vectors, base.5);
                }
            }
            assert_eq!(data, initial_data);
        }

        /// Test the get_unchecked and get_ptr of AlignedData(Mut)?
        #[test]
        fn get_aligned((mut data, idx) in aligned_init_input(false).prop_flat_map(with_data_index)) {
            let initial_data = data.clone();
            {
                let elem = data.simd_element(idx);
                let base_ptr = data.base_ptr();
                let is_last = data.is_last(idx);

                {
                    let mut aligned = AlignedV::from(data.as_slice());
                    assert_eq!(unsafe { aligned.get_ptr(idx) }.as_ptr(),
                               base_ptr.as_ptr().wrapping_add(idx));
                    assert_eq!(unsafe { aligned.get_unchecked(idx, is_last) }, elem);
                    assert_eq!(unsafe { aligned.get_unchecked_ref(idx, is_last) }, elem);
                }

                {
                    let mut aligned_mut = AlignedVMut::from(data.as_mut_slice());
                    assert_eq!(unsafe { aligned_mut.get_unchecked(idx, is_last) }, elem);
                    assert_eq!(*unsafe { aligned_mut.get_unchecked_ref(idx, is_last) }, elem);
                }
            }
            assert_eq!(data, initial_data);
        }

        /// Test treating arrays as shared slices
        #[test]
        fn get_array((mut data, idx) in any_aligned_array().prop_flat_map(with_data_index)) {
            let initial_data = data.clone();
            {
                let elem = data.simd_element(idx);
                let is_last = data.is_last(idx);
                assert_eq!(unsafe { VectorizedDataImpl::get_unchecked(&data, idx, is_last) }, elem);
                assert_eq!(unsafe { VectorizedDataImpl::get_unchecked_ref(&mut data, idx, is_last) }, elem);
            }
            assert_eq!(data, initial_data);
        }

        /// Test the get_unchecked and get_ptr of UnalignedData(Mut)?
        #[test]
        fn get_unaligned((mut data, idx) in unaligned_init_input(V::LANES).prop_flat_map(with_data_index)) {
            let initial_data = data.clone();
            {
                let elem = data.simd_element(idx);
                let base_ptr = data.base_ptr();
                let is_last = data.is_last(idx);

                {
                    let mut unaligned = UnalignedV::from(data.as_slice());
                    assert_eq!(unsafe { unaligned.get_ptr(idx) }.as_ptr(),
                               base_ptr.as_ptr().wrapping_add(idx));
                    assert_eq!(unsafe { unaligned.get_unchecked_ref(idx, is_last) }, elem);
                }

                {
                    let mut unaligned_mut = UnalignedVMut::from(data.as_mut_slice());
                    assert_eq!(unsafe { unaligned_mut.get_unchecked(idx, is_last) }, elem);
                    assert_eq!(*unsafe { unaligned_mut.get_unchecked_ref(idx, is_last) }, elem);
                }
            }
            assert_eq!(data, initial_data);
        }

        /// Test the get_unchecked and get_ptr of PaddedData(Mut)?
        #[test]
        fn get_padded((mut padded_data, idx) in padded_init_input(false).prop_flat_map(with_data_index)) {
            let initial_data = padded_data.0.clone();
            {
                let elem = padded_data.simd_element(idx);
                let base_ptr = padded_data.base_ptr();
                let is_last = padded_data.is_last(idx);

                {
                    let (data, padding) = &padded_data;
                    let mut padded = PaddedV::new(data.as_slice(), *padding).unwrap().0;
                    assert_eq!(unsafe { padded.get_ptr(idx) }.as_ptr(),
                               base_ptr.as_ptr().wrapping_add(idx));
                    assert_eq!(unsafe { padded.get_unchecked_ref(idx, is_last) }, elem);
                }

                {
                    let (data, padding) = &mut padded_data;
                    let mut padded_mut = PaddedVMut::new(data.as_mut_slice(), *padding).unwrap();
                    assert_eq!(unsafe { padded_mut.get_unchecked(idx, is_last) }, elem);
                    assert_eq!(*unsafe { padded_mut.get_unchecked_ref(idx, is_last) }, elem);
                }
            }
            assert_eq!(padded_data.0, initial_data);
        }

        /// Test reading through the get_unchecked(_ref) of TupleData
        #[test]
        fn get_tuple((mut data, idx) in tuple_init_input(false).prop_flat_map(with_data_index)) {
            let initial_data = data.clone();
            let elem = data.simd_element(idx);
            let is_last = data.is_last(idx);
            {
                let mut tuple = data.as_tuple_data();
                assert_eq!(unsafe { tuple.get_unchecked(idx, is_last) }, elem);
                let output = unsafe { tuple.get_unchecked_ref(idx, is_last) };
                assert_eq!(read_from_tuple(&output), elem);
            }
            assert_eq!(data, initial_data);
        }

        /// Test writing through the get_unchecked_mut of AlignedDataMut
        #[test]
        fn set_aligned(
            ((mut data, idx), new_elem) in (aligned_init_input(false)
                                                .prop_flat_map(with_data_index),
                                            any_v())
        ) {
            let initial_data = data.clone();
            let len = data.simd_len();
            let is_last = data.is_last(idx);

            {
                let mut aligned_mut = AlignedVMut::from(data.as_mut_slice());
                let elem_ref = unsafe { aligned_mut.get_unchecked_ref(idx, is_last) };
                *elem_ref = new_elem;
            }

            for i in 0..len {
                if i == idx {
                    assert_eq!(data.simd_element(idx), new_elem);
                } else {
                    assert_eq!(data.simd_element(i), initial_data.simd_element(i));
                }
            }
        }

        /// Test writing through the get_unchecked_mut of UnalignedDataMut
        #[test]
        fn set_unaligned(
            ((mut data, idx), new_elem) in (unaligned_init_input(V::LANES)
                                                .prop_flat_map(with_data_index),
                                            any_v())
        ) {
            let initial_data = data.clone();
            let len = data.simd_len();
            let is_last = data.is_last(idx);

            {
                let mut unaligned_mut = UnalignedVMut::from(data.as_mut_slice());
                let mut elem_ref = unsafe { unaligned_mut.get_unchecked_ref(idx, is_last) };
                *elem_ref = new_elem;
            }

            for i in 0..len {
                if i == idx {
                    assert_eq!(data.simd_element(idx), new_elem);
                } else {
                    assert_eq!(data.simd_element(i), initial_data.simd_element(i));
                }
            }
        }

        /// Test writing through the get_unchecked_mut of PaddedDataMut
        #[test]
        fn set_padded(
            ((mut padded_data, idx), new_elem) in (padded_init_input(false)
                                                        .prop_flat_map(with_data_index),
                                                   any_v())
        ) {
            let initial_data = padded_data.clone();
            let len = padded_data.simd_len();
            let is_last = padded_data.is_last(idx);

            let mut num_last_elems = padded_data.0.len() % V::LANES;
            if num_last_elems == 0 {
                num_last_elems = V::LANES;
            }

            {
                let (data, padding) = &mut padded_data;
                let mut padded_mut = PaddedVMut::new(data.as_mut_slice(), *padding).unwrap();
                let mut elem_ref = unsafe { padded_mut.get_unchecked_ref(idx, is_last) };
                *elem_ref = new_elem;
            }

            for i in 0..len {
                if i == idx {
                    assert_eq!(
                        padded_data.simd_element(idx),
                        V::from_fn(|i| {
                            if !is_last || i < num_last_elems {
                                new_elem[i]
                            } else {
                                padded_data.1.unwrap()
                            }
                        })
                    );
                } else {
                    assert_eq!(padded_data.simd_element(i), initial_data.simd_element(i));
                }
            }
        }

        /// Test writing through the get_unchecked_mut of TupleData
        #[test]
        fn set_tuple(((mut data, idx), write) in (tuple_init_input(false)
                                                        .prop_flat_map(with_data_index),
                                                     any_tuple_write())) {
            let initial_data = data.clone();
            let len = data.simd_len();
            let old_elem = data.simd_element(idx);
            let is_last = data.is_last(idx);

            {
                let mut tuple = data.as_tuple_data();
                assert_eq!(unsafe { tuple.get_unchecked(idx, is_last) }, old_elem);
                let mut output = unsafe { tuple.get_unchecked_ref(idx, is_last) };
                write_tuple(&mut output, write);
            }

            for i in 0..len {
                if i == idx {
                    check_tuple_write(&data, idx, old_elem, write);
                } else {
                    assert_eq!(data.simd_element(i), initial_data.simd_element(i));
                }
            }
        }

        /// Test the split_at_unchecked of AlignedData(Mut)?
        #[test]
        fn split_aligned((mut data, mid) in aligned_init_input(false).prop_flat_map(with_split_index)) {
            let initial_data = data.clone();
            let base_ptr = data.base_ptr();
            let len = data.simd_len();
            let is_end = mid == len;
            let nonnull = |ptr| NonNull::new(ptr).unwrap();

            {
                let aligned = AlignedV::from(data.as_slice());
                let (lhs, rhs) = unsafe { aligned.split_at_unchecked(mid, len) };
                assert_eq!(lhs, base_ptr);
                if !is_end {
                    assert_eq!(rhs, nonnull(base_ptr.as_ptr().wrapping_add(mid)));
                }
            }
            assert_eq!(data, initial_data);

            {
                let aligned_mut = AlignedVMut::from(data.as_mut_slice());
                let (lhs, rhs) = unsafe { aligned_mut.split_at_unchecked(mid, len) };
                assert_eq!(lhs, base_ptr);
                if !is_end {
                    assert_eq!(rhs, nonnull(base_ptr.as_ptr().wrapping_add(mid)));
                }
            }
            assert_eq!(data, initial_data);
        }

        /// Test the split_at_unchecked of UnalignedData(Mut)?
        #[test]
        fn split_unaligned((mut data, mid) in unaligned_init_input(V::LANES).prop_flat_map(with_split_index)) {
            let initial_data = data.clone();
            let base_ptr = data.base_ptr();
            let len = data.simd_len();
            let is_end = mid == len;
            let nonnull = |ptr| NonNull::new(ptr).unwrap();

            {
                let unaligned = UnalignedV::from(data.as_slice());
                let (lhs, rhs) = unsafe { unaligned.split_at_unchecked(mid, len) };
                assert_eq!(lhs, base_ptr);
                if !is_end {
                    assert_eq!(rhs, nonnull(base_ptr.as_ptr().wrapping_add(mid)));
                }
            }
            assert_eq!(data, initial_data);

            {
                let unaligned_mut = UnalignedVMut::from(data.as_mut_slice());
                let (lhs, rhs) = unsafe { unaligned_mut.split_at_unchecked(mid, len) };
                assert_eq!(lhs, base_ptr);
                if !is_end {
                    assert_eq!(rhs, nonnull(base_ptr.as_ptr().wrapping_add(mid)));
                }
            }
            assert_eq!(data, initial_data);
        }

        /// Test the split_at_unchecked of PaddedData(Mut)?
        #[test]
        fn split_padded((mut padded_data, mid) in padded_init_input(false).prop_flat_map(with_split_index)) {
            let initial_data = padded_data.0.clone();
            let elem_before_mid = padded_data.simd_element(mid.saturating_sub(1));
            let base_ptr = padded_data.base_ptr();
            let len = padded_data.simd_len();
            let nonnull = |ptr| NonNull::new(ptr).unwrap();

            let check_padded_split = |lhs: &PaddedV, rhs: &PaddedV, last_vector: V| {
                if mid != 0 {
                    assert_eq!(lhs.vectors, base_ptr);
                }
                if mid != len {
                    assert_eq!(rhs.vectors, nonnull(base_ptr.as_ptr().wrapping_add(mid)));
                }

                if mid != 0 && mid != len {
                    assert_eq!(
                        unsafe { lhs.last_vector.assume_init() },
                        elem_before_mid,
                    );
                    assert_eq!(
                        unsafe { rhs.last_vector.assume_init() },
                        last_vector,
                    );
                } else if mid == 0 {
                    assert_eq!(
                        unsafe { rhs.last_vector.assume_init() },
                        last_vector,
                    );
                } else if mid == len {
                    assert_eq!(
                        unsafe { lhs.last_vector.assume_init() },
                        last_vector,
                    );
                }
            };

            let num_last_elems = {
                let (data, padding) = &padded_data;
                let (padded, num_last_elems) = PaddedV::new(data.as_slice(), *padding).unwrap();
                let last_vector = unsafe { padded.last_vector.assume_init() };
                let (lhs, rhs) = unsafe { padded.split_at_unchecked(mid, len) };
                check_padded_split(&lhs, &rhs, last_vector);
                num_last_elems
            };
            assert_eq!(padded_data.0, initial_data);

            {
                let (data, padding) = &mut padded_data;
                let padded_mut = PaddedVMut::new(data.as_mut_slice(), *padding).unwrap();
                let last_vector = unsafe { padded_mut.inner.last_vector.assume_init() };
                let (lhs, rhs) = unsafe { padded_mut.split_at_unchecked(mid, len) };
                check_padded_split(&lhs.inner, &rhs.inner, last_vector);
                if mid > 0 && mid < len {
                    assert_eq!(lhs.num_last_elems, V::LANES);
                    assert_eq!(rhs.num_last_elems, num_last_elems);
                } else if mid == 0 {
                    assert_eq!(lhs.num_last_elems, 0);
                    assert_eq!(rhs.num_last_elems, num_last_elems);
                } else if mid == len {
                    assert_eq!(lhs.num_last_elems, num_last_elems);
                    assert_eq!(rhs.num_last_elems, 0);
                }
            }
            assert_eq!(padded_data.0, initial_data);
        }

        /// Test the split_at_unchecked of TupleData
        #[test]
        fn split_tuple((mut data, mid) in tuple_init_input(false).prop_flat_map(with_split_index)) {
            let initial_data = data.clone();
            {
                let base_ptr = data.base_ptr();
                let len = data.simd_len();
                let tuple = data.as_tuple_data();
                let (lhs, rhs) = unsafe { tuple.split_at_unchecked(mid, len) };
                if mid != 0 {
                    assert_eq!(lhs.0, base_ptr.0);
                    assert_eq!(lhs.1, base_ptr.1);
                    assert_eq!(lhs.2, base_ptr.2);
                    assert_eq!(lhs.3, base_ptr.3);
                    assert_eq!(lhs.4.vectors, base_ptr.4);
                    assert_eq!(lhs.5.inner.vectors, base_ptr.5);
                }
                if mid != len {
                    let nonnull_v = |ptr: *mut V| NonNull::new(ptr).unwrap();
                    let nonnull_array = |ptr: *mut VArray| NonNull::new(ptr).unwrap();
                    assert_eq!(rhs.0, nonnull_v(base_ptr.0.as_ptr().wrapping_add(mid)));
                    assert_eq!(rhs.1, nonnull_v(base_ptr.1.as_ptr().wrapping_add(mid)));
                    assert_eq!(rhs.2, nonnull_array(base_ptr.2.as_ptr().wrapping_add(mid)));
                    assert_eq!(rhs.3, nonnull_array(base_ptr.3.as_ptr().wrapping_add(mid)));
                    assert_eq!(rhs.4.vectors, nonnull_array(base_ptr.4.as_ptr().wrapping_add(mid)));
                    assert_eq!(rhs.5.inner.vectors, nonnull_array(base_ptr.5.as_ptr().wrapping_add(mid)));
                }
            }
            assert_eq!(data, initial_data);
        }

        /// Test comparison of AlignedData(Mut)?
        #[test]
        fn cmp_aligned(mut data_pair in uniform2(aligned_init_input(true))) {
            let initial_data = data_pair.clone();
            let base: [_; 2] = core::array::from_fn(|idx| data_pair[idx].base_ptr());
            let eq = base[0] == base[1]; // Should usually be false, we're testing true above
            let cmp = base[0].partial_cmp(&base[1]);

            {
                let aligned: [_; 2] = core::array::from_fn(|idx| AlignedV::from(data_pair[idx].as_slice()));
                assert_eq!(aligned[0] == aligned[1], eq);
                assert_eq!(aligned[0].partial_cmp(&aligned[1]), cmp);
            }
            assert_eq!(data_pair, initial_data);

            {
                let [data1, data2] = &mut data_pair;
                let aligned_mut = [
                    AlignedVMut::from(data1.as_mut_slice()),
                    AlignedVMut::from(data2.as_mut_slice())
                ];
                assert_eq!(aligned_mut[0] == aligned_mut[1], eq);
                assert_eq!(aligned_mut[0].partial_cmp(&aligned_mut[1]), cmp);
            }
            assert_eq!(data_pair, initial_data);
        }

        /// Test comparison of UnalignedData(Mut)?
        #[test]
        fn cmp_unaligned(mut data_pair in uniform2(unaligned_init_input(0))) {
            let initial_data = data_pair.clone();
            let base: [_; 2] = core::array::from_fn(|idx| data_pair[idx].base_ptr());
            let eq = base[0] == base[1]; // Should usually be false, we're testing true above
            let cmp = base[0].partial_cmp(&base[1]);

            {
                let unaligned: [_; 2] = core::array::from_fn(|idx| UnalignedV::from(data_pair[idx].as_slice()));
                assert_eq!(unaligned[0] == unaligned[1], eq);
                assert_eq!(unaligned[0].partial_cmp(&unaligned[1]), cmp);
            }
            assert_eq!(data_pair, initial_data);

            {
                let [data1, data2] = &mut data_pair;
                let unaligned_mut = [
                    UnalignedVMut::from(data1.as_mut_slice()),
                    UnalignedVMut::from(data2.as_mut_slice())
                ];
                assert_eq!(unaligned_mut[0] == unaligned_mut[1], eq);
                assert_eq!(unaligned_mut[0].partial_cmp(&unaligned_mut[1]), cmp);
            }
            assert_eq!(data_pair, initial_data);
        }
    }
}
