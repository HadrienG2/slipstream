//! SIMD data access
//!
//! This module implements the Vectorized trait family, whose purpose is to let
//! you treat slices and containers of `Vector`s and scalar elements as if they
//! were slices of `Vector`s.
//!
//! It is focused on the low-level unsafe plumbing needed to perform this data
//! reinterpretation. The high-level API that users invoke to manipulate the
//! output slice is mostly defined in the `vectors` sibling module.

mod proxies;

use super::{VectorInfo, VectorizeError};
#[cfg(doc)]
use crate::{
    vectorize::{Vectorizable, Vectors},
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

pub use proxies::{PaddedMut, UnalignedMut};

/// Outcome of reinterpreting [`Vectorizable`] data reinterpreted as [`Vectors`]
///
/// This trait tells you what types you can expect out of the [`Vectors`]
/// collection that is emitted by the [`Vectorizable`] trait.
///
/// To recapitulate the basic rules concerning [`Element`](Self::Element) are:
///
/// - For all supported types T, a [`Vectors`] collection built out of `&[T]` or
///   a collection of T has iterators and getters that emit owned [`Vector`]
///   values.
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
    /// Owned element of the output [`Vectors`] collection
    ///
    /// Yielded by the `IntoIterator` impl that consumes the collection.
    type Element: Sized;

    /// Borrowed element of the output [`Vectors`] collection
    ///
    /// Returned by methods that take `&mut Vectors` and yield individual
    /// [`Vector`] or `&mut Vector` elements.
    ///
    /// Will always be [`Self::Element`] with a reduced lifetime, but this
    /// constraint cannot be expressed in current Rust.
    type ElementRef<'result>: Sized
    where
        Self: 'result;

    /// Slice of this dataset
    ///
    /// Returned by methods that let you borrow a subset of `Vectors`.
    type Slice<'result>: Vectorized<V, Element = Self::ElementRef<'result>> + VectorizedSliceImpl<V>
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
/// get two copies of the same element. In other words, `Element` should be
/// the maximal-lifetime version of `ElementRef`.
///
/// Further, `Slice::ElementRef` should be pretty much the same GAT as
/// `Self::ElementRef`, with just a different `Self` lifetime bound.
///
/// Finally, a `Vectorized` impl is only allowed to implement `Copy` if
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

    /// Reinterpretation of this data as SIMD data that may not be aligned
    type Unaligned: Vectorized<V> + VectorizedImpl<V>;

    /// Unsafely cast this data to the equivalent slice or collection of
    /// unaligned `Vector`s
    ///
    /// # Safety
    ///
    /// The underlying scalar data must have a number of elements that is
    /// a multiple of `V::LANES`.
    unsafe fn as_unaligned_unchecked(self) -> Self::Unaligned;

    /// Reinterpretation of this data as SIMD data with optimal layout
    type Aligned: Vectorized<V> + VectorizedImpl<V>;

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
pub unsafe trait VectorizedSliceImpl<V: VectorInfo>:
    VectorizedImpl<V> + Sized + Debug
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
impl<'other, V: VectorInfo, Ptr: Borrow<NonNull<V>>> PartialOrd<Ptr> for AlignedData<'_, V> {
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
unsafe impl<'target, V: VectorInfo> Vectorized<V> for AlignedData<'target, V> {
    type Element = V;
    type ElementRef<'result> = V where Self: 'result;
    type Slice<'result> = AlignedData<'result, V> where Self: 'result;
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

    type Unaligned = Self;

    unsafe fn as_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = Self;

    unsafe fn as_aligned_unchecked(self) -> Self {
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

// Owned arrays of Vector must be stored as-is in the Vectors collection,
// but otherwise behave like &[Vector]
unsafe impl<V: VectorInfo, const SIZE: usize> Vectorized<V> for [V; SIZE] {
    type Element = V;
    type ElementRef<'result> = V;
    type Slice<'result> = AlignedData<'result, V>;
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

    type Unaligned = Self;

    unsafe fn as_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = Self;

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
impl<'other, V: VectorInfo, Ptr: Borrow<NonNull<V>>> PartialOrd<Ptr> for AlignedDataMut<'_, V> {
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
unsafe impl<'target, V: VectorInfo> Vectorized<V> for AlignedDataMut<'target, V> {
    type Element = &'target mut V;
    type ElementRef<'result> = &'result mut V where Self: 'result;
    type Slice<'result> = AlignedDataMut<'result, V> where Self: 'result;
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

    type Unaligned = Self;

    unsafe fn as_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = Self;

    unsafe fn as_aligned_unchecked(self) -> Self {
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
impl<'other, V: VectorInfo, Ptr: Borrow<NonNull<V::Array>>> PartialOrd<Ptr>
    for UnalignedData<'_, V>
{
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
unsafe impl<'target, V: VectorInfo> Vectorized<V> for UnalignedData<'target, V> {
    type Element = V;
    type ElementRef<'result> = V where Self: 'result;
    type Slice<'result> = UnalignedData<'result, V> where Self: 'result;
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

    type Unaligned = Self;

    unsafe fn as_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = AlignedData<'target, V>;

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
impl<'other, V: VectorInfo, Ptr: Borrow<NonNull<V::Array>>> PartialOrd<Ptr>
    for UnalignedDataMut<'_, V>
{
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
unsafe impl<'target, V: VectorInfo> Vectorized<V> for UnalignedDataMut<'target, V> {
    type Element = Self::ElementRef<'target>;
    type ElementRef<'result> = UnalignedMut<'result, V> where Self: 'result;
    type Slice<'result> = UnalignedDataMut<'result, V> where Self: 'result;
}
//
unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for UnalignedDataMut<'target, V> {
    #[inline(always)]
    unsafe fn get_unchecked(&mut self, idx: usize, _is_last: bool) -> UnalignedMut<V> {
        let target = self.0.get_ptr(idx).as_mut();
        let vector = V::from(*target);
        UnalignedMut::new(vector, target)
    }

    #[inline(always)]
    fn as_slice(&mut self) -> UnalignedDataMut<V> {
        UnalignedDataMut(self.0.as_slice(), PhantomData)
    }

    type Unaligned = Self;

    unsafe fn as_unaligned_unchecked(self) -> Self {
        self
    }

    type Aligned = AlignedDataMut<'target, V>;

    unsafe fn as_aligned_unchecked(self) -> AlignedDataMut<'target, V> {
        AlignedDataMut(self.0.as_aligned_unchecked(), PhantomData)
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
unsafe impl<'target, V: VectorInfo> Vectorized<V> for PaddedData<'target, V> {
    type Element = V;
    type ElementRef<'result> = V where Self: 'result;
    type Slice<'result> = PaddedData<'result, V> where Self: 'result;
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

    type Unaligned = UnalignedData<'target, V>;

    unsafe fn as_unaligned_unchecked(self) -> UnalignedData<'target, V> {
        self.vectors
    }

    type Aligned = AlignedData<'target, V>;

    unsafe fn as_aligned_unchecked(self) -> AlignedData<'target, V> {
        unsafe { self.vectors.as_aligned_unchecked() }
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
unsafe impl<'target, V: VectorInfo> Vectorized<V> for PaddedDataMut<'target, V> {
    type Element = PaddedMut<'target, V>;
    type ElementRef<'result> = PaddedMut<'result, V> where Self: 'result;
    type Slice<'result> = PaddedDataMut<'result, V> where Self: 'result;
}
//
unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for PaddedDataMut<'target, V> {
    #[inline(always)]
    unsafe fn get_unchecked(&mut self, idx: usize, is_last: bool) -> PaddedMut<V> {
        PaddedMut::new(
            self.inner.get_unchecked(idx, is_last),
            core::slice::from_raw_parts_mut(
                self.inner.get_ptr(idx).cast::<V::Scalar>().as_ptr(),
                self.num_elems(is_last),
            ),
        )
    }

    #[inline(always)]
    fn as_slice(&mut self) -> PaddedDataMut<V> {
        PaddedDataMut {
            inner: self.inner.as_slice(),
            num_last_elems: self.num_last_elems,
            lifetime: PhantomData,
        }
    }

    type Unaligned = UnalignedDataMut<'target, V>;

    unsafe fn as_unaligned_unchecked(self) -> UnalignedDataMut<'target, V> {
        unsafe { UnalignedDataMut(self.inner.as_unaligned_unchecked(), PhantomData) }
    }

    type Aligned = AlignedDataMut<'target, V>;

    unsafe fn as_aligned_unchecked(self) -> AlignedDataMut<'target, V> {
        unsafe { AlignedDataMut(self.inner.as_aligned_unchecked(), PhantomData) }
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

/// Tuples of pointers yield tuples of deref results
macro_rules! impl_vectorized_for_tuple {
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

            type Unaligned = ($($t::Unaligned,)*);

            unsafe fn as_unaligned_unchecked(self) -> Self::Unaligned {
                let ($($t,)*) = self;
                unsafe { ($($t.as_unaligned_unchecked(),)*) }
            }

            type Aligned = ($($t::Aligned,)*);

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
    use proptest::prelude::*;
    use std::{collections::hash_map::DefaultHasher, hash::Hasher};

    // === COMMON TEST HARNESS ===

    /// Maximum length (in SIMD vector elements) that we need to test in order
    /// to be sure to observe all interesting effects
    const MAX_SIMD_LEN: usize = 4;

    /// Generate the building blocks to initialize AlignedData(Mut)?
    pub(crate) fn aligned_init_input() -> impl Strategy<Value = Vec<V>> {
        prop::collection::vec(any_v(), 0..=MAX_SIMD_LEN)
    }

    /// Arbitrary array of aligned data
    pub(crate) type AlignedArray = [V; MAX_SIMD_LEN];
    pub(crate) fn any_aligned_array() -> impl Strategy<Value = AlignedArray> {
        prop::array::uniform4(any_v())
    }

    /// Generate the building blocks to initialize UnalignedData(Mut)?
    pub(crate) fn unaligned_init_input() -> impl Strategy<Value = Vec<VScalar>> {
        prop::collection::vec(any::<VScalar>(), 0..=MAX_SIMD_LEN * V::LANES)
    }

    /// Generate the building blocks to initialize PaddedData(Mut)?
    pub(crate) fn padded_init_input() -> impl Strategy<Value = (Vec<VScalar>, Option<VScalar>)> {
        (unaligned_init_input(), any::<Option<VScalar>>())
    }

    // === TESTS FOR THIS MODULE ===

    // Hash a value
    fn hash<T: Hash>(t: &T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    // 'static version of most types for empty slice testing
    type AlignedStatic = AlignedData<'static, V>;
    type AlignedStaticMut = AlignedDataMut<'static, V>;
    type UnalignedStatic = UnalignedData<'static, V>;
    type UnalignedStaticMut = UnalignedDataMut<'static, V>;
    type PaddedStatic = PaddedData<'static, V>;
    type PaddedStaticMut = PaddedDataMut<'static, V>;

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
        // Can't even test equality, will just check constructor doesn't crash
        PaddedStatic::empty();
        PaddedStaticMut::empty();
    }

    // Test properties of a freshly initialized data pointer
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
            + VectorizedImpl<V>,
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

        assert_eq!(data, data);
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

    proptest! {
        // Test initializing AlignedVectors(Mut)?
        #[test]
        fn init_aligned(mut data in aligned_init_input()) {
            let slice_ptr = NonNull::from(&data[..]);
            let aligned_raw = unsafe { AlignedData::from_data_ptr(slice_ptr) };

            let mut aligned = AlignedData::from(&data[..]);
            test_init(
                slice_ptr,
                aligned_raw,
                aligned,
                AlignedStatic::empty(),
                AlignedStaticMut::empty(),
            );
            assert_eq!(aligned.as_slice(), aligned_raw);
            unsafe {
                assert_eq!(aligned.as_aligned_unchecked(), aligned_raw);
                assert_eq!(aligned.as_unaligned_unchecked(), aligned_raw);
            }

            test_init(
                slice_ptr,
                aligned_raw,
                AlignedDataMut::from(&mut data[..]),
                AlignedStatic::empty(),
                AlignedStaticMut::empty(),
            );
            let mut aligned_mut = AlignedDataMut::from(&mut data[..]);
            assert_eq!(aligned_mut.as_slice(), aligned_raw);
            unsafe {
                assert_eq!(aligned_mut.as_aligned_unchecked(), aligned_raw);
                assert_eq!(AlignedDataMut::from(&mut data[..]).as_unaligned_unchecked(), aligned_raw);
            }
        }

        // Test treating arrays as shared slices
        #[test]
        fn init_array(mut data in any_aligned_array()) {
            let data_ptr = |data: &AlignedArray| NonNull::from(data).cast::<V>();
            let slice = |data| <AlignedArray as VectorizedImpl<V>>::as_slice(data);

            let mut aligned = unsafe { data.as_aligned_unchecked() };
            let mut unaligned = unsafe { data.as_unaligned_unchecked() };

            let mut ptr = data_ptr(&data);
            assert_eq!(slice(&mut data), ptr);

            ptr = data_ptr(&aligned);
            assert_eq!(slice(&mut aligned), ptr);

            ptr = data_ptr(&unaligned);
            assert_eq!(slice(&mut unaligned), ptr);
        }

        // Test initializing UnalignedVectors(Mut)?
        #[test]
        fn init_unaligned(mut data in unaligned_init_input()) {
            let slice_ptr = NonNull::from(&data[..]);
            let unaligned_raw = unsafe { UnalignedData::from_data_ptr(slice_ptr) };

            let array_slice = unsafe { std::slice::from_raw_parts(
                slice_ptr.cast::<VArray>().as_ptr(),
                data.len() / V::LANES
            ) };
            let array_slice_ptr = NonNull::from(array_slice);

            let mut unaligned = UnalignedData::<V>::from(&data[..]);
            test_init(
                array_slice_ptr,
                unaligned_raw,
                unaligned,
                UnalignedStatic::empty(),
                UnalignedStaticMut::empty(),
            );
            assert_eq!(unaligned.as_slice(), unaligned_raw);
            unsafe {
                // FIXME: Must meet preconditions first, do this via slice::align_mut
                //        then extract main test logic in a function that takes
                //        &mut [VScalar] and recurse to that.
                // assert_eq!(unaligned.as_aligned_unchecked(), unaligned_raw);
                assert_eq!(unaligned.as_unaligned_unchecked(), unaligned_raw);
            }

            type UnalignedMut<'a> = UnalignedDataMut<'a, V>;
            let mut unaligned_mut = UnalignedMut::from(&mut data[..]);
            test_init(
                array_slice_ptr,
                unaligned_raw,
                unaligned_mut,
                UnalignedStatic::empty(),
                UnalignedStaticMut::empty(),
            );
            unaligned_mut = UnalignedMut::from(&mut data[..]);
            assert_eq!(unaligned_mut.as_slice(), unaligned_raw);
            unsafe {
                // FIXME: See above
                // assert_eq!(aligned_mut.as_aligned_unchecked(), aligned_raw);
                assert_eq!(UnalignedMut::from(&mut data[..]).as_unaligned_unchecked(), unaligned_raw);
            }
        }

        // TODO: Test PaddedData(Mut)?, test moar ops
    }

    /* TODO: Ops that still need testing

    // Implemented for everything including AlignedArray
    unsafe impl<'target, V: VectorInfo> VectorizedImpl<V> for AlignedData<'target, V> {
        #[inline(always)]
        unsafe fn get_unchecked(&mut self, idx: usize, _is_last: bool) -> V {
            unsafe { *self.get_ptr(idx).as_ref() }
        }
    }

    // Not implemented for *Mut and AlignedArray
    impl<'target, V: VectorInfo> AlignedData<'target, V> {
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

    // Not implemented for PaddedData(Mut)?
    impl<V: VectorInfo> PartialEq for AlignedData<'_, V> {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
    //
    impl<V: VectorInfo> PartialOrd for AlignedData<'_, V> {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    // Not implemented for AlignedArray
    unsafe impl<'target, V: VectorInfo> VectorizedSliceImpl<V> for AlignedData<'target, V> {
        #[inline(always)]
        unsafe fn split_at_unchecked(self, mid: usize, _len: usize) -> (Self, Self) {
            let wrap = |ptr| Self(ptr, PhantomData);
            (wrap(self.0), wrap(self.get_ptr(mid)))
        }
    }

    */
}
