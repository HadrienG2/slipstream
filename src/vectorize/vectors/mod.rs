//! SIMD data collection
//!
//! This module builds upon the `Vectorized` abstraction to provide the
//! `Vectors` type, a `[Vector]`-like view of a heterogeneous data set
//! composed of slices and containers of `Vector`s and scalar elements.
//!
//! It defines most of the public API of the reinterpreted vector data.

mod index;
mod iterators;

use super::{
    array_from_fn,
    data::{VectorizedImpl, VectorizedSliceImpl},
    VectorInfo, Vectorized,
};
#[cfg(doc)]
use crate::{vectorize::Vectorizable, Vector};
use core::{marker::PhantomData, num::NonZeroUsize};

pub use index::VectorIndex;
pub use iterators::{Chunks, IntoIter, Iter};

/// SIMD data
///
/// This container is built using the [`Vectorizable`] trait.
///
/// It behaves conceptually like an array of [`Vector`] or tuples thereof,
/// with iteration and indexing operations yielding the type that is
/// described in the documentation of [`Vectorizable`].
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
    pub(crate) unsafe fn from_raw_parts(data: Data, len: usize) -> Self {
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

    /// Like [`get()`](Vectors::get()), but panics if index is out of range
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
    /// - Given a single index, it emits [`Data::ElementRef<'_>`](Vectorized::ElementRef).
    /// - Given a range of indices, it emits [`Data::Slice<'_>`](Vectorized::Slice).
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
    pub unsafe fn get_unchecked<I>(&mut self, index: I) -> <I as VectorIndex<V, Data>>::Output<'_>
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

    /// Returns an iterator over `chunk_size` elements of the dataset at a time,
    /// starting at the beginning of the dataset
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not
    /// divide the length of the dataset, then the last chunk will not have
    /// length `chunk_size`.
    ///
    /// See [`chunks_exact()`] for a variant of this iterator that returns
    /// chunks of always exactly `chunk_size` elements, and [`rchunks()`] for
    /// the same iterator but starting at the end of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    #[inline]
    pub fn chunks(&mut self, chunk_size: usize) -> Chunks<V, Data> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("Chunks must have nonzero size");
        Chunks::new(self.as_slice(), chunk_size)
    }

    // TODO: chunks_exact : mark inline

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

    // TODO: Figure out inlining discipline for the following
    // TODO: r?split(_inclusive)?, r?splitn,
    //       contains, (starts|ends)_with,
    //       (sort|select_nth|binary_search)(_unstable)?_by((_cached)?_key)?,
    //       copy_from_slice (optimiser !), partition_point
}
//
/// # Slice-specific methods
///
/// These operations are currently only available on slices of [`Vectors`]. You
/// can extract a slice covering all data within a [`Vectors`] container using
/// [`Vectors::as_slice()`].
//
// --- Internal docs start here ---
//
// The reason why these are not provided for owned vectors and have a signature
// that differs from the equivalent standard slice methods is that in current
// Rust, it is not possible to express that
// `for<'a> Data::Slice::Slice<'a> = Data::Slice<'a>`.
//
// Without having that assertion, we cannot allow the common convenient pattern
// of iteratively splitting a slice to generate smaller slices, unless we
// redefine slice splitting as a consuming operation that returns the same type.
//
// And since owned data cannot be split in general (think arrays), this means
// that splitting must be specific to slices.
impl<V: VectorInfo, Data: VectorizedSliceImpl<V>> Vectors<V, Data> {
    /// Construct an empty slice
    #[inline]
    pub fn empty() -> Self {
        unsafe { Vectors::from_raw_parts(Data::empty(), 0) }
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
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len(), "Split point is out of range");
        unsafe { self.split_at_unchecked(mid) }
    }

    /// Like [`split_at()`](Vectors::split_at()), but without bounds checking
    ///
    /// # Safety
    ///
    /// Calling this method with an out-of-bounds index is undefined
    /// behavior even if the resulting reference is not used. The caller has
    /// to ensure that `0 <= mid <= self.len()`.
    #[inline(always)]
    pub unsafe fn split_at_unchecked(self, mid: usize) -> (Self, Self) {
        let total_len = self.len();
        let (left_data, right_data) = unsafe { self.data.split_at_unchecked(mid, total_len) };
        let wrap = |data, len| unsafe { Vectors::from_raw_parts(data, len) };
        (wrap(left_data, mid), wrap(right_data, total_len - mid))
    }
}

/// Slice of [`Vectors`]
pub type Slice<'a, V, Data> = Vectors<V, <Data as Vectorized<V>>::Slice<'a>>;

/// Aligned SIMD data
pub type AlignedVectors<V, Data> = Vectors<V, <Data as VectorizedImpl<V>>::Aligned>;

/// Unaligned SIMD data
pub type UnalignedVectors<V, Data> = Vectors<V, <Data as VectorizedImpl<V>>::Unaligned>;

/// Padded scalar data treated as SIMD data
pub type PaddedVectors<V, Data> = Vectors<V, Data>;

// FIXME: Tests
