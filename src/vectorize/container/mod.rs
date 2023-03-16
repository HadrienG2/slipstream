//! SIMD data collection
//!
//! This module builds upon the `VectorizedData` abstraction to provide the
//! `Vectorized` type, a `[Vector]`-like view of a heterogeneous data set
//! composed of slices and containers of `Vector`s and scalar elements.
//!
//! It defines most of the public API of the reinterpreted vector data.

mod index;
mod iterators;

use super::{
    data::{VectorizedDataImpl, VectorizedSliceImpl},
    VectorInfo, VectorizedData,
};
#[cfg(doc)]
use crate::{vectorize::Vectorizable, Vector};
use core::{
    borrow::Borrow,
    fmt::{self, Debug},
    marker::PhantomData,
    num::NonZeroUsize,
};

// NOTE: Remember to re-export these in the parent vectorize module
pub use index::VectorIndex;
pub use iterators::{
    Chunks, ChunksExact, GenericChunks, GenericChunksExact, IntoIter, Iter, RefChunks,
    RefChunksExact, RefIter,
};

/// SIMD data
///
/// This container is built using the [`Vectorizable`] trait.
///
/// It behaves conceptually like an array of [`Vector`] or tuples thereof,
/// with iteration and indexing operations yielding the type that is
/// described in the documentation of [`Vectorizable`].
#[derive(Copy, Clone)]
pub struct Vectorized<V: VectorInfo, Data: VectorizedDataImpl<V>> {
    data: Data,
    len: usize,
    vectors: PhantomData<V>,
}
//
impl<V: VectorInfo, Data: VectorizedDataImpl<V>> Vectorized<V, Data> {
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
    pub fn first(&self) -> Option<Data::ElementCopy> {
        self.get(0)
    }

    /// Returns the first element, a mutable reference to it if the underlying
    /// dataset is mutable, or None if the container is empty
    #[inline]
    pub fn first_ref<'a>(&'a mut self) -> Option<Data::ElementRef<'a>> {
        self.get_ref(0)
    }

    /// Returns the first and all the rest of the elements of the container,
    /// or None if it is empty.
    #[inline(always)]
    pub fn split_first(&self) -> Option<(Data::ElementCopy, Slice<V, Data>)> {
        (!self.is_empty()).then(move || {
            let (head, tail) = unsafe { self.as_slice().split_at_unchecked(1) };
            (head.into_iter().next().unwrap(), tail)
        })
    }

    /// Returns the first and all the rest of the elements of the container,
    /// or None if it is empty, allowing in-place access
    #[inline(always)]
    pub fn split_first_ref(&mut self) -> Option<(Data::ElementRef<'_>, RefSlice<V, Data>)> {
        (!self.is_empty()).then(move || {
            let (head, tail) = unsafe { self.as_ref_slice().split_at_unchecked(1) };
            (head.into_iter().next().unwrap(), tail)
        })
    }

    /// Returns the last and all the rest of the elements of the container,
    /// or None if it is empty.
    #[inline(always)]
    pub fn split_last(&self) -> Option<(Data::ElementCopy, Slice<V, Data>)> {
        self.last_idx().map(move |last| {
            let (head, tail) = unsafe { self.as_slice().split_at_unchecked(last) };
            (tail.into_iter().next().unwrap(), head)
        })
    }

    /// Returns the last and all the rest of the elements of the container,
    /// or None if it is empty, allowing in-place access.
    #[inline(always)]
    pub fn split_last_ref(&mut self) -> Option<(Data::ElementRef<'_>, RefSlice<V, Data>)> {
        self.last_idx().map(move |last| {
            let (head, tail) = unsafe { self.as_ref_slice().split_at_unchecked(last) };
            (tail.into_iter().next().unwrap(), head)
        })
    }

    /// Returns the last element, or None if the container is empty
    #[inline]
    pub fn last(&self) -> Option<Data::ElementCopy> {
        self.last_idx().and_then(move |last| self.get(last))
    }

    /// Returns the last element, a mutable reference to it if the underlying
    /// dataset is mutable, or None if the container is empty
    #[inline]
    pub fn last_ref(&mut self) -> Option<Data::ElementRef<'_>> {
        self.last_idx().and_then(move |last| self.get_ref(last))
    }

    /// Index of the last element
    #[inline(always)]
    fn last_idx(&self) -> Option<usize> {
        self.len.checked_sub(1)
    }

    /// Like [`get()`](Vectorized::get_ref()), but panics if index is out of range
    ///
    /// # Panics
    ///
    /// If index is out of range
    //
    // NOTE: We can't implement the real Index trait because it requires
    //       returning a &V that we don't have for padded/unaligned data.
    #[inline(always)]
    pub fn index<I>(&self, index: I) -> <I as VectorIndex<V, Data>>::Output<'_>
    where
        I: VectorIndex<V, Data>,
    {
        self.get(index).expect("Index is out of range")
    }

    /// Like [`get_ref()`](Vectorized::get_ref()), but panics if index is out of range
    ///
    /// # Panics
    ///
    /// If index is out of range
    //
    // NOTE: We can't implement the real Index trait because it requires
    //       returning a &V that we don't have for padded/unaligned data.
    #[inline(always)]
    pub fn index_ref<I>(&mut self, index: I) -> <I as VectorIndex<V, Data>>::RefOutput<'_>
    where
        I: VectorIndex<V, Data>,
    {
        self.get_ref(index).expect("Index is out of range")
    }

    /// Returns the specified element(s) of the container
    ///
    /// This operation accepts either a single `usize` index or a range of
    /// `usize` indices:
    ///
    /// - Given a single index, it emits [`Data::ElementCopy`](VectorizedData::ElementCopy).
    /// - Given a range of indices, it emits [`Slice`].
    ///
    /// If one or more of the specified indices is out of range, None is
    /// returned.
    #[inline(always)]
    pub fn get<I>(&self, index: I) -> Option<<I as VectorIndex<V, Data>>::Output<'_>>
    where
        I: VectorIndex<V, Data>,
    {
        (index.is_valid_index(self)).then(move || unsafe { self.get_unchecked(index) })
    }

    /// Access the specified element(s) of the container, allowing in-place
    /// mutation if possible.
    ///
    /// This operation accepts either a single `usize` index or a range of
    /// `usize` indices:
    ///
    /// - Given a single index, it emits [`Data::ElementRef<'_>`](VectorizedData::ElementRef).
    /// - Given a range of indices, it emits [`RefSlice`]
    ///
    /// If one or more of the specified indices is out of range, None is
    /// returned.
    #[inline(always)]
    pub fn get_ref<I>(&mut self, index: I) -> Option<<I as VectorIndex<V, Data>>::RefOutput<'_>>
    where
        I: VectorIndex<V, Data>,
    {
        (index.is_valid_index(self)).then(move || unsafe { self.get_unchecked_ref(index) })
    }

    /// Returns the specified element(s) of the container without bounds
    /// checking
    ///
    /// # Safety
    ///
    /// Indices covered by `index` must be in range `0..self.len()`
    #[inline(always)]
    pub unsafe fn get_unchecked<I>(&self, index: I) -> <I as VectorIndex<V, Data>>::Output<'_>
    where
        I: VectorIndex<V, Data>,
    {
        unsafe { index.get_unchecked(self) }
    }

    /// Access the specified element(s) of the container without bounds
    /// checking, allowing in-place mutation if possible
    ///
    /// # Safety
    ///
    /// Indices covered by `index` must be in range `0..self.len()`
    #[inline(always)]
    pub unsafe fn get_unchecked_ref<I>(
        &mut self,
        index: I,
    ) -> <I as VectorIndex<V, Data>>::RefOutput<'_>
    where
        I: VectorIndex<V, Data>,
    {
        unsafe { index.get_unchecked_ref(self) }
    }

    /// Returns an iterator over copies of contained elements
    #[inline]
    pub fn iter(&self) -> Iter<V, Data> {
        <&Self>::into_iter(self)
    }

    /// Returns an iterator over contained elements
    #[inline]
    pub fn iter_ref(&mut self) -> RefIter<V, Data> {
        <&mut Self>::into_iter(self)
    }

    // TODO: windows: mark inline

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
    ///
    /// [`chunks_exact()`]: Vectorized::chunks_exact()
    /// [`rchunks()`]: Vectorized::rchunks()
    #[inline]
    pub fn chunks(&self, chunk_size: usize) -> Chunks<V, Data> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("Chunks must have nonzero size");
        Chunks::<V, Data>::new(self.as_slice(), chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the dataset at a time,
    /// starting at the beginning of the dataset and allowing mutation.
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not
    /// divide the length of the dataset, then the last chunk will not have
    /// length `chunk_size`.
    ///
    /// See [`chunks_exact_ref()`] for a variant of this iterator that returns
    /// chunks of always exactly `chunk_size` elements, and [`rchunks_ref()`] for
    /// the same iterator but starting at the end of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// [`chunks_exact_ref()`]: Vectorized::chunks_exact_ref()
    /// [`rchunks_ref()`]: Vectorized::rchunks_ref()
    #[inline]
    pub fn chunks_ref(&mut self, chunk_size: usize) -> RefChunks<V, Data> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("Chunks must have nonzero size");
        RefChunks::<V, Data>::new(self.as_ref_slice(), chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the dataset at a time,
    /// starting at the beginning of the dataset
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not
    /// divide the length of the dataset, then the last up to `chunk_size-1`
    /// elements will be omitted and can be retrieved from the
    /// [`remainder()`](ChunksExact::remainder()) function of the iterator.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler can
    /// often optimize the resulting code better than in the case of
    /// [`chunks()`].
    ///
    /// See [`chunks()`] for a variant of this iterator that also returns the
    /// remainder as a smaller chunk, and [`rchunks_exact()`] for the same
    /// iterator but starting at the end of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// [`chunks()`]: Vectorized::chunks()
    /// [`rchunks()`]: Vectorized::rchunks()
    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<V, Data> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("Chunks must have nonzero size");
        ChunksExact::<V, Data>::new(self.as_slice(), chunk_size)
    }

    /// Returns an iterator over `chunk_size` elements of the dataset at a time,
    /// starting at the beginning of the dataset and allowing mutation
    ///
    /// The chunks are slices and do not overlap. If `chunk_size` does not
    /// divide the length of the dataset, then the last up to `chunk_size-1`
    /// elements will be omitted and can be retrieved from the
    /// [`into_remainder()`](RefChunksExact::into_remainder()) function of the
    /// iterator.
    ///
    /// Due to each chunk having exactly `chunk_size` elements, the compiler can
    /// often optimize the resulting code better than in the case of
    /// [`chunks_ref()`].
    ///
    /// See [`chunks_ref()`] for a variant of this iterator that also returns the
    /// remainder as a smaller chunk, and [`rchunks_exact_ref()`] for the same
    /// iterator but starting at the end of the slice.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    ///
    /// [`chunks_ref()`]: Vectorized::chunks_ref()
    /// [`rchunks_ref()`]: Vectorized::rchunks_ref()
    #[inline]
    pub fn chunks_exact_ref(&mut self, chunk_size: usize) -> RefChunksExact<V, Data> {
        let chunk_size = NonZeroUsize::new(chunk_size).expect("Chunks must have nonzero size");
        RefChunksExact::<V, Data>::new(self.as_ref_slice(), chunk_size)
    }

    // TODO: rchunks(_exact)?(_ref)? : mark inline

    /// Extract a slice covering the entire dataset, with read-only access
    ///
    /// Equivalent to `self.index(..)`
    #[inline]
    pub fn as_slice(&self) -> Slice<V, Data> {
        unsafe { Vectorized::from_raw_parts(self.data.as_slice(), self.len) }
    }

    /// Extract a slice covering the entire dataset, allowing in-place access
    ///
    /// Equivalent to `self.index_ref(..)`
    #[inline]
    pub fn as_ref_slice(&mut self) -> RefSlice<V, Data> {
        unsafe { Vectorized::from_raw_parts(self.data.as_ref_slice(), self.len) }
    }

    // TODO: Figure out inlining discipline for the following
    // TODO: Comparison-based methods: r?split(_inclusive)?(_ref)?, r?splitn(_ref)?,
    //       contains, (starts|ends)_with,
    //       (select_nth|binary_search)(_unstable)?_by((_cached)?_key)?,
    //       partition_point
    // TODO: is_ascii, eq_ignore_ascii_case, escape_ascii
    // TODO: Methods that require dataset mutability and could only be
    //       implemented for Vectorized with an underlying mutable dataset (which
    //       can be done via a requirement on ElementRef):
    //       copy_from_slice, copy_within, make_ascii_(lower|upper)case, fill,
    //       fill_with, reverse, rotate_left, rotate_right, sort(_unstable)?_by((_cached)?_key)?,
    //       swap, swap_with_slice
}
//
/// # Slice-specific methods
///
/// These operations are currently only available on slices of [`Vectorized`]. You
/// can extract a slice covering all data within a [`Vectorized`] container using
/// [`Vectorized::as_slice()`] or [`Vectorized::as_ref_slice()`].
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
impl<V: VectorInfo, Data: VectorizedSliceImpl<V>> Vectorized<V, Data> {
    /// Construct an empty slice
    #[inline]
    pub fn empty() -> Self {
        unsafe { Vectorized::from_raw_parts(Data::empty(), 0) }
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

    /// Like [`split_at()`](Vectorized::split_at()), but without bounds checking
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
        let wrap = |data, len| unsafe { Vectorized::from_raw_parts(data, len) };
        (wrap(left_data, mid), wrap(right_data, total_len - mid))
    }
}
//
/// V: Debug implies Data::ElementCopy: Debug, but we can't prove it to rustc yet
impl<V: VectorInfo + Debug, Data: VectorizedDataImpl<V>> Debug for Vectorized<V, Data>
where
    Data::ElementCopy: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
//
/// Applies if the two vectors emit elements of type V or tuples of the same arity
impl<V: VectorInfo + PartialEq, Data1: VectorizedDataImpl<V>, Data2: VectorizedDataImpl<V>>
    PartialEq<Vectorized<V, Data2>> for Vectorized<V, Data1>
where
    Data1::ElementCopy: PartialEq<Data2::ElementCopy>,
{
    fn eq(&self, other: &Vectorized<V, Data2>) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
//
/// Applies if this was built from a single slice/container, not a tuple of data
impl<V: VectorInfo + PartialEq, Data: VectorizedDataImpl<V>> PartialEq<[V]> for Vectorized<V, Data>
where
    Data::ElementCopy: Borrow<V>,
{
    fn eq(&self, other: &[V]) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .map(|a| *a.borrow())
                .zip(other.iter().copied())
                .all(|(a, b)| a == b)
    }
}

/// Read-only slice of [`Vectorized`]
pub type Slice<'a, V, Data> = Vectorized<V, <Data as VectorizedData<V>>::CopySlice<'a>>;

/// Slice of [`Vectorized`] allowing in-place mutation
pub type RefSlice<'a, V, Data> = Vectorized<V, <Data as VectorizedData<V>>::RefSlice<'a>>;

/// Aligned SIMD data
pub type VectorizedAligned<V, Data> = Vectorized<V, <Data as VectorizedDataImpl<V>>::Aligned>;

/// Unaligned SIMD data
pub type VectorizedUnaligned<V, Data> = Vectorized<V, <Data as VectorizedDataImpl<V>>::Unaligned>;

/// Padded scalar data treated as SIMD data
pub type VectorizedPadded<V, Data> = Vectorized<V, Data>;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::vectorize::{
        data::tests::{read_tuple, tuple_init_input, SimdData, TupleData, TupleInitInput},
        tests::V,
    };
    use proptest::prelude::*;

    // TODO: pub(crate) slice index generators
    // TODO: Add tuple data generator to data/mod.rs and use it for tuple tests

    proptest! {
        /// Test properties of a freshly created Vectorized container
        #[test]
        fn init(mut data in tuple_init_input(true)) {
            let initial_data = data.clone();
            {
                let len = data.simd_len();
                let is_empty = len == 0;
                let first = (!is_empty).then(|| data.simd_element(0));
                let last = (!is_empty).then(|| data.simd_element(len - 1));

                fn make_vectorized(data: &mut TupleInitInput) -> Vectorized<V, TupleData> {
                    let len = data.simd_len();
                    let data = data.as_tuple_data();
                    unsafe { Vectorized::from_raw_parts(data, len) }
                }
                let mut vectorized = make_vectorized(&mut data);

                fn check_basics<V: VectorInfo, Data: VectorizedDataImpl<V>>(
                    vectorized: &mut Vectorized<V, Data>,
                    len: usize,
                    first: Option<Data::ElementCopy>,
                    last: Option<Data::ElementCopy>,
                )
                    where Data::ElementCopy: Debug + PartialEq,
                {
                    assert_eq!(vectorized.len(), len);
                    assert_eq!(!vectorized.is_empty(), len != 0);
                    assert_eq!(vectorized.first(), first);
                    assert_eq!(vectorized.last(), last);
                }
                check_basics(&mut vectorized, len, first, last);
                check_basics(&mut vectorized.as_slice(), len, first, last);
                check_basics(&mut vectorized.as_ref_slice(), len, first, last);
                assert_eq!(vectorized, vectorized.as_slice());
                assert_eq!(vectorized == Vectorized::<V, TupleData>::empty(), is_empty);

                // Handle empty input special case
                if len == 0 {
                    fn check_empty<V: VectorInfo, Data: VectorizedDataImpl<V>>(
                        vectorized: &mut Vectorized<V, Data>,
                    ) {
                        assert!(vectorized.first_ref().is_none());
                        assert!(vectorized.last_ref().is_none());
                        assert!(vectorized.split_first().is_none());
                        assert!(vectorized.split_first_ref().is_none());
                        assert!(vectorized.split_last().is_none());
                        assert!(vectorized.split_last_ref().is_none());
                    }
                    check_empty(&mut vectorized);
                    check_empty(&mut vectorized.as_slice());
                    check_empty(&mut vectorized.as_ref_slice());
                    return Ok(());
                }

                // Handle non-empty inputs, starting with basic sanity checks which,
                // due to current annoying borrow checker limitations that lead to
                // an assumption of 'static lifetime on GATs, must be partially
                // macro'ed instead of fully implemented as functions.
                let first = first.unwrap();
                let last = last.unwrap();
                macro_rules! check_non_empty_ref {
                    ($vectorized:expr) => { #[allow(unused_mut)] {
                        let mut vectorized = $vectorized;
                        assert_eq!(vectorized.first_ref().map(read_tuple), Some(first));
                        assert_eq!(vectorized.last_ref().map(read_tuple), Some(last));
                    } }
                }
                check_non_empty_ref!(&mut vectorized);
                check_non_empty_ref!(vectorized.as_ref_slice());
                {
                    let mut vectorized_slice = vectorized.as_slice();
                    assert_eq!(vectorized_slice.first_ref(), Some(first));
                    assert_eq!(vectorized_slice.last_ref(), Some(last));
                }

                // Splitting is destructive, so we must recreate the container
                // or slice being split after every test.
                type TupleElem = <TupleInitInput as SimdData>::Element;
                fn test_split<
                    'vec,
                    'data: 'vec,
                    BorrowedElem,
                    SourceData: VectorizedDataImpl<V> + 'data,
                    RestData: VectorizedSliceImpl<V> + 'data,
                >(
                    vectorized: &'vec mut Vectorized<V, SourceData>,
                    split: impl FnOnce(&'vec mut Vectorized<V, SourceData>) -> Option<(BorrowedElem, Vectorized<V, RestData>)>,
                    expected_elem: TupleElem,
                    check_elem: impl FnOnce(BorrowedElem) -> TupleElem,
                    expected_other_side: TupleElem,
                    check_other_side: impl FnOnce(&Vectorized<V, RestData>) -> Option<TupleElem>,
                ) {
                    let len = vectorized.len();
                    let (elem, rest) = split(vectorized).unwrap();
                    assert_eq!(check_elem(elem), expected_elem);
                    assert_eq!(rest.len(), len - 1);
                    let expected_other_side = (rest.len() > 0).then_some(expected_other_side);
                    assert_eq!(check_other_side(&rest), expected_other_side);
                }
                // This part must be a macro because make_vectorized(&mut data)
                // cannot be packaged as a lambda as that would entail leaking
                // out references to the lambda's inner state, which is forbidden
                macro_rules! test_all_splits(
                    (
                        $transform_vectorized:expr,
                        $read_ref:path
                    ) => {
                        {
                            let mut vectorized = make_vectorized(&mut data);
                            let mut transformed = $transform_vectorized(&mut vectorized);
                            test_split(
                                &mut transformed,
                                |vectorized| vectorized.split_first(),
                                first,
                                core::convert::identity,
                                last,
                                |rest| rest.last()
                            );
                        }
                        {
                            let mut vectorized = make_vectorized(&mut data);
                            let mut transformed = $transform_vectorized(&mut vectorized);
                            test_split(
                                &mut transformed,
                                |vectorized| vectorized.split_first_ref(),
                                first,
                                $read_ref,
                                last,
                                |rest| rest.last()
                            );
                        }
                        {
                            let mut vectorized = make_vectorized(&mut data);
                            let mut transformed = $transform_vectorized(&mut vectorized);
                            test_split(
                                &mut transformed,
                                |vectorized| vectorized.split_last(),
                                last,
                                core::convert::identity,
                                first,
                                |rest| rest.first()
                            );
                        }
                        {
                            let mut vectorized = make_vectorized(&mut data);
                            let mut transformed = $transform_vectorized(&mut vectorized);
                            test_split(
                                &mut transformed,
                                |vectorized| vectorized.split_last_ref(),
                                last,
                                $read_ref,
                                first,
                                |rest| rest.first()
                            );
                        }
                    }
                );
                test_all_splits!(core::convert::identity, read_tuple);
                test_all_splits!(Vectorized::as_slice, core::convert::identity);
                test_all_splits!(Vectorized::as_ref_slice, read_tuple);
            }
            assert_eq!(data, initial_data);
        }

        // TODO: Test setting data via _ref accessors
        // TODO: Test split_at(_unchecked)?, remember to test self.len() and above
        // TODO: Leave testing of indexing, iterators and chunks to the dedicated modules
    }
}
