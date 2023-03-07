//! SIMD data collection
//!
//! This module builds upon the `Vectorized` abstraction to provide the
//! `Vectors` type, a `[Vector]`-like view of a heterogeneous data set
//! composed of slices and containers of `Vector`s and scalar elements.
//!
//! It defines most of the public API of the reinterpreted vector data.

use super::{
    array_from_fn,
    data::{VectorizedImpl, VectorizedSliceImpl},
    VectorInfo, Vectorized,
};
use core::{
    iter::FusedIterator,
    marker::PhantomData,
    ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive},
};

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
    /// This operation is only available on slices. You can turn a full dataset
    /// into a slice using `as_slice()`.
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

    /// Like `split_at`, but without bounds checking
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

/// A helper trait used for `Vectors` indexing operations
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

// FIXME: Tests
