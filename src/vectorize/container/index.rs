//! Indexing of `Vectorized`

#[cfg(doc)]
use crate::vectorize::VectorizedData;
use crate::vectorize::{
    data::{VectorizedDataImpl, VectorizedSliceImpl},
    RefSlice, Slice, VectorInfo, Vectorized,
};
use core::ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// A helper trait used for [`Vectorized`] indexing operations
///
/// Analogous to the standard (unstable)
/// [`SliceIndex`](core::slice::SliceIndex) trait, but for [`Vectorized`].
///
/// # Safety
///
/// Unsafe code can rely on this trait being implemented correctly
pub unsafe trait VectorIndex<V: VectorInfo, Data: VectorizedDataImpl<V>> {
    /// Truth that `self` is a valid index for `vectors`
    fn is_valid_index(&self, vectors: &Vectorized<V, Data>) -> bool;

    /// The output type returned by `get_unchecked`
    ///
    /// [`Data::ElementCopy`] if this is a single index, [`Slice`] if this is a
    /// range of target indices.
    ///
    /// [`Data::ElementCopy`]: VectorizedData::ElementCopy
    type Output<'out>
    where
        Self: 'out,
        Data: 'out;

    /// Perform unchecked indexing, only providing read-only access
    ///
    /// # Safety
    ///
    /// `self` must be a valid index for `vectors` (see is_valid_index)
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_>;

    /// The output type returned by `get_unchecked_ref`
    ///
    /// [`Data::ElementRef`] if this is a single index, [`RefSlice`] if this is
    /// a range of target indices.
    ///
    /// [`Data::ElementRef`]: VectorizedData::ElementRef
    type RefOutput<'out>
    where
        Self: 'out,
        Data: 'out;

    /// Perform unchecked indexing, allowing in-place data access
    ///
    /// # Safety
    ///
    /// `self` must be a valid index for `vectors` (see is_valid_index)
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_>;
}

unsafe impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorIndex<V, Data> for usize {
    #[inline(always)]
    fn is_valid_index(&self, vectors: &Vectorized<V, Data>) -> bool {
        *self < vectors.len()
    }

    type Output<'out> = Data::ElementCopy where Self: 'out, Data: 'out;

    #[inline(always)]
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_> {
        unsafe {
            vectors
                .data
                .get_unchecked(self, self == vectors.last_idx().unwrap())
        }
    }

    type RefOutput<'out> = Data::ElementRef<'out> where Self: 'out, Data: 'out;

    #[inline(always)]
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_> {
        unsafe {
            vectors
                .data
                .get_unchecked_ref(self, self == vectors.last_idx().unwrap())
        }
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorIndex<V, Data> for RangeFull {
    #[inline]
    fn is_valid_index(&self, _vectors: &Vectorized<V, Data>) -> bool {
        true
    }

    type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_> {
        vectors.as_slice()
    }

    type RefOutput<'out> = RefSlice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_> {
        vectors.as_ref_slice()
    }
}

// Implementation of get_unchecked for RangeFrom
unsafe fn get_range_from_unchecked<V: VectorInfo, Data: VectorizedSliceImpl<V>>(
    slice: Vectorized<V, Data>,
    start: usize,
) -> Vectorized<V, Data> {
    unsafe { slice.split_at_unchecked(start) }.1
}

unsafe impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorIndex<V, Data> for RangeFrom<usize> {
    #[inline]
    fn is_valid_index(&self, vectors: &Vectorized<V, Data>) -> bool {
        self.start < vectors.len()
    }

    type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_> {
        unsafe { get_range_from_unchecked(vectors.as_slice(), self.start) }
    }

    type RefOutput<'out> = RefSlice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_> {
        unsafe { get_range_from_unchecked(vectors.as_ref_slice(), self.start) }
    }
}

// Implementation of get_unchecked for RangeTo
unsafe fn get_range_to_unchecked<V: VectorInfo, Data: VectorizedSliceImpl<V>>(
    slice: Vectorized<V, Data>,
    end: usize,
) -> Vectorized<V, Data> {
    unsafe { slice.split_at_unchecked(end) }.0
}

unsafe impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorIndex<V, Data> for RangeTo<usize> {
    #[inline]
    fn is_valid_index(&self, vectors: &Vectorized<V, Data>) -> bool {
        self.end <= vectors.len()
    }

    type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_> {
        unsafe { get_range_to_unchecked(vectors.as_slice(), self.end) }
    }

    type RefOutput<'out> = RefSlice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_> {
        unsafe { get_range_to_unchecked(vectors.as_ref_slice(), self.end) }
    }
}

// Implementation of get_unchecked for Range
unsafe fn get_range_unchecked<V: VectorInfo, Data: VectorizedSliceImpl<V>>(
    slice: Vectorized<V, Data>,
    range: Range<usize>,
) -> Vectorized<V, Data> {
    if range.start < range.end {
        let after_start = unsafe { slice.split_at_unchecked(range.start) }.1;
        unsafe { after_start.split_at_unchecked(range.end - range.start) }.0
    } else {
        Vectorized::empty()
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorIndex<V, Data> for Range<usize> {
    #[inline]
    fn is_valid_index(&self, vectors: &Vectorized<V, Data>) -> bool {
        if self.start < self.end {
            self.start < vectors.len() && self.end <= vectors.len()
        } else {
            true
        }
    }

    type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_> {
        get_range_unchecked(vectors.as_slice(), self)
    }

    type RefOutput<'out> = RefSlice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_> {
        get_range_unchecked(vectors.as_ref_slice(), self)
    }
}

// Implementation of get_unchecked for RangeInclusive
unsafe fn get_range_inclusive_unchecked<V: VectorInfo, Data: VectorizedSliceImpl<V>>(
    slice: Vectorized<V, Data>,
    range: RangeInclusive<usize>,
) -> Vectorized<V, Data> {
    let (&start, &end) = (range.start(), range.end());
    if end == usize::MAX {
        core::hint::unreachable_unchecked()
    } else {
        get_range_unchecked(slice, start..end + 1)
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorIndex<V, Data>
    for RangeInclusive<usize>
{
    #[inline]
    fn is_valid_index(&self, vectors: &Vectorized<V, Data>) -> bool {
        let (&start, &end) = (self.start(), self.end());
        if end == usize::MAX {
            false
        } else {
            (start..end + 1).is_valid_index(vectors)
        }
    }

    type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_> {
        get_range_inclusive_unchecked(vectors.as_slice(), self)
    }

    type RefOutput<'out> = RefSlice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_> {
        get_range_inclusive_unchecked(vectors.as_ref_slice(), self)
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorIndex<V, Data>
    for RangeToInclusive<usize>
{
    #[inline]
    fn is_valid_index(&self, vectors: &Vectorized<V, Data>) -> bool {
        (0..=self.end).is_valid_index(vectors)
    }

    type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_> {
        (0..=self.end).get_unchecked(vectors)
    }

    type RefOutput<'out> = RefSlice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_> {
        (0..=self.end).get_unchecked_ref(vectors)
    }
}

// Implementation of get_unchecked for raw pair of bounds
unsafe fn get_bounds_unchecked<V: VectorInfo, Data: VectorizedSliceImpl<V>>(
    slice: Vectorized<V, Data>,
    lower: Bound<usize>,
    upper: Bound<usize>,
) -> Vectorized<V, Data> {
    let (lower_excluded, upper) = match (lower, upper) {
        (Bound::Included(s), Bound::Included(e)) => {
            return get_range_inclusive_unchecked(slice, s..=e)
        }
        (Bound::Included(s), Bound::Excluded(e)) => return get_range_unchecked(slice, s..e),
        (Bound::Included(s), Bound::Unbounded) => return get_range_from_unchecked(slice, s),
        (Bound::Unbounded, Bound::Included(e)) => {
            return get_range_inclusive_unchecked(slice, 0..=e)
        }
        (Bound::Unbounded, Bound::Excluded(e)) => return get_range_to_unchecked(slice, e),
        (Bound::Unbounded, Bound::Unbounded) => return slice,
        (Bound::Excluded(s), upper) => (s, upper),
    };
    let lower_included = if lower_excluded == usize::MAX {
        core::hint::unreachable_unchecked()
    } else {
        lower_excluded + 1
    };
    get_bounds_unchecked(slice, Bound::Included(lower_included), upper)
}

unsafe impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorIndex<V, Data>
    for (Bound<usize>, Bound<usize>)
{
    #[inline]
    fn is_valid_index(&self, vectors: &Vectorized<V, Data>) -> bool {
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

    type Output<'out> = Slice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked(self, vectors: &Vectorized<V, Data>) -> Self::Output<'_> {
        get_bounds_unchecked(vectors.as_slice(), self.0, self.1)
    }

    type RefOutput<'out> = RefSlice<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    unsafe fn get_unchecked_ref(self, vectors: &mut Vectorized<V, Data>) -> Self::RefOutput<'_> {
        get_bounds_unchecked(vectors.as_ref_slice(), self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use crate::vectorize::{
        container::tests::make_vectorized,
        data::tests::{read_tuple, tuple_init_input, with_data_index, SimdData},
    };
    use proptest::prelude::*;
    use std::panic::AssertUnwindSafe;

    // Variation of with_data_index that only produces _invalid_ data indices
    ///
    /// Used to check that safe indexing performs bound checking properly.
    /// Should not be used with unsafe indexing.
    fn with_invalid_data_index<Data: SimdData>(data: Data) -> impl Strategy<Value = (Data, usize)> {
        let simd_len = data.simd_len();
        (Just(data), simd_len..)
    }

    // TODO: slice index generators, think about half-invalid configs too when
    //       two indices are involved. Flip a coin, if false then only right
    //       side is wrong, if true then left side is wrong too (you cannot have
    //       only the left side wrong since 0 is the minimal left index and it's
    //       valid. Think about "weird zeros" as well where left >= right.

    proptest! {
        /// Read SIMD element at a given valid index
        #[test]
        fn get_valid_elem((mut data, idx) in tuple_init_input(false).prop_flat_map(with_data_index)) {
            let initial_data = data.clone();
            {
                let expected = data.simd_element(idx);
                let mut vectorized = make_vectorized(&mut data);
                assert_eq!(vectorized.index(idx), expected);
                assert_eq!(read_tuple(vectorized.index_ref(idx)), expected);
                assert_eq!(vectorized.get(idx).unwrap(), expected);
                assert_eq!(read_tuple(vectorized.get_ref(idx).unwrap()), expected);
                assert_eq!(unsafe { vectorized.get_unchecked(idx) }, expected);
                assert_eq!(read_tuple(unsafe { vectorized.get_unchecked_ref(idx) }), expected);
            }
            assert_eq!(data, initial_data);
        }

        /// Read SIMD element at a given invalid index
        #[test]
        fn get_invalid_elem((mut data, idx) in tuple_init_input(true).prop_flat_map(with_invalid_data_index)) {
            let initial_data = data.clone();
            {
                let mut vectorized = make_vectorized(&mut data);
                std::panic::catch_unwind(AssertUnwindSafe(|| {
                    vectorized.index(idx);
                })).unwrap_err();
                std::panic::catch_unwind(AssertUnwindSafe(|| {
                    vectorized.index_ref(idx);
                })).unwrap_err();
                assert!(vectorized.get(idx).is_none());
                assert!(vectorized.get_ref(idx).is_none());
                // No _unchecked operation at an invalid index!
            }
            assert_eq!(data, initial_data);
        }

        // TODO: Write at valid index => Extract write validation function container::tests::set_first_last()::check_data_write()
        // TODO: Do it again with other forms of slicing/indexing
    }
}
