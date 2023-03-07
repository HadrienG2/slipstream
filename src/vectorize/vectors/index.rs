//! Indexing of `Vectors`

use crate::vectorize::{data::VectorizedImpl, Slice, VectorInfo, Vectors};
use core::ops::{Bound, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// A helper trait used for [`Vectors`] indexing operations
///
/// Analogous to the standard (unstable)
/// [`SliceIndex`](core::slice::SliceIndex) trait, but for [`Vectors`].
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
            Self::Output::empty()
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
