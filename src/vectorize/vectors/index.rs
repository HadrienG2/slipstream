//! Indexing of `Vectors`

use crate::vectorize::{data::VectorizedImpl, SliceMut, VectorInfo, Vectors};
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
    type OutputMut<'out>
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
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_>;
}

unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for usize {
    type OutputMut<'out> = Data::ElementMut<'out> where Self: 'out, Data: 'out;

    #[inline(always)]
    fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
        *self < vectors.len()
    }

    #[inline(always)]
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_> {
        unsafe {
            vectors
                .data
                .get_unchecked_mut(self, self == vectors.last_idx())
        }
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for RangeFull {
    type OutputMut<'out> = SliceMut<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    fn is_valid_index(&self, _vectors: &Vectors<V, Data>) -> bool {
        true
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_> {
        vectors.as_mut_slice()
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for RangeFrom<usize> {
    type OutputMut<'out> = SliceMut<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
        self.start < vectors.len()
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_> {
        unsafe { vectors.as_mut_slice().split_at_unchecked(self.start) }.1
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for RangeTo<usize> {
    type OutputMut<'out> = SliceMut<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
        self.end <= vectors.len()
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_> {
        unsafe { vectors.as_mut_slice().split_at_unchecked(self.end) }.0
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for Range<usize> {
    type OutputMut<'out> = SliceMut<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
        if self.start < self.end {
            self.start < vectors.len() && self.end <= vectors.len()
        } else {
            true
        }
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_> {
        if self.start < self.end {
            let after_start = unsafe { vectors.as_mut_slice().split_at_unchecked(self.start) }.1;
            unsafe { after_start.split_at_unchecked(self.end - self.start) }.0
        } else {
            Self::OutputMut::empty()
        }
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data> for RangeInclusive<usize> {
    type OutputMut<'out> = SliceMut<'out, V, Data> where Self: 'out, Data: 'out;

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
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_> {
        let (&start, &end) = (self.start(), self.end());
        if end == usize::MAX {
            core::hint::unreachable_unchecked()
        } else {
            (start..end + 1).get_unchecked_mut(vectors)
        }
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data>
    for RangeToInclusive<usize>
{
    type OutputMut<'out> = SliceMut<'out, V, Data> where Self: 'out, Data: 'out;

    #[inline]
    fn is_valid_index(&self, vectors: &Vectors<V, Data>) -> bool {
        (0..=self.end).is_valid_index(vectors)
    }

    #[inline]
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_> {
        (0..=self.end).get_unchecked_mut(vectors)
    }
}

unsafe impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorIndex<V, Data>
    for (Bound<usize>, Bound<usize>)
{
    type OutputMut<'out> = SliceMut<'out, V, Data> where Self: 'out, Data: 'out;

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
    unsafe fn get_unchecked_mut(self, vectors: &mut Vectors<V, Data>) -> Self::OutputMut<'_> {
        let (lower_excluded, upper) = match (self.0, self.1) {
            (Bound::Included(s), Bound::Included(e)) => return (s..=e).get_unchecked_mut(vectors),
            (Bound::Included(s), Bound::Excluded(e)) => return (s..e).get_unchecked_mut(vectors),
            (Bound::Included(s), Bound::Unbounded) => return (s..).get_unchecked_mut(vectors),
            (Bound::Unbounded, Bound::Included(e)) => return (..=e).get_unchecked_mut(vectors),
            (Bound::Unbounded, Bound::Excluded(e)) => return (..e).get_unchecked_mut(vectors),
            (Bound::Unbounded, Bound::Unbounded) => return (..).get_unchecked_mut(vectors),
            (Bound::Excluded(s), upper) => (s, upper),
        };
        let lower_included = if lower_excluded == usize::MAX {
            core::hint::unreachable_unchecked()
        } else {
            lower_excluded + 1
        };
        (Bound::Included(lower_included), upper).get_unchecked_mut(vectors)
    }
}

// FIXME: Tests
