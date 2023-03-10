//! Iteration over values within `Vectors`

use crate::vectorize::{data::VectorizedImpl, RefSlice, Slice, VectorInfo, Vectors};
use core::iter::FusedIterator;

/// Read-only [`Vectors`] iterator
pub struct Iter<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> {
    vectors: &'vectors Vectors<V, Data>,
    start: usize,
    end: usize,
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V> + 'vectors> IntoIterator
    for &'vectors Vectors<V, Data>
{
    type Item = Data::ElementCopy;
    type IntoIter = Iter<'vectors, V, Data>;

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

/// In-place [`Vectors`] iterator
pub struct RefIter<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> {
    vectors: &'vectors mut Vectors<V, Data>,
    start: usize,
    end: usize,
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V> + 'vectors> IntoIterator
    for &'vectors mut Vectors<V, Data>
{
    type Item = Data::ElementRef<'vectors>;
    type IntoIter = RefIter<'vectors, V, Data>;

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

/// Consuming [`Vectors`] iterator
pub struct IntoIter<V: VectorInfo, Data: VectorizedImpl<V>> {
    vectors: Vectors<V, Data>,
    start: usize,
    end: usize,
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V> + 'vectors> IntoIterator
    for Vectors<V, Data>
{
    type Item = Data::Element;
    type IntoIter = IntoIter<V, Data>;

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

// Common Iterator implementation boilerplate
macro_rules! impl_iterator {
    (
        ($name:ident$(<$iter_lifetime:lifetime>)?, $get_unchecked:ident, $slice:ident, Data::$elem:ident$(<$elem_lifetime:lifetime>)?)
    ) => {
        impl<$($iter_lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> $name<$($iter_lifetime,)? V, Data> {
            /// Get the i-th element
            ///
            /// # Safety
            ///
            /// - This should only be called once per index, i.e. you must ensure
            ///   that the iterator will not visit this index again.
            /// - This should only be called for valid indices of the underlying Vectors.
            #[inline(always)]
            unsafe fn get_elem<'iter>(&'iter mut self, idx: usize) -> Data::$elem$(<$elem_lifetime>)? {
                debug_assert!(idx < self.vectors.len());
                let result = unsafe { self.vectors.$get_unchecked(idx) };
                unsafe { core::mem::transmute_copy(&result) }
            }
        }
        //
        impl<$($iter_lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> Iterator
            for $name<$($iter_lifetime,)? V, Data>
        {
            type Item = Data::$elem$(<$elem_lifetime>)?;

            #[inline(always)]
            fn next(&mut self) -> Option<Self::Item> {
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
            fn last(mut self) -> Option<Self::Item> {
                self.next_back()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                if self.len() >= n {
                    self.start += n - 1;
                    self.next()
                } else {
                    self.start = self.end;
                    None
                }
            }
        }
        //
        impl<$($iter_lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> DoubleEndedIterator
            for $name<$($iter_lifetime,)? V, Data>
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
                if self.len() >= n {
                    self.end -= n - 1;
                    self.next_back()
                } else {
                    self.end = self.start;
                    None
                }
            }
        }
        //
        impl<$($iter_lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> ExactSizeIterator
            for $name<$($iter_lifetime,)? V, Data>
        {
            #[inline]
            fn len(&self) -> usize {
                self.end - self.start
            }
        }
        //
        impl<$($iter_lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> FusedIterator
            for $name<$($iter_lifetime,)? V, Data>
        {
        }
        //
        #[cfg(feature = "iterator_ilp")]
        unsafe impl<$($iter_lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> iterator_ilp::TrustedLowerBound
            for $name<$($iter_lifetime,)? V, Data>
        {
        }
    }
}
//
impl_iterator!((Iter<'vectors>, get_unchecked, Slice, Data::ElementCopy));
impl_iterator!((
    RefIter<'vectors>,
    get_unchecked_ref,
    SliceMut,
    Data::ElementRef<'vectors>
));
impl_iterator!((IntoIter, get_unchecked_ref, SliceMut, Data::Element));

// Conversion to slice is iterator type specific
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> Iter<'vectors, V, Data> {
    /// Views the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the iterator
    /// can continue to be used while this exists.
    #[inline]
    pub fn as_slice(&self) -> Slice<'vectors, V, Data> {
        unsafe { self.vectors.get_unchecked(self.start..self.end) }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> RefIter<'vectors, V, Data> {
    /// Views the underlying data as a subslice of the original
    /// data.
    ///
    /// To avoid creating `&mut V` references that alias, this
    /// is forced to consume the iterator
    #[inline]
    pub fn into_slice(self) -> RefSlice<'vectors, V, Data> {
        unsafe { self.vectors.get_unchecked_ref(self.start..self.end) }
    }

    /// Views the underlying data as a subslice of the original data.
    ///
    /// To avoid creating `&mut V` references that alias, the
    /// returned slice borrows its lifetime from the iterator.
    #[inline]
    pub fn as_slice(&mut self) -> Slice<V, Data> {
        unsafe { self.vectors.get_unchecked(self.start..self.end) }
    }
}
//
impl<V: VectorInfo, Data: VectorizedImpl<V>> IntoIter<V, Data> {
    /// Remaining items of this iterator, as a read-only slice
    #[inline]
    pub fn as_slice(&self) -> Slice<V, Data> {
        unsafe { self.vectors.get_unchecked(self.start..self.end) }
    }

    /// Remaining items of this iterator, as an in-place slice
    #[inline]
    pub fn as_ref_slice(&mut self) -> RefSlice<V, Data> {
        unsafe { self.vectors.get_unchecked_ref(self.start..self.end) }
    }
}

// FIXME: Tests
