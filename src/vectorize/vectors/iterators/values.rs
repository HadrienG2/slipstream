//! Iteration over values within `Vectors`

use crate::vectorize::{data::VectorizedImpl, RefSlice, Slice, VectorInfo, Vectors};
use core::iter::FusedIterator;

/// Genericity over Vectors, &Vectors and &mut Vectors
#[doc(hidden)]
pub trait VectorsLike {
    fn len(&self) -> usize;

    type Item;

    /// Extract the idx-th element under assumption it won't be extracted again
    ///
    /// # Safety
    ///
    /// - This should only be called once per index, i.e. you must ensure
    ///   that the iterator will not visit this index again
    /// - This should only be called for valid indices of the underlying Vectors
    unsafe fn extract_item(&mut self, idx: usize) -> Self::Item;
}
//
impl<V: VectorInfo, Data: VectorizedImpl<V>> VectorsLike for Vectors<V, Data> {
    fn len(&self) -> usize {
        Vectors::len(self)
    }

    type Item = Data::Element;

    unsafe fn extract_item<'iter>(&'iter mut self, idx: usize) -> Self::Item {
        debug_assert!(idx < self.len());
        let result = unsafe { self.get_unchecked_ref(idx) };
        // This is safe because the VectorizedImpl contract says that ElementRef
        // should just be a lower-lifetime version of Element.
        unsafe { core::mem::transmute_copy::<Data::ElementRef<'iter>, Data::Element>(&result) }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> VectorsLike for &'vectors Vectors<V, Data> {
    fn len(&self) -> usize {
        Vectors::len(self)
    }

    type Item = Data::ElementCopy;

    unsafe fn extract_item(&mut self, idx: usize) -> Self::Item {
        debug_assert!(idx < self.len());
        unsafe { self.get_unchecked(idx) }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> VectorsLike
    for &'vectors mut Vectors<V, Data>
{
    fn len(&self) -> usize {
        Vectors::len(self)
    }

    type Item = Data::ElementRef<'vectors>;

    unsafe fn extract_item<'iter>(&'iter mut self, idx: usize) -> Self::Item {
        debug_assert!(idx < self.len());
        let result = unsafe { self.get_unchecked_ref(idx) };
        // This is safe under the extract_item contract
        unsafe {
            core::mem::transmute_copy::<Data::ElementRef<'iter>, Data::ElementRef<'vectors>>(
                &result,
            )
        }
    }
}

/// Iterator over owned, copied or borrowed [`Vectors`] elements
pub struct GenericIter<Vecs: VectorsLike> {
    vectors: Vecs,
    start: usize,
    end: usize,
}
//
impl<Vecs: VectorsLike> GenericIter<Vecs> {
    /// Start iteration
    fn new(vectors: Vecs) -> Self {
        let end = vectors.len();
        Self {
            vectors,
            start: 0,
            end,
        }
    }
}
//
impl<Vecs: VectorsLike> Iterator for GenericIter<Vecs> {
    type Item = Vecs::Item;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            self.start += 1;
            Some(unsafe { self.vectors.extract_item(self.start - 1) })
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
impl<Vecs: VectorsLike> DoubleEndedIterator for GenericIter<Vecs> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            self.end -= 1;
            Some(unsafe { self.vectors.extract_item(self.end + 1) })
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
impl<Vecs: VectorsLike> ExactSizeIterator for GenericIter<Vecs> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.start
    }
}
//
impl<Vecs: VectorsLike> FusedIterator for GenericIter<Vecs> {}
//
#[cfg(feature = "iterator_ilp")]
unsafe impl<Vecs: VectorsLike> iterator_ilp::TrustedLowerBound for GenericIter<Vecs> {}

/// Read-only [`Vectors`] iterator
pub type Iter<'vectors, V, Data> = GenericIter<&'vectors Vectors<V, Data>>;
//
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
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> IntoIterator for &'vectors Vectors<V, Data> {
    type Item = Data::ElementCopy;
    type IntoIter = Iter<'vectors, V, Data>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

/// In-place [`Vectors`] iterator
pub type RefIter<'vectors, V, Data> = GenericIter<&'vectors mut Vectors<V, Data>>;
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
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> IntoIterator
    for &'vectors mut Vectors<V, Data>
{
    type Item = Data::ElementRef<'vectors>;
    type IntoIter = RefIter<'vectors, V, Data>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

/// Consuming [`Vectors`] iterator
pub type IntoIter<V, Data> = GenericIter<Vectors<V, Data>>;
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
//
impl<V: VectorInfo, Data: VectorizedImpl<V>> IntoIterator for Vectors<V, Data> {
    type Item = Data::Element;
    type IntoIter = IntoIter<V, Data>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

// FIXME: Tests
