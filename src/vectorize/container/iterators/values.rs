//! Iteration over values within `Vectorized`

use crate::vectorize::{data::VectorizedDataImpl, RefSlice, Slice, VectorInfo, Vectorized};
use core::iter::FusedIterator;

/// Genericity over Vectorized, &Vectorized and &mut Vectorized
#[doc(hidden)]
pub trait VectorizedLike {
    fn len(&self) -> usize;

    type Item;

    /// Extract the idx-th element under assumption it won't be extracted again
    ///
    /// # Safety
    ///
    /// - This should only be called once per index, i.e. you must ensure
    ///   that the iterator will not visit this index again
    /// - This should only be called for valid indices of the underlying `Vectorized`
    ///   slice
    unsafe fn extract_item(&mut self, idx: usize) -> Self::Item;
}
//
impl<V: VectorInfo, Data: VectorizedDataImpl<V>> VectorizedLike for Vectorized<V, Data> {
    fn len(&self) -> usize {
        Vectorized::len(self)
    }

    type Item = Data::Element;

    unsafe fn extract_item<'iter>(&'iter mut self, idx: usize) -> Self::Item {
        debug_assert!(idx < self.len());
        let result = unsafe { self.get_unchecked_ref(idx) };
        // This is safe because the VectorizedDataImpl contract says that ElementRef
        // should just be a lower-lifetime version of Element.
        unsafe { core::mem::transmute_copy::<Data::ElementRef<'iter>, Data::Element>(&result) }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedDataImpl<V>> VectorizedLike
    for &'vectors Vectorized<V, Data>
{
    fn len(&self) -> usize {
        Vectorized::len(self)
    }

    type Item = Data::ElementCopy;

    unsafe fn extract_item(&mut self, idx: usize) -> Self::Item {
        debug_assert!(idx < self.len());
        unsafe { self.get_unchecked(idx) }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedDataImpl<V>> VectorizedLike
    for &'vectors mut Vectorized<V, Data>
{
    fn len(&self) -> usize {
        Vectorized::len(self)
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

/// Iterator over owned, copied or borrowed [`Vectorized`] elements
pub struct GenericIter<Vecs: VectorizedLike> {
    vectors: Vecs,
    start: usize,
    end: usize,
}
//
impl<Vecs: VectorizedLike> GenericIter<Vecs> {
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
impl<Vecs: VectorizedLike> Iterator for GenericIter<Vecs> {
    type Item = Vecs::Item;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start != self.end {
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
impl<Vecs: VectorizedLike> DoubleEndedIterator for GenericIter<Vecs> {
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start != self.end {
            self.end -= 1;
            Some(unsafe { self.vectors.extract_item(self.end) })
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
impl<Vecs: VectorizedLike> ExactSizeIterator for GenericIter<Vecs> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.start
    }
}
//
impl<Vecs: VectorizedLike> FusedIterator for GenericIter<Vecs> {}
//
#[cfg(feature = "iterator_ilp")]
unsafe impl<Vecs: VectorizedLike> iterator_ilp::TrustedLowerBound for GenericIter<Vecs> {}

/// Read-only [`Vectorized`] iterator
pub type Iter<'vectors, V, Data> = GenericIter<&'vectors Vectorized<V, Data>>;
//
impl<'vectors, V: VectorInfo, Data: VectorizedDataImpl<V>> Iter<'vectors, V, Data> {
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
impl<'vectors, V: VectorInfo, Data: VectorizedDataImpl<V>> IntoIterator
    for &'vectors Vectorized<V, Data>
{
    type Item = Data::ElementCopy;
    type IntoIter = Iter<'vectors, V, Data>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

/// In-place [`Vectorized`] iterator
pub type RefIter<'vectors, V, Data> = GenericIter<&'vectors mut Vectorized<V, Data>>;
//
impl<'vectors, V: VectorInfo, Data: VectorizedDataImpl<V>> RefIter<'vectors, V, Data> {
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
impl<'vectors, V: VectorInfo, Data: VectorizedDataImpl<V>> IntoIterator
    for &'vectors mut Vectorized<V, Data>
{
    type Item = Data::ElementRef<'vectors>;
    type IntoIter = RefIter<'vectors, V, Data>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

/// Consuming [`Vectorized`] iterator
pub type IntoIter<V, Data> = GenericIter<Vectorized<V, Data>>;
//
impl<V: VectorInfo, Data: VectorizedDataImpl<V>> IntoIter<V, Data> {
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
impl<V: VectorInfo, Data: VectorizedDataImpl<V>> IntoIterator for Vectorized<V, Data> {
    type Item = Data::Element;
    type IntoIter = IntoIter<V, Data>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

// FIXME: Tests
