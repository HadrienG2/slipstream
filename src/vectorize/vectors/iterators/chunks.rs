//! Iterator over chunks of `Vectors`' inner data

use crate::vectorize::{data::VectorizedImpl, RefSlice, VectorInfo};
use core::{hint::unreachable_unchecked, iter::FusedIterator, num::NonZeroUsize};

// FIXME: Implement non-Ref versions of these iterators, fix comments, fix ChunksExact impl

/// An iterator over [`Vectors`] in (non-overlapping) in-place chunks
/// (`chunk_size` elements at a time), starting at the beginning of the dataset
///
/// When the dataset length is not evenly divided by the chunk size, the last
/// slice of the iteration will be the remainder.
///
/// This struct is created by [`Vectors::chunks_mut()`].
pub struct RefChunks<'vectors, V: VectorInfo, Data: VectorizedImpl<V> + 'vectors> {
    remainder: Option<RefSlice<'vectors, V, Data>>,
    chunk_size: NonZeroUsize,
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> RefChunks<'vectors, V, Data> {
    /// Set up a chunks iterator
    #[inline]
    pub(crate) fn new(vectors: RefSlice<'vectors, V, Data>, chunk_size: NonZeroUsize) -> Self {
        Self {
            remainder: Some(vectors),
            chunk_size,
        }
    }

    /// Chunk size (guaranteed not to be zero)
    #[inline(always)]
    fn chunk_size(&self) -> usize {
        self.chunk_size.into()
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> Iterator for RefChunks<'vectors, V, Data> {
    type Item = RefSlice<'vectors, V, Data>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let remainder = self.remainder.take()?;
        let chunk = if remainder.len() > self.chunk_size() {
            let (chunk, remainder) = unsafe { remainder.split_at_unchecked(self.chunk_size()) };
            self.remainder = Some(remainder);
            chunk
        } else {
            remainder
        };
        Some(chunk)
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
            if n > 0 {
                let mut remainder = self.remainder.take()?;
                remainder = unsafe { remainder.split_at_unchecked((n - 1) * self.chunk_size()) }.1;
                self.remainder = Some(remainder);
            }
            self.next()
        } else {
            self.remainder = None;
            None
        }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> DoubleEndedIterator
    for RefChunks<'vectors, V, Data>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        let remainder = self.remainder.take()?;
        let chunk = if remainder.len() > self.chunk_size() {
            let last_chunk_len = NonZeroUsize::new(remainder.len() % self.chunk_size())
                .map(usize::from)
                .unwrap_or(self.chunk_size());
            let remainder_len = remainder.len();
            let (remainder, chunk) =
                unsafe { remainder.split_at_unchecked(remainder_len - last_chunk_len) };
            self.remainder = Some(remainder);
            chunk
        } else {
            remainder
        };
        Some(chunk)
    }

    #[inline]
    fn nth_back(&mut self, mut n: usize) -> Option<Self::Item> {
        // Handle out-of-bounds n
        if self.len() < n {
            self.remainder = None;
            return None;
        }

        // Handle oddly sized last chunk
        let last_chunk = self.next_back()?;
        if n == 0 {
            return Some(last_chunk);
        }
        n -= 1;

        // Skip extra regular chunks in the back we're not interested in
        if n > 0 {
            let mut remainder = self.remainder.take()?;
            let remainder_len = remainder.len();
            remainder = unsafe {
                remainder.split_at_unchecked(remainder_len - (n - 1) * self.chunk_size())
            }
            .0;
            self.remainder = Some(remainder);
        }

        // The chunk we want is now in the back
        self.next_back()
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> ExactSizeIterator
    for RefChunks<'vectors, V, Data>
{
    #[inline]
    fn len(&self) -> usize {
        if let Some(remainder) = self.remainder.as_ref() {
            (remainder.len() / self.chunk_size) + (remainder.len() % self.chunk_size != 0) as usize
        } else {
            0
        }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> FusedIterator
    for RefChunks<'vectors, V, Data>
{
}
//
#[cfg(feature = "iterator_ilp")]
unsafe impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> iterator_ilp::TrustedLowerBound
    for RefChunks<'vectors, V, Data>
{
}

/// An iterator over [`Vectors`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the beginning of the dataset
///
/// When the dataset length is not evenly divided by the chunk size, the last
/// up to `chunk_size-1` elements will be omitted but can be retrieved from the
/// iterator using the `into_remainder()` function.
///
/// This struct is created by [`Vectors::chunks_exact_mut()`].
pub struct RefChunksExact<'vectors, V: VectorInfo, Data: VectorizedImpl<V> + 'vectors> {
    regular: RefChunks<'vectors, V, Data>,
    remainder: RefSlice<'vectors, V, Data>,
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> RefChunksExact<'vectors, V, Data> {
    /// Set up an exact-size chunks iterator
    #[inline]
    pub(crate) fn new(vectors: RefSlice<'vectors, V, Data>, chunk_size: NonZeroUsize) -> Self {
        let total_len = vectors.len();
        let remainder_len = total_len % chunk_size;
        let (regular_vectors, remainder) = if remainder_len > 0 {
            unsafe { vectors.split_at_unchecked(total_len - remainder_len) }
        } else {
            (vectors, RefSlice::<'vectors, V, Data>::empty())
        };
        Self {
            regular: RefChunks::new(regular_vectors, chunk_size),
            remainder,
        }
    }

    /// Returns the remainder of the original dataset that is not going to be
    /// returned by the iterator. The returned slice has at most `chunk_size-1`
    /// elements.
    #[inline]
    pub fn into_remainder(self) -> RefSlice<'vectors, V, Data> {
        self.remainder
    }

    /// Assert that any chunk returned by this iterator has the expected size
    ///
    /// # Safety
    ///
    /// The chunk must originate from `self.regular`, without further tampering
    #[inline(always)]
    unsafe fn assume_exact(
        &self,
        item: Option<<Self as Iterator>::Item>,
    ) -> Option<<Self as Iterator>::Item> {
        match item {
            Some(item) if item.len() == self.regular.chunk_size() => Some(item),
            None => None,
            _ => unsafe { unreachable_unchecked() },
        }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> Iterator
    for RefChunksExact<'vectors, V, Data>
{
    type Item = RefSlice<'vectors, V, Data>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.regular.next();
        unsafe { self.assume_exact(result) }
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
        let result = self.regular.nth(n);
        unsafe { self.assume_exact(result) }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> DoubleEndedIterator
    for RefChunksExact<'vectors, V, Data>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        let result = self.regular.next_back();
        unsafe { self.assume_exact(result) }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let result = self.regular.nth_back(n);
        unsafe { self.assume_exact(result) }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> ExactSizeIterator
    for RefChunksExact<'vectors, V, Data>
{
    #[inline]
    fn len(&self) -> usize {
        if let Some(remainder) = self.regular.remainder.as_ref() {
            remainder.len() / self.regular.chunk_size()
        } else {
            0
        }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> FusedIterator
    for RefChunksExact<'vectors, V, Data>
{
}
//
#[cfg(feature = "iterator_ilp")]
unsafe impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> iterator_ilp::TrustedLowerBound
    for RefChunksExact<'vectors, V, Data>
{
}

// TODO: Windows

// FIXME: Tests
