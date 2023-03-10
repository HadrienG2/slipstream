//! Iterator over chunks of `Vectors`' inner data

use crate::vectorize::{data::VectorizedSliceImpl, VectorInfo, Vectorized, Vectors};
use core::{iter::FusedIterator, num::NonZeroUsize};

// === VARIABLE-SIZE CHUNKS ===

/// Common generic implementation behind Chunks and RefChunks
pub struct GenericChunks<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> {
    /// Remainder of the initial slice of output, or None if iteration is over
    remainder: Option<Vectors<V, SliceData>>,

    /// Size of the chunks that we're splitting
    chunk_size: NonZeroUsize,
}
//
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> GenericChunks<V, SliceData> {
    /// Set up a chunks iterator
    #[inline]
    pub(crate) fn new(slice: Vectors<V, SliceData>, chunk_size: NonZeroUsize) -> Self {
        Self {
            remainder: Some(slice),
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
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> Iterator for GenericChunks<V, SliceData> {
    type Item = Vectors<V, SliceData>;

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
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> DoubleEndedIterator
    for GenericChunks<V, SliceData>
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
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> ExactSizeIterator
    for GenericChunks<V, SliceData>
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
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> FusedIterator
    for GenericChunks<V, SliceData>
{
}
//
#[cfg(feature = "iterator_ilp")]
unsafe impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> iterator_ilp::TrustedLowerBound
    for GenericChunks<V, SliceData>
{
}

/// An iterator over [`Vectors`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the beginning of the dataset
///
/// When the dataset length is not evenly divided by the chunk size, the last
/// slice of the iteration will be the remainder.
///
/// This struct is created by [`Vectors::chunks()`].
pub type Chunks<'vectors, V, Data> = GenericChunks<V, <Data as Vectorized<V>>::CopySlice<'vectors>>;

/// An iterator over [`Vectors`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the beginning of the dataset and providing
/// mutable access to mutable elements of the dataset.
///
/// When the dataset length is not evenly divided by the chunk size, the last
/// slice of the iteration will be the remainder.
///
/// This struct is created by [`Vectors::chunks_ref()`].
pub type RefChunks<'vectors, V, Data> =
    GenericChunks<V, <Data as Vectorized<V>>::RefSlice<'vectors>>;

// === EXACT-SIZE CHUNKS ===

// FIXME: Implement non-Ref versions of these iterators, fix comments, fix ChunksExact impl

/// Common generic implementation behind ChunksExact and RefChunksExact
pub struct GenericChunksExact<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> {
    regular: GenericChunks<V, SliceData>,
    remainder: Vectors<V, SliceData>,
}
//
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> GenericChunksExact<V, SliceData> {
    /// Set up an exact-size chunks iterator
    #[inline]
    pub(crate) fn new(vectors: Vectors<V, SliceData>, chunk_size: NonZeroUsize) -> Self {
        let total_len = vectors.len();
        let remainder_len = total_len % chunk_size;
        let (regular_vectors, remainder) = if remainder_len > 0 {
            unsafe { vectors.split_at_unchecked(total_len - remainder_len) }
        } else {
            (vectors, Vectors::empty())
        };
        Self {
            regular: GenericChunks::new(regular_vectors, chunk_size),
            remainder,
        }
    }

    /// Read-only view of the remainder of the original dataset that is not
    /// going to be returned by the iterator. The returned slice has at most
    /// `chunk_size-1` elements.
    #[inline]
    pub fn remainder(&self) -> &Vectors<V, SliceData> {
        &self.remainder
    }

    /// Reemainder of the original dataset that is not going to be returned by
    /// the iterator. The returned slice has at most `chunk_size-1` elements.
    #[inline]
    pub fn into_remainder(self) -> Vectors<V, SliceData> {
        self.remainder
    }

    /// Assert that the underlying iterator emits slices of length chunk_size
    #[inline(always)]
    unsafe fn assume_regular(
        &self,
        item: Option<Vectors<V, SliceData>>,
    ) -> Option<Vectors<V, SliceData>> {
        Some(unsafe { item?.split_at_unchecked(self.regular.chunk_size()).0 })
    }
}
//
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> Iterator
    for GenericChunksExact<V, SliceData>
{
    type Item = Vectors<V, SliceData>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.regular.next();
        unsafe { self.assume_regular(result) }
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
        unsafe { self.assume_regular(result) }
    }
}
//
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> DoubleEndedIterator
    for GenericChunksExact<V, SliceData>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<Self::Item> {
        let result = self.regular.next_back();
        unsafe { self.assume_regular(result) }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let result = self.regular.nth_back(n);
        unsafe { self.assume_regular(result) }
    }
}
//
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> ExactSizeIterator
    for GenericChunksExact<V, SliceData>
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
impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> FusedIterator
    for GenericChunksExact<V, SliceData>
{
}
//
#[cfg(feature = "iterator_ilp")]
unsafe impl<V: VectorInfo, SliceData: VectorizedSliceImpl<V>> iterator_ilp::TrustedLowerBound
    for GenericChunksExact<V, SliceData>
{
}

/// An iterator over [`Vectors`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the beginning of the dataset
///
/// When the slice len is not evenly divided by the chunk size, the last up to
/// `chunk_size-1` elements will be omitted but can be retrieved from the
/// `remainder()` function from the iterator.
///
/// This struct is created by [`Vectors::chunks_exact()`].
pub type ChunksExact<'vectors, V, Data> =
    GenericChunksExact<V, <Data as Vectorized<V>>::CopySlice<'vectors>>;

/// An iterator over [`Vectors`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the beginning of the dataset and providing
/// mutable access to mutable elements of the dataset.
///
/// When the slice len is not evenly divided by the chunk size, the last up to
/// `chunk_size-1` elements will be omitted but can be retrieved from the
/// `into_remainder()` function from the iterator.
///
/// This struct is created by [`Vectors::chunks_exact_ref()`].
pub type RefChunksExact<'vectors, V, Data> =
    GenericChunksExact<V, <Data as Vectorized<V>>::RefSlice<'vectors>>;

// TODO: Windows

// FIXME: Tests
