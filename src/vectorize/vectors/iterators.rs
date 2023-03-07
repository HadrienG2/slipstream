//! Iteration over `Vectors`

use crate::vectorize::{data::VectorizedImpl, Slice, VectorInfo, Vectors};
use core::{iter::FusedIterator, num::NonZeroUsize};

// IntoIterator impls for Vectors and &mut Vectors
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
                /// To avoid creating `&mut V` references that alias, this
                /// is forced to consume the iterator
                #[inline]
                pub fn into_slice(self) -> Slice<$lifetime, V, Data> {
                    unsafe { self.vectors.get_unchecked(self.start..self.end) }
                }
            )?

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
        impl<$($lifetime,)? V: VectorInfo, Data: VectorizedImpl<V>> Iterator
            for $name<$($lifetime,)? V, Data>
        {
            type Item = Data::$elem$(<$lifetime>)?;

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
    /// Borrowing iterator over [`Vectors`]' elements
    (Iter, Data::ElementRef<'vectors>)
);
impl_iterator!(
    /// Consuming iterator of [`Vectors`]' elements
    (IntoIter, Data::Element)
);

/// An iterator over [`Vectors`] in (non-overlapping) chunks (`chunk_size`
/// elements at a time), starting at the beginning of the dataset
///
/// When the dataset length is not evenly divided by the chunk size, the last
/// slice of the iteration will be the remainder.
///
/// This struct is created by [`Vectors::chunks()`]
pub struct Chunks<'vectors, V: VectorInfo, Data: VectorizedImpl<V> + 'vectors> {
    remainder: Option<Slice<'vectors, V, Data>>,
    chunk_size: NonZeroUsize,
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> Chunks<'vectors, V, Data> {
    /// Set up a chunks iterator
    #[inline]
    pub(crate) fn new(vectors: Slice<'vectors, V, Data>, chunk_size: NonZeroUsize) -> Self {
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
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> Iterator for Chunks<'vectors, V, Data> {
    type Item = Slice<'vectors, V, Data>;

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
    for Chunks<'vectors, V, Data>
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
    for Chunks<'vectors, V, Data>
{
    #[inline]
    fn len(&self) -> usize {
        if let Some(remainder) = self.remainder.as_ref() {
            (remainder.len() / self.chunk_size) + (remainder.len() % self.chunk_size == 0) as usize
        } else {
            0
        }
    }
}
//
impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> FusedIterator for Chunks<'vectors, V, Data> {}
//
#[cfg(feature = "iterator_ilp")]
unsafe impl<'vectors, V: VectorInfo, Data: VectorizedImpl<V>> iterator_ilp::TrustedLowerBound
    for Chunks<'vectors, V, Data>
{
}

// FIXME: Tests