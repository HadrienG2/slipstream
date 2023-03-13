//! Conversion from vectorizable data to `Vectorized`
//!
//! Using the `Vectorizable` trait provided by this module, users can turn
//! collections of SIMD vectors and scalar data into a `Vectorized` view. If the
//! scalar data exhibits some good properties from a SIMD perspective, they can
//! be asserted as part of the conversion process, in order to reduce code bloat
//! and achieve better runtime performance.

use super::{
    data::{AlignedData, AlignedDataMut, PaddedData, PaddedDataMut, VectorizedDataImpl},
    VectorInfo, Vectorized, VectorizedAligned, VectorizedData, VectorizedPadded,
    VectorizedUnaligned,
};
use crate::{inner::Repr, vector::align::Align, Vector};

/// Data that can be processed using SIMD
///
/// Implemented for slices and containers of vectors and scalars,
/// as well as for tuples of these entities.
///
/// Provides you with ways to create the [`Vectorized`] collection, which
/// behaves conceptually like a slice of [`Vector`]s or tuples thereof, with
/// iteration and indexing operations yielding the following types:
///
/// - If built out of a read-only slice or owned container of vectors or
///   scalars, it yields owned [`Vector`]s of data.
/// - If built out of `&mut [Vector]`, or `&mut [Scalar]` that is assumed
///   to be SIMD-aligned (see below), it yields `&mut Vector` references.
/// - If built out of `&mut [Scalar]` that is not assumed to SIMD-aligned, it
///   yields a proxy type which can be used like an `&mut Vector` (but cannot
///   literally be `&mut Vector` for alignment and padding reasons)
/// - If built out of a tuple of the above entities, it yields tuples of the
///   aforementioned elements.
///
/// There are three ways to create [`Vectorized`] using this trait depending on
/// what kind of data you're starting from:
///
/// - If starting out of arbitrary data, you can use the [`vectorize_pad()`]
///   method to get a SIMD view that does not make any assumption, but
///   tends to exhibit poor performance on scalar data as a result.
/// - If you know that every scalar slice in your dataset has a number of
///   elements that is a multiple of the SIMD vector width, you can use the
///   [`vectorize()`] method to get a SIMD view that assumes this (or a
///   panic if this is not true), resulting in much better performance.
/// - If, in addition to the above, you know that every scalar slice in your
///   dataset is SIMD-aligned, you can use the [`vectorize_aligned()`]
///   method to get a SIMD view that assumes this (or a panic if this is not
///   true), which may result in even better performance.
///
/// Note that even on hardware architectures like x86 where SIMD alignment
/// is not a prerequisite for good code generation (and hence you may not
/// need to call [`vectorize_aligned()`] for optimal performance), it is
/// always a hardware prerequisite for good computational performance, so
/// you should aim for it whenever possible!
///
/// # Safety
///
/// Unsafe code may rely on the implementation being correct.
///
/// [`vectorize()`]: Vectorizable::vectorize()
/// [`vectorize_aligned()`]: Vectorizable::vectorize_aligned()
/// [`vectorize_pad()`]: Vectorizable::vectorize_pad()
pub unsafe trait Vectorizable<V: VectorInfo>: Sized {
    /// Vectorized representation of this data
    ///
    /// You can use the [`VectorizedData`] trait to query at compile time which
    /// type of Vectorized collections you are going to get and what kind of
    /// elements iterators and getters of this collection will emit.
    ///
    /// `VectorizedDataImpl` is an implementation detail of this crate.
    type VectorizedData: VectorizedData<V> + VectorizedDataImpl<V>;

    // Required methods

    /// Implementation of the vectorization methods
    //
    // --- Internal docs starts here ---
    //
    // The returned building blocks are...
    //
    // - A pointer-like entity for treating the data as a slice of Vector
    //   (see [`VectorizedDataImpl`] for more information)
    // - The number of Vector elements that the emulated slice contains
    //
    // # Errors
    //
    // - [`NeedsPadding`] if padding was needed, but not provided
    // - [`InhomogeneousLength`] if input is (or contains) a tuple and not all
    //   tuple elements yield the same amount of SIMD vectors
    //
    // [`NeedsPadding`]: VectorizeError::NeedsPadding
    // [`InhomogeneousLength`]: VectorizeError::InhomogeneousLength
    fn into_vectorized_parts(
        self,
        padding: Option<V::Scalar>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError>;

    // Provided methods

    /// Create a SIMD view of this data, asserting that its length is a
    /// multiple of the SIMD vector length
    ///
    /// # Panics
    ///
    /// - If called on a scalar slice whose length is not a multiple of the
    ///   number of SIMD vector lanes (consider using [`vectorize_pad()`] if
    ///   you cannot avoid this)
    /// - If called on a tuple and not all tuple elements yield the same
    ///   amount of SIMD elements.
    ///
    /// [`vectorize_pad()`]: Vectorizable::vectorize_pad()
    fn vectorize(self) -> VectorizedUnaligned<V, Self::VectorizedData> {
        let (base, len) = self.into_vectorized_parts(None).unwrap();
        unsafe { Vectorized::from_raw_parts(base.into_unaligned_unchecked(), len) }
    }

    /// Create a SIMD view of this data, providing some padding
    ///
    /// Vector slices do not need padding and will ignore it.
    ///
    /// For scalar slices whose size is not a multiple of the number of SIMD
    /// vector lanes, padding will be inserted where incomplete `Vector`s
    /// would be produced, to fill in the missing vector lanes. One would
    /// normally set the padding to the neutral element of the computation
    /// being performed so that its presence doesn't affect results.
    ///
    /// The use of padding makes it harder for the compiler to optimize the
    /// code even if the padding ends up not being used, so using this
    /// option will generally result in lower runtime performance on scalar data.
    ///
    /// # Panics
    ///
    /// - If called on a tuple and not all tuple elements yield the same
    ///   amount of SIMD elements.
    fn vectorize_pad(self, padding: V::Scalar) -> VectorizedPadded<V, Self::VectorizedData> {
        let (base, len) = self.into_vectorized_parts(Some(padding)).unwrap();
        unsafe { Vectorized::from_raw_parts(base, len) }
    }

    /// Create a SIMD view of this data, assert it has a memory layout optimized
    /// for SIMD processing
    ///
    /// Vector data always meets this requirement, and indeed, you do not need
    /// to call `vectorize_aligned()` to get the associated performance boost, it
    /// will be present even if you only call `vectorize()` or `vectorize_pad()`.
    ///
    /// But scalar data only passes this check if it meets two conditions:
    ///
    /// - The start of the data slice is aligned as
    ///   [`core::mem::align_of::<V>()`](core::mem::align_of).
    /// - The number of inner scalar elements is a multiple of the number
    ///   of SIMD vector lanes of V.
    ///
    /// If this is true, that data is reinterpreted in a manner that will
    /// simplify the implementation, which should result in less edge cases
    /// where the compiler does not generate good code.
    ///
    /// Furthermore, note that the above are actually _hardware_
    /// requirements for good vectorization: even if the compiler does
    /// generate good code with `vectorize()` already, the resulting binary will
    /// perform less well if the data does not have the above properties. So you
    /// should enforce them whenever possible!
    ///
    /// If you need to process a single slice of scalar data that does not meet
    /// these requirements, note that you can extract an aligned subset that
    /// meets them using [`[Scalar]::align_to::<V>()`](slice::align_to()) or
    /// [`[Scalar]::align_to_mut::<V>()`](slice::align_to_mut), as an
    /// alternative to using `Vectorizable`. Beware that this will not help if
    /// you need to jointly process multiple slices of scalar data, which might
    /// be misaligned by a different number of elements (and thus the output
    /// aligned slices of V may not line up).
    ///
    /// # Panics
    ///
    /// - If the data is not in a SIMD-optimized layout.
    /// - If called on a tuple and not all tuple elements yield the same
    ///   amount of SIMD elements.
    fn vectorize_aligned(self) -> VectorizedAligned<V, Self::VectorizedData> {
        let (base, len) = self.into_vectorized_parts(None).unwrap();
        unsafe { Vectorized::from_raw_parts(base.into_aligned_unchecked(), len) }
    }
}

/// Error returned by [`Vectorizable::into_vectorized_parts()`]
#[doc(hidden)]
#[derive(Debug)]
pub enum VectorizeError {
    /// Padding data was needed, but not provided
    NeedsPadding,

    /// Input contains tuples of data with inhomogeneous SIMD length
    InhomogeneousLength,
}

// === Vectorizable implementation is trivial for slices of vector data ===

unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'target [Vector<A, B, S>]
{
    type VectorizedData = AlignedData<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        _padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        Ok((self.into(), self.len()))
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'target mut [Vector<A, B, S>]
{
    type VectorizedData = AlignedDataMut<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        _padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        let len = self.len();
        Ok((self.into(), len))
    }
}

unsafe impl<A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for [Vector<A, B, S>; ARRAY_SIZE]
{
    type VectorizedData = [Vector<A, B, S>; ARRAY_SIZE];

    fn into_vectorized_parts(
        self,
        _padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        Ok((self, ARRAY_SIZE))
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for &'target [Vector<A, B, S>; ARRAY_SIZE]
{
    type VectorizedData = AlignedData<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        self.as_slice().into_vectorized_parts(padding)
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for &'target mut [Vector<A, B, S>; ARRAY_SIZE]
{
    type VectorizedData = AlignedDataMut<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        self.as_mut_slice().into_vectorized_parts(padding)
    }
}

// === For scalar data, must cautiously handle padding and alignment ===

unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'target [B]
{
    type VectorizedData = PaddedData<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        Ok((
            PaddedData::new(self, padding)?.0,
            self.len() / S + (self.len() % S != 0) as usize,
        ))
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'target mut [B]
{
    type VectorizedData = PaddedDataMut<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        let simd_len = self.len() / S + (self.len() % S != 0) as usize;
        Ok((PaddedDataMut::new(self, padding)?, simd_len))
    }
}

// NOTE: Cannot be implemented for owned scalar arrays yet due to const
//       generics limitations around vectorized_aligned().

unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for &'target [B; ARRAY_SIZE]
{
    type VectorizedData = PaddedData<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        self.as_slice().into_vectorized_parts(padding)
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for &'target mut [B; ARRAY_SIZE]
{
    type VectorizedData = PaddedDataMut<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
        self.as_mut_slice().into_vectorized_parts(padding)
    }
}

// === Tuples must have homogeneous length and vector type ===

macro_rules! impl_vectorizable_for_tuple {
    (
        $($t:ident),*
    ) => {
        #[allow(non_snake_case)]
        unsafe impl<V: VectorInfo $(, $t: Vectorizable<V>)*> Vectorizable<V> for ($($t,)*) {
            type VectorizedData = ($($t::VectorizedData,)*);

            fn into_vectorized_parts(
                self,
                padding: Option<V::Scalar>,
            ) -> Result<(Self::VectorizedData, usize), VectorizeError> {
                // Pattern-match the tuple to variables named after inner types
                let ($($t,)*) = self;

                // Reinterpret tuple fields as SIMD vectors
                let ($($t,)*) = ($($t.into_vectorized_parts(padding)?,)*);

                // Analyze tuple field lengths and need for padding
                let mut len = None;
                $(
                    let (_, t_len) = $t;

                    // All tuple fields should have the same SIMD length
                    #[allow(unused_assignments)]
                    if let Some(len) = len {
                        if len != t_len {
                            return Err(VectorizeError::InhomogeneousLength);
                        }
                    } else {
                        len = Some(t_len);
                    }
                )*

                // All good, return Vectorized building blocks
                Ok((
                    ($($t.0,)*),
                    len.expect("This should not be implemented for zero-sized tuples"),
                ))
            }
        }
    };
}
impl_vectorizable_for_tuple!(A);
impl_vectorizable_for_tuple!(A, B);
impl_vectorizable_for_tuple!(A, B, C);
impl_vectorizable_for_tuple!(A, B, C, D);
impl_vectorizable_for_tuple!(A, B, C, D, E);
impl_vectorizable_for_tuple!(A, B, C, D, E, F);
impl_vectorizable_for_tuple!(A, B, C, D, E, F, G);
impl_vectorizable_for_tuple!(A, B, C, D, E, F, G, H);

// FIXME: Tests
