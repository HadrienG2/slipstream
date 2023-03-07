//! Conversion from vectorizable data to `Vectors`
//!
//! Using the `Vectorizable` trait provided by this module, users can turn
//! collections of SIMD vectors and scalar data into a `Vectors` view. If the
//! scalar data exhibits some good properties from a SIMD perspective, they can
//! be asserted as part of the conversion process, in order to reduce code bloat
//! and achieve better runtime performance.

use super::{
    data::{AlignedData, AlignedDataMut, PaddedData, PaddedDataMut, VectorizedImpl},
    AlignedVectors, PaddedVectors, UnalignedVectors, VectorInfo, Vectorized, Vectors,
};
use crate::{inner::Repr, vector::align::Align, Vector};

/// Trait for data that can be processed using SIMD
///
/// Implemented for slices and containers of vectors and scalars,
/// as well as for tuples of these entities.
///
/// Provides you with ways to create the `Vectors` collection, which
/// behaves conceptually like a slice of `Vector` or tuples thereof, with
/// iteration and indexing operations yielding the following types:
///
/// - If built out of a read-only slice or owned container of vectors or
///   scalars, it yields owned `Vector`s of data.
/// - If built out of `&mut [Vector]`, or `&mut [Scalar]` that is assumed
///   to be SIMD-aligned (see below), it yields `&mut Vector` references.
/// - If built out of `&mut [Scalar]` that is not SIMD-aligned, it yields
///   a proxy type which can be used like an `&mut Vector` (but cannot
///   literally be `&mut Vector` for alignment and padding reasons)
/// - If built out of a tuple of the above entities, it yields tuples of the
///   aforementioned elements.
///
/// There are three ways to create `Vectors` using this trait depending on
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
/// need to call `vectorize_aligned()` for optimal performance), it is
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
    /// You can use the Vectorized trait to query at compile time which type
    /// of Vectors collections you are going to get and what kind of
    /// elements iterators and getters of this collection will emit.
    ///
    /// VectorizedImpl is an implementation detail of this crate.
    type Vectorized: Vectorized<V> + VectorizedImpl<V>;

    // Required methods

    /// Implementation of the `vectorize()` methods
    //
    // --- Internal docs starts here ---
    //
    // The returned building blocks are...
    //
    // - A pointer-like entity for treating the data as a slice of Vector
    //   (see VectorizedImpl for more information)
    // - The number of Vector elements that the emulated slice contains
    //
    // # Errors
    //
    // - NeedsPadding if padding was needed, but not provided
    // - InhomogeneousLength if input is (or contains) a tuple and not all
    //   tuple elements yield the same amount of SIMD vectors
    fn into_vectorized_parts(
        self,
        padding: Option<V::Scalar>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError>;

    // Provided methods

    /// Create a SIMD view of this data, asserting that its length is a
    /// multiple of the SIMD vector length
    ///
    /// # Panics
    ///
    /// - If called on a scalar slice whose length is not a multiple of the
    ///   number of SIMD vector lanes (you need `vectorize_pad()`)
    /// - If called on a tuple and not all tuple elements yield the same
    ///   amount of SIMD elements.
    fn vectorize(self) -> UnalignedVectors<V, Self::Vectorized> {
        let (base, len) = self.into_vectorized_parts(None).unwrap();
        unsafe { Vectors::from_raw_parts(base.as_unaligned_unchecked(), len) }
    }

    /// Create a SIMD view of this data, providing some padding
    ///
    /// Vector slices do not need padding and will ignore it.
    ///
    /// For scalar slices whose size is not a multiple of the number of SIMD
    /// vector lanes, padding will be inserted where incomplete Vectors
    /// would be produced, to fill in the missing vector lanes. One would
    /// normally set the padding to the neutral element of the computation
    /// being performed so that its presence doesn't affect results.
    ///
    /// The use of padding makes it harder for the compiler to optimize the
    /// code even if the padding ends up not being used, so using this
    /// option will generally result in lower runtime performance.
    ///
    /// # Panics
    ///
    /// - If called on a tuple and not all tuple elements yield the same
    ///   amount of SIMD elements.
    fn vectorize_pad(self, padding: V::Scalar) -> PaddedVectors<V, Self::Vectorized> {
        let (base, len) = self.into_vectorized_parts(Some(padding)).unwrap();
        unsafe { Vectors::from_raw_parts(base, len) }
    }

    /// Create a SIMD view of this data, assert it is (or can be moved to) a
    /// layout optimized for SIMD processing
    ///
    /// Vector data always passes this check, but scalar data only passes it
    /// if it meets two conditions:
    ///
    /// - The start of the data is aligned as `std::mem::align_of::<V>()`,
    ///   or the data can be moved around in memory to enforce this.
    /// - The number of inner scalar elements is a multiple of the number
    ///   of SIMD vector lanes of V.
    ///
    /// If this is true, the data is reinterpreted in a manner that will
    /// simplify the implementation, which should result in less edge cases
    /// where the compiler does not generate good code.
    ///
    /// Furthermore, note that the above are actually _hardware_
    /// requirements for good vectorization: even if the compiler does
    /// generate good code, the resulting binary will perform less well if
    /// the data does not have the above properties. So you should enforce
    /// them whenever possible!
    ///
    /// # Panics
    ///
    /// - If the data is not in a SIMD-optimized layout.
    /// - If called on a tuple and not all tuple elements yield the same
    ///   amount of SIMD elements.
    fn vectorize_aligned(self) -> AlignedVectors<V, Self::Vectorized> {
        let (base, len) = self.into_vectorized_parts(None).unwrap();
        unsafe { Vectors::from_raw_parts(base.as_aligned_unchecked(), len) }
    }
}

/// Error returned by `Vectorizable::into_vectorized_parts`
#[doc(hidden)]
#[derive(Debug)]
pub enum VectorizeError {
    /// Padding data was needed, but not provided
    NeedsPadding,

    /// Input contains tuples of data with inhomogeneous SIMD length
    InhomogeneousLength,
}

// === Vectorize implementation is trivial for slices of vector data ===

unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'target [Vector<A, B, S>]
{
    type Vectorized = AlignedData<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        _padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
        Ok((self.into(), self.len()))
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'target mut [Vector<A, B, S>]
{
    type Vectorized = AlignedDataMut<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        _padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
        let len = self.len();
        Ok((self.into(), len))
    }
}

unsafe impl<A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for [Vector<A, B, S>; ARRAY_SIZE]
{
    type Vectorized = [Vector<A, B, S>; ARRAY_SIZE];

    fn into_vectorized_parts(
        self,
        _padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
        Ok((self, ARRAY_SIZE))
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for &'target [Vector<A, B, S>; ARRAY_SIZE]
{
    type Vectorized = AlignedData<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
        self.as_slice().into_vectorized_parts(padding)
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for &'target mut [Vector<A, B, S>; ARRAY_SIZE]
{
    type Vectorized = AlignedDataMut<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
        self.as_mut_slice().into_vectorized_parts(padding)
    }
}

// === For scalar data, must cautiously handle padding and alignment ===

unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'target [B]
{
    type Vectorized = PaddedData<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
        Ok((
            PaddedData::new(self, padding)?.0,
            self.len() / S + (self.len() % S != 0) as usize,
        ))
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize> Vectorizable<Vector<A, B, S>>
    for &'target mut [B]
{
    type Vectorized = PaddedDataMut<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
        let simd_len = self.len() / S + (self.len() % S != 0) as usize;
        Ok((PaddedDataMut::new(self, padding)?, simd_len))
    }
}

// NOTE: Cannot be implemented for owned scalar arrays yet due to const
//       generics limitations around vectorized_aligned().

unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for &'target [B; ARRAY_SIZE]
{
    type Vectorized = PaddedData<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
        self.as_slice().into_vectorized_parts(padding)
    }
}

unsafe impl<'target, A: Align, B: Repr, const S: usize, const ARRAY_SIZE: usize>
    Vectorizable<Vector<A, B, S>> for &'target mut [B; ARRAY_SIZE]
{
    type Vectorized = PaddedDataMut<'target, Vector<A, B, S>>;

    fn into_vectorized_parts(
        self,
        padding: Option<B>,
    ) -> Result<(Self::Vectorized, usize), VectorizeError> {
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
            type Vectorized = ($($t::Vectorized,)*);

            fn into_vectorized_parts(
                self,
                padding: Option<V::Scalar>,
            ) -> Result<(Self::Vectorized, usize), VectorizeError> {
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
                        assert_eq!(
                            t_len, len,
                            "Tuple elements do not produce the same amount of SIMD vectors"
                        );
                    } else {
                        len = Some(t_len);
                    }
                )*

                // All good, return Vectors building blocks
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
