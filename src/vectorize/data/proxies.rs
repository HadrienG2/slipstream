//! Proxy objects that mimick `&mut Vector` in cases where we cannot construct
//! a true mutable reference for alignment and data length reasons

use crate::vectorize::VectorInfo;
#[cfg(doc)]
use crate::{
    vectorize::{Vectorizable, Vectors},
    Vector,
};
use core::{
    borrow::{Borrow, BorrowMut},
    ops::{Deref, DerefMut},
};

/// [`Vector`] mutation proxy for unaligned SIMD data
///
/// For mutation from `&mut [Scalar]`, even if the number of elements is a
/// multiple of the number of vector lanes, we can't provide an `&mut Vector`
/// as it could be misaligned. So we provide a proxy object that acts as
/// closely to `&mut Vector` as possible.
pub struct UnalignedMut<'target, V: VectorInfo> {
    vector: V,
    target: &'target mut V::Array,
}
//
impl<'target, V: VectorInfo> UnalignedMut<'target, V> {
    /// Create a mutation project that acts like `&mut vector` but will
    /// eventually spill back into the `target` unaligned Vector reference
    #[inline(always)]
    pub(crate) fn new(vector: V, target: &'target mut V::Array) -> Self {
        Self { vector, target }
    }
}
//
impl<V: VectorInfo> Borrow<V> for UnalignedMut<'_, V> {
    #[inline(always)]
    fn borrow(&self) -> &V {
        &self.vector
    }
}
//
impl<V: VectorInfo> BorrowMut<V> for UnalignedMut<'_, V> {
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut V {
        &mut self.vector
    }
}
//
impl<V: VectorInfo> Deref for UnalignedMut<'_, V> {
    type Target = V;

    #[inline(always)]
    fn deref(&self) -> &V {
        &self.vector
    }
}
//
impl<V: VectorInfo> DerefMut for UnalignedMut<'_, V> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut V {
        &mut self.vector
    }
}
//
impl<V: VectorInfo> Drop for UnalignedMut<'_, V> {
    #[inline(always)]
    fn drop(&mut self) {
        *self.target = self.vector.into()
    }
}

/// [`Vector`] mutation proxy for padded scalar slices
///
/// For mutation from `&mut [Scalar]`, we can't provide an `&mut Vector` as it
/// could be misaligned and out of bounds. So we provide a proxy object
/// that acts as closely to `&mut Vector` as possible.
///
/// Trailing elements that do not match actual scalar data are initialized with
/// the padding value that was specified to [`Vectorizable::vectorize_pad`] and
/// will be discarded on `Drop`.
pub struct PaddedMut<'target, V: VectorInfo> {
    vector: V,
    target: &'target mut [V::Scalar],
}
//
impl<'target, V: VectorInfo> PaddedMut<'target, V> {
    /// Create a mutation project that acts like `&mut vector` but will
    /// eventually spill back into the `target` scalar slice
    #[inline(always)]
    pub(crate) fn new(vector: V, target: &'target mut [V::Scalar]) -> Self {
        Self { vector, target }
    }
}
//
impl<V: VectorInfo> Borrow<V> for PaddedMut<'_, V> {
    #[inline(always)]
    fn borrow(&self) -> &V {
        &self.vector
    }
}
//
impl<V: VectorInfo> BorrowMut<V> for PaddedMut<'_, V> {
    #[inline(always)]
    fn borrow_mut(&mut self) -> &mut V {
        &mut self.vector
    }
}
//
impl<V: VectorInfo> Deref for PaddedMut<'_, V> {
    type Target = V;

    #[inline(always)]
    fn deref(&self) -> &V {
        &self.vector
    }
}
//
impl<V: VectorInfo> DerefMut for PaddedMut<'_, V> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut V {
        &mut self.vector
    }
}
//
impl<V: VectorInfo> Drop for PaddedMut<'_, V> {
    #[inline(always)]
    fn drop(&mut self) {
        self.target
            .copy_from_slice(&self.vector.as_ref()[..self.target.len()]);
    }
}

// FIXME: Tests
