//! Proxy objects that mimick `&mut Vector` in cases where we cannot construct
//! a true mutable reference for alignment and data length reasons

use crate::vectorize::VectorInfo;
#[cfg(doc)]
use crate::{vectorize::Vectorizable, Vector};
use core::{
    borrow::{Borrow, BorrowMut},
    fmt::{self, Debug},
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
impl<V: VectorInfo> AsMut<V> for UnalignedMut<'_, V> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut V {
        &mut self.vector
    }
}
//
impl<V: VectorInfo> AsRef<V> for UnalignedMut<'_, V> {
    #[inline(always)]
    fn as_ref(&self) -> &V {
        &self.vector
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
impl<V: VectorInfo + Debug> Debug for UnalignedMut<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UnalignedMut({:?} @ {:p})", self.vector, self.target)
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
//
impl<V: VectorInfo + Eq> Eq for UnalignedMut<'_, V> {}
//
impl<V: VectorInfo + PartialEq, VLike: Borrow<V>> PartialEq<VLike> for UnalignedMut<'_, V> {
    #[inline(always)]
    fn eq(&self, vector: &VLike) -> bool {
        self.vector.eq(vector.borrow())
    }
}

/// [`Vector`] mutation proxy for padded scalar slices
///
/// For mutation from `&mut [Scalar]`, we can't provide an `&mut Vector` as it
/// could be misaligned and out of bounds. So we provide a proxy object
/// that acts as closely to `&mut Vector` as possible.
///
/// Trailing elements that do not match actual scalar data are initialized with
/// the padding value that was specified to [`Vectorizable::vectorize_pad()`]
/// and will be discarded on `Drop`.
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
impl<V: VectorInfo> AsMut<V> for PaddedMut<'_, V> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut V {
        &mut self.vector
    }
}
//
impl<V: VectorInfo> AsRef<V> for PaddedMut<'_, V> {
    #[inline(always)]
    fn as_ref(&self) -> &V {
        &self.vector
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
impl<V: VectorInfo + Debug> Debug for PaddedMut<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PaddedMut({:?} @ {:p}[..{}])",
            self.vector,
            self.target,
            self.target.len()
        )
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
//
impl<V: VectorInfo + Eq> Eq for PaddedMut<'_, V> {}
//
impl<V: VectorInfo + PartialEq, VLike: Borrow<V>> PartialEq<VLike> for PaddedMut<'_, V> {
    #[inline(always)]
    fn eq(&self, vector: &VLike) -> bool {
        self.vector.eq(vector.borrow())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorize::tests::{any_v, VArray, V};
    use proptest::prelude::*;
    use std::ptr;

    /// Generate the building blocks to initialize UnalignedMut
    fn unaligned_init_input() -> impl Strategy<Value = (V, VArray)> {
        (any_v(), any::<VArray>())
    }

    /// Generate the building blocks to initialize a PaddedMut
    fn padded_init_input() -> impl Strategy<Value = (V, VArray, usize)> {
        (unaligned_init_input(), 0..V::LANES).prop_map(|((init, target), len)| (init, target, len))
    }

    // Test properties of a freshly initialized mutability proxy
    fn test_init<
        MutProxy: AsMut<V>
            + AsRef<V>
            + Borrow<V>
            + BorrowMut<V>
            + Deref<Target = V>
            + DerefMut<Target = V>
            + Debug
            + PartialEq<V>,
    >(
        init: V,
        mut proxy: MutProxy,
    ) {
        // Check reference accessors
        assert_eq!(*proxy.as_ref(), init);
        assert!(ptr::eq(proxy.as_ref(), proxy.as_mut()));
        assert!(ptr::eq(proxy.as_ref(), proxy.borrow()));
        assert!(ptr::eq(proxy.as_ref(), proxy.borrow_mut()));
        assert!(ptr::eq(proxy.as_ref(), proxy.deref()));
        assert!(ptr::eq(proxy.as_ref(), proxy.deref_mut()));

        // Check debug printout doesn't crash and produces non-empty output
        assert_ne!(format!("{proxy:?}").len(), 0);

        // Check PartialEq with V
        assert_eq!(proxy, init);
    }

    proptest! {
        // Test initializing UnalignedMut
        #[test]
        fn init_unaligned((init, mut target) in unaligned_init_input()) {
            test_init(init, UnalignedMut::new(init, &mut target));
            assert_eq!(target, VArray::from(init));
        }

        // Test initializing a PaddedMut
        #[test]
        fn init_padded((init, mut target, target_len) in padded_init_input()) {
            let old_target = target.clone();
            test_init(init, PaddedMut::new(init, &mut target[..target_len]));
            let init_arr = VArray::from(init);
            assert_eq!(&target[..target_len], &init_arr[..target_len]);
            assert_eq!(&target[target_len..], &old_target[target_len..]);
        }

        // Test setting an UnalignedMut to a new value
        #[test]
        fn set_unaligned(((init, mut target), new) in (unaligned_init_input(), any_v())) {
            let mut proxy = UnalignedMut::new(init, &mut target);
            *proxy = new;
            test_init(new, proxy);
            assert_eq!(target, VArray::from(new));
        }

        // Test setting a PaddedMut to a new value
        #[test]
        fn set_padded(((init, mut target, target_len), new) in (padded_init_input(), any_v())) {
            let old_target = target.clone();
            let mut proxy = PaddedMut::new(init, &mut target[..target_len]);
            *proxy = new;
            test_init(new, proxy);
            let new_arr = VArray::from(new);
            assert_eq!(&target[..target_len], &new_arr[..target_len]);
            assert_eq!(&target[target_len..], &old_target[target_len..]);
        }
    }
}
