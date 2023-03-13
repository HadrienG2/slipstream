//! Iteration over `Vectorized`

mod chunks;
mod values;

pub use chunks::{
    Chunks, ChunksExact, GenericChunks, GenericChunksExact, RefChunks, RefChunksExact,
};
pub use values::{GenericIter, IntoIter, Iter, RefIter};
