//! Fair warning: There are many things to think about when optimizing SIMD
//! matrix multiplication, and we did most of them here in the interest of
//! showing the library's peak SIMD throughput on a nontrivial example.
//!
//! If you are relatively unfamiliar with SIMD in general or this library in
//! particular, you may want to start with the simpler `dot_product` example.

use multiversion::{multiversion, target::target_cfg_f};
use rand::random;
use slipstream::prelude::*;
use std::fmt::Display;
use std::hint::black_box;
use std::iter;
use std::ops::Mul;
use std::time::Instant;

// Size of the matrices that are being computed
//
// This small matrix size is chosen such that the working set fits in L1 cache,
// which means we don't have to implement cache blocking optimizations to achive
// compute-bound performance and show the optimal effect of SIMD.
//
// The matrix size should be divisible by `V::LANES * ILP_STREAMS`.
//
const SIZE: usize = 64;

// Number of output SIMD vectors we process in parallel
//
// See the dot product example for more details on this operation
//
const ILP_STREAMS: usize = 8;

// Vector type
type Scalar = f32;
type V = f32x8;

// Number of benchmark repetitions
const RUNS: u32 = 10_000;

// FIXME: Depending on how lucky you are with memory allocator lottery, you may
//        or may not get a vector that's properly aligned for SIMD processing.
//        Using a Vec<V> would be better from this perspective.
#[derive(Debug, PartialEq)]
struct Matrix(Vec<Scalar>);

impl Matrix {
    fn random() -> Self {
        Self(iter::repeat_with(random).take(SIZE * SIZE).collect())
    }
}

impl Mul for &'_ Matrix {
    type Output = Matrix;

    #[inline(never)]
    fn mul(self, rhs: &Matrix) -> Matrix {
        let mut out = vec![0.0; SIZE * SIZE];
        // The textbook algorithm: iterate over output and lhs rows...
        for (lhs_row, out_row) in self.0.chunks_exact(SIZE).zip(out.chunks_exact_mut(SIZE)) {
            // ...then over output elements and rhs columns...
            for (col, out_elem) in out_row.iter_mut().enumerate() {
                let rhs_col = rhs.0.iter().skip(col).step_by(SIZE);
                // ...and compute dot product of selected lhs row and rhs column
                for (lhs_elem, rhs_elem) in lhs_row.iter().zip(rhs_col) {
                    *out_elem += *lhs_elem * *rhs_elem;
                }
            }
        }
        Matrix(out)
    }
}

// SIMD algorithm with compile-time or run-time SIMD instruction set detection
macro_rules! generate_mat_mult {
    ($name:ident, $dispatcher:literal) => {
        #[inline(never)]
        #[multiversion(targets = "simd", dispatcher = $dispatcher)]
        fn $name(lhs: &Matrix, rhs: &Matrix) -> Matrix {
            // We will produce output vectors in batches of ILP_STREAMS vectors,
            // assuming that each row cleanly divides into batches for simplicity
            const CHUNK_SIZE: usize = V::LANES * ILP_STREAMS;
            assert_eq!(SIZE % CHUNK_SIZE, 0);
            const CHUNKS_PER_ROW: usize = SIZE / CHUNK_SIZE;

            // Set up output buffer
            // FIXME: Use overaligned storage here too
            const NUM_ELEMS: usize = SIZE * SIZE;
            let mut out = vec![0.0; NUM_ELEMS];

            // Vectorize the right-hand-side matrix upfront
            let mut rhs_vecs = (&rhs.0[..]).vectorize();

            // Jointly iterate over output and lhs rows
            for (out_row, lhs_row) in out.chunks_exact_mut(SIZE).zip(lhs.0.chunks_exact(SIZE)) {
                // Prepare to concurrently generate ILP_STREAMS output vectors
                // within the current row. Keep track of where we are in the row.
                for (chunk_idx, mut out_chunk) in
                    out_row.vectorize().chunks_exact(ILP_STREAMS).enumerate()
                {
                    // Set up output accumulators
                    let mut accumulators = [V::default(); ILP_STREAMS];

                    // Jointly iterate over columns of lhs and rows and rhs.
                    // Within the selected row of rhs, target the columns that
                    // correspond to the output columns that we're generating.
                    for (&lhs_elem, mut rhs_chunk) in lhs_row.iter().zip(
                        rhs_vecs
                            .chunks_exact(ILP_STREAMS)
                            .skip(chunk_idx)
                            .step_by(CHUNKS_PER_ROW),
                    ) {
                        // Turn the active lhs element into a vector
                        let lhs_elem_vec = V::splat(lhs_elem);

                        // Add contribution from this rhs row to the output accumulator
                        for (acc, rhs_vec) in accumulators.iter_mut().zip(rhs_chunk.iter()) {
                            if target_cfg_f!(target_feature = "fma") {
                                *acc = lhs_elem_vec.mul_add(rhs_vec, *acc);
                            } else {
                                *acc += lhs_elem_vec * rhs_vec;
                            }
                        }
                    }

                    // Write down results into output storage
                    for (mut out_vec, &acc) in out_chunk.iter().zip(accumulators.iter()) {
                        *out_vec = acc;
                    }
                }
            }
            Matrix(out)
        }
    };
}
generate_mat_mult!(mat_mult_static, "static");
generate_mat_mult!(mat_mult_dynamic, "default");

fn timed<N: Display, R, F: FnMut() -> R>(name: N, mut f: F) -> R {
    let mut result = None;
    let start = Instant::now();
    for _ in 0..RUNS {
        result = Some(black_box(f()));
    }
    let elapsed = start.elapsed();
    println!("{} took:\t{:?} ({:?}/run)", name, elapsed, elapsed / RUNS);
    result.unwrap()
}

fn main() {
    let a = Matrix::random();
    let b = Matrix::random();

    let m0 = timed("Scalar multiplication", || black_box(&a) * black_box(&b));
    let m1 = timed("Compile-time detected", || {
        mat_mult_static(black_box(&a), black_box(&b))
    });
    let m2 = timed("Run-time detected", || {
        mat_mult_dynamic(black_box(&a), black_box(&b))
    });

    let assert_close = |mref: &Matrix, mtest: &Matrix| {
        const TOLERANCE: Scalar = 1e-6;
        assert!(mref
            .0
            .iter()
            .zip(mtest.0.iter())
            .all(|(eref, etest)| (eref - etest).abs() < TOLERANCE * eref.abs()));
    };
    assert_close(&m0, &m1);
    assert_close(&m0, &m2);
}
