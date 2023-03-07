use multiversion::multiversion;
use rand::random;
use slipstream::prelude::*;
use std::fmt::Display;
use std::hint::black_box;
use std::iter;
use std::ops::Mul;
use std::time::Instant;

// Size of the vectors that are being multiplied
//
// This small vector size is chosen such that the working set fits in L1 cache,
// which is required to get optimal performance out of SIMD.
//
// The vector size should be divisible by `V::LANES * ILP_STREAMS`.
//
const SIZE: usize = 4096;

#[cfg(feature = "iterator_ilp")]
// Number of output SIMD vectors we process concurrently in the
// instruction-parallel version.
//
// Most compute-oriented CPUs can process multiple independent streams of
// arithmetic operations concurrently (e.g. current Intel and AMD CPUs can
// process two independent FMAs per CPU cycle). If we only feed those with a
// single stream of instructions that depend on each other, we lose performance,
// as demonstrated in this example.
//
// Do not tune this too high, otherwise you will run out of CPU registers or the
// compiler optimizer will give up and trash your code!
//
const ILP_STREAMS: usize = 4;

// Scalar and vector type
type Scalar = f32;
type V = f32x8;

// Number of benchmark repetitions
const RUNS: u32 = 1_000_000;

// FIXME: Depending on how lucky you are with memory allocator lottery, you may
//        or may not get a vector that's properly aligned for SIMD processing.
//        Using a Vec<V> would be better from this perspective.
#[derive(Debug, PartialEq)]
struct Vector(Vec<Scalar>);

impl Vector {
    fn random() -> Self {
        Self(iter::repeat_with(random).take(SIZE).collect())
    }
}

impl Mul for &'_ Vector {
    type Output = Scalar;

    #[inline(never)]
    fn mul(self, rhs: &Vector) -> Scalar {
        // The textbook algorithm: sum of component products
        self.0
            .iter()
            .zip(rhs.0.iter())
            .fold(0.0, |acc, (&l, &r)| acc + l * r)
    }
}

/// Simple SIMD dot product without parallel instruction streams
macro_rules! generate_simple_dot {
    ($name:ident, $dispatcher:literal) => {
        #[inline(never)]
        #[multiversion(targets = "simd", dispatcher = $dispatcher)]
        fn $name(lhs: &Vector, rhs: &Vector) -> Scalar {
            (&lhs.0[..], &rhs.0[..])
                .vectorize()
                .into_iter()
                .fold(V::splat(0.0), |acc, (lvec, rvec)| acc + lvec * rvec)
                .horizontal_sum()
        }
    };
}
generate_simple_dot!(simple_dot_static, "static");
generate_simple_dot!(simple_dot_dynamic, "default");

/// More advanced SIMD dot product with instruction-level parallelism and
/// Fused Multiply-Add.
///
/// While it may seem unfair to the simple version that FMA is only used in this
/// version, it is actually only beneficial here because the simple version is
/// latency-bound and FMA has a higher latency than addition.
#[cfg(feature = "iterator_ilp")]
mod iterator_ilp_based {
    use super::*;
    use iterator_ilp::IteratorILP;
    use multiversion::target::target_cfg_f;

    macro_rules! generate_parallel_dot {
        ($name:ident, $dispatcher:literal) => {
            #[inline(never)]
            #[multiversion(targets = "simd", dispatcher = $dispatcher)]
            pub(super) fn $name(lhs: &Vector, rhs: &Vector) -> Scalar {
                (&lhs.0[..], &rhs.0[..])
                    .vectorize()
                    .into_iter()
                    .fold_ilp::<ILP_STREAMS, _>(
                        || V::splat(0.0),
                        |acc, (lvec, rvec)| {
                            if target_cfg_f!(target_feature = "fma") {
                                lvec.mul_add(rvec, acc)
                            } else {
                                acc + lvec * rvec
                            }
                        },
                        |acc1, acc2| acc1 + acc2,
                    )
                    .horizontal_sum()
            }
        };
    }
    generate_parallel_dot!(parallel_dot_static, "static");
    generate_parallel_dot!(parallel_dot_dynamic, "default");
}

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
    let a = Vector::random();
    let b = Vector::random();

    let r0 = timed("Scalar dot product", || black_box(&a) * black_box(&b));

    let assert_close = |rtest: Scalar| {
        const TOLERANCE: f32 = 1e-5;
        assert!((rtest - r0).abs() < TOLERANCE * r0.abs());
    };

    let r1 = timed("Simple SIMD, compile-time detected", || {
        simple_dot_static(black_box(&a), black_box(&b))
    });
    assert_close(r1);
    let r2 = timed("Simple SIMD, run-time detected", || {
        simple_dot_dynamic(black_box(&a), black_box(&b))
    });
    assert_close(r2);

    #[cfg(feature = "iterator_ilp")]
    {
        let r3 = timed("Parallel SIMD, compile-time detected", || {
            iterator_ilp_based::parallel_dot_static(black_box(&a), black_box(&b))
        });
        assert_close(r3);
        let r4 = timed("Parallel SIMD, run-time detected", || {
            iterator_ilp_based::parallel_dot_dynamic(black_box(&a), black_box(&b))
        });
        assert_close(r4);
    }

    #[cfg(not(feature = "iterator_ilp"))]
    println!(
        "Please enable the iterator_ilp feature of this crate \
        to see the impact of instruction-level parallelism on performance"
    );
}
