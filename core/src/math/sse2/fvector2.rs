// Imports from core
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Imports from std
use std::mem::MaybeUninit;

// Import from crate
use crate::math::{Vector, Vector2};

/// `FVector2` is a structure that represents a 2D vector with `f32` components.
/// It encapsulates two floating-point values and is used for various purposes in graphical applications
/// including points, vectors, and texture coordinates.
///
/// This structure is implemented with SIMD (Single Instruction, Multiple Data) which is a type of parallel computing
/// involving vectors that allows for multiple data points to be processed at once, resulting in performance improvements.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FVector2(pub(crate) __m128);

impl Default for FVector2 {
    #[inline]
    fn default() -> Self {
        Self(unsafe { _mm_setzero_ps() })
    }
}

/// This is a Vector2 implementation for f32 type.
impl Vector2<f32> for FVector2 {
    /// Constructs a new `FVector2`.
    ///
    /// # Arguments
    ///
    /// * `x` - A float that holds the x component of the vector.
    /// * `y` - A float that holds the y component of the vector.
    #[inline]
    fn new(x: f32, y: f32) -> Self {
        Self(unsafe { _mm_set_ps(0.0, 0.0, y, x) })
    }

    /// Returns the x component of the vector.
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn x(self) -> f32 {
        unsafe {
            let mut x: f32 = MaybeUninit::uninit().assume_init();
            _mm_store_ss(&mut x, self.0);

            x
        }
    }

    /// Returns the y component of the vector.
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn y(self) -> f32 {
        unsafe {
            let mut y: f32 = MaybeUninit::uninit().assume_init();

            let v = _mm_shuffle_ps::<1>(self.0, self.0);
            _mm_store_ss(&mut y, v);

            y
        }
    }

    /// Returns the x and y components of the vector as a tuple.
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn xy(self) -> (f32, f32) {
        unsafe {
            let mut x: f32 = MaybeUninit::uninit().assume_init();
            let mut y: f32 = MaybeUninit::uninit().assume_init();

            _mm_store_ss(&mut x, self.0);
            let v = _mm_shuffle_ps::<1>(self.0, self.0);
            _mm_store_ss(&mut y, v);

            (x, y)
        }
    }

    /// Returns the y and x components of the vector as a tuple.
    #[inline]
    fn yx(self) -> (f32, f32) {
        let xy = self.xy();
        (xy.1, xy.0)
    }

    /// Returns the x component of the vector twice as a tuple.
    #[inline]
    fn xx(self) -> (f32, f32) {
        let x = self.x();
        (x, x)
    }

    /// Returns the y component of the vector twice as a tuple.
    #[inline]
    fn yy(self) -> (f32, f32) {
        let y = self.y();
        (y, y)
    }
}

impl Vector<f32> for FVector2 {
    /// Returns the dot product of the vector and another vector.
    ///
    /// # Arguments
    ///
    /// * `other` - Another vector to calculate the dot product with.
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn dot(self, other: Self) -> f32 {
        unsafe {
            let mut dot_product: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(self.0, other.0);
            let sum = _mm_hadd_ps(product, product);

            _mm_store_ss(&mut dot_product, sum);

            dot_product
        }
    }

    /// Returns the length (magnitude) of the vector.
    ///
    /// The length of a vector is a non-negative number that describes the extent of the vector in space. It is always
    /// positive unless the vector has all components equal to zero, in which case the length is zero.
    ///
    /// The length is calculated by first squaring the components of the vector (`length_sq` function), and then taking
    /// the square root of the sum.
    ///
    /// This function may be used in graphics programming to determine the distance of a point represented by the vector
    /// from the origin of the vector space. As such, it can be used to calculate distances between points.
    #[inline]
    fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Returns the squared length of the vector.
    ///
    /// This function is often used in graphics programming when comparing lengths, as it avoids the computationally
    /// expensive square root operation. In comparison operations, the exact length is often not necessary,
    /// so the square length can be used instead.
    ///
    /// The squared length is calculated by performing the dot product of the vector with itself (`dot` function).
    #[inline]
    fn length_sq(self) -> f32 {
        self.dot(self)
    }

    /// Returns the inverse of the length (reciprocal of the magnitude) of the vector.
    ///
    /// The inverse length may be used in graphics programming to normalize a vector. Normalization refers to the process
    /// of scaling the vector to make its length equal to one, while preserving its direction. By multiplying a vector
    /// by its inverse length, we can quickly obtain a unit vector.
    ///
    /// The inverse length is calculated by taking the reciprocal of the length of the vector (`length` function).
    #[inline]
    fn length_inv(self) -> f32 {
        self.length().recip()
    }
}

#[test]
fn test_fvector2_sse2() {
    use std::time::SystemTime;

    let a = FVector2::new(2.0, 3.0);
    let b = FVector2::new(4.0, 3.2);
    let ref_time = SystemTime::now();
    let dot = a.dot(b);

    let span_dot = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let length = a.length();

    let span_length = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    assert!(dot == 17.6, "FVector2::dot | Wrong output");
    assert!(length == 3.6055512, "FVector2::length | Wrong output");

    println!(
        "FVector2::dot | {span_dot} ns | {:?} {:?} = {dot}",
        a.xy(),
        b.xy()
    );

    println!(
        "FVector2::length | {span_length} ns | {:?} = {length}",
        a.xy(),
    );
}
