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
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn length(self) -> f32 {
        unsafe {
            let mut length: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(self.0, self.0);
            let sum = _mm_hadd_ps(product, product);
            let root = _mm_sqrt_ps(sum);

            _mm_store_ss(&mut length, root);

            length
        }
    }

    /// Returns the squared length of the vector.
    ///
    /// This function is often used in graphics programming when comparing lengths, as it avoids the computationally
    /// expensive square root operation. In comparison operations, the exact length is often not necessary,
    /// so the square length can be used instead.
    ///
    /// The squared length is calculated by performing the dot product of the vector with itself (`dot` function).
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn length_sq(self) -> f32 {
        unsafe {
            let mut length_sq: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(self.0, self.0);
            let sum = _mm_hadd_ps(product, product);

            _mm_store_ss(&mut length_sq, sum);

            length_sq
        }
    }

    //// Returns the inverse of the length (reciprocal of the magnitude) of the vector,
    /// If the length of the vector is zero, this function will return an `f32::INFINITY`.
    ///
    ///
    /// The inverse length is calculated by taking the reciprocal of the length of the vector (`length` function).
    ///
    /// # Examples
    ///
    /// ```
    /// use vector::FVector2;
    /// let v = FVector2::new(3.0, 4.0);
    /// let inv_length = v.length_inv();
    /// println!("{}", inv_length); // prints: 0.2
    /// ```

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn length_inv(self) -> f32 {
        unsafe {
            let mut length_inv: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(self.0, self.0);
            let sum = _mm_hadd_ps(product, product);
            let root = _mm_sqrt_ps(sum);
            let recip = _mm_rcp_ps(root);

            _mm_store_ss(&mut length_inv, recip);

            length_inv
        }
    }

    /// Calculates the Euclidean distance between two 2D vectors.
    ///
    /// This function computes the Euclidean distance between two `FVector2`
    ///
    /// # Arguments
    ///
    /// * `self` - The first vector for which to calculate the distance.
    /// * `other` - The second vector to which the distance is calculated.
    ///
    /// # Returns
    ///
    /// The calculated Euclidean distance between the two input vectors as a `f32`.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use some_vector_library::Vector2;
    /// let vector1 = Vector2::new(1.0, 2.0);
    /// let vector2 = Vector2::new(4.0, 6.0);
    /// let distance = vector1.distance(vector2);
    /// println!("Distance: {}", distance);
    /// ```
    ///
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn distance(self, other: Self) -> f32 {
        unsafe {
            let mut distance: f32 = MaybeUninit::uninit().assume_init();
            let sub = _mm_sub_ps(self.0, other.0);
            let product = _mm_mul_ps(sub, sub);
            let sum = _mm_hadd_ps(product, product);
            let root = _mm_sqrt_ps(sum);

            _mm_store_ss(&mut distance, root);

            distance
        }
    }

    /// The `normalize` function normalizes a vector using SIMD instructions for efficiency.
    /// if the vector is at the origin `(0.0, 0.0)`, as this would lead
    /// to a division by zero when calculating the reciprocal of the root. The result in such case
    /// would be `(f32::NAN, f32::NAN)`. It is recommended to check if the vector is at the origin before
    /// calling this function.
    ///
    /// # Arguments
    ///
    /// * `self` - A FVector2 to be normalized.
    ///
    /// # Returns
    ///
    /// * `FVector2` - A new Vector2 that is the normalized version of `self`.
    /// # Example
    ///
    /// ```rust
    /// let v = FVector2::new(3.0, 4.0);
    /// let normalized_v = v.normalize();
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn normalize(self) -> Self {
        unsafe {
            let product = _mm_mul_ps(self.0, self.0);
            let sum = _mm_hadd_ps(product, product);
            let root = _mm_sqrt_ps(sum);
            let recip = _mm_set1_ps(_mm_cvtss_f32(_mm_rcp_ps(root)));
            let normalized = _mm_mul_ps(self.0, recip);

            Self(normalized)
        }
    }
}

#[test]
fn test_fvector2_sse2() {
    use std::time::SystemTime;

    let a = FVector2::new(3.0, 4.0);
    let b = FVector2::new(4.0, 3.2);
    let w = a.dot(b); // warm up
    println!("Warm up dot = {w}");

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

    let ref_time = SystemTime::now();
    let normalize = a.normalize();

    let span_normalize = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let distance = a.distance(b);

    let span_distance = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    assert!(dot == 24.8, "FVector2::dot | Wrong output");
    assert!(length == 5.0, "FVector2::length | Wrong output");

    println!(
        "FVector2::dot | {span_dot} ns | {:?} {:?} = {dot}",
        a.xy(),
        b.xy()
    );

    println!(
        "FVector2::length | {span_length} ns | {:?} = {length}",
        a.xy(),
    );

    println!(
        "FVector2::normalize | {span_normalize} ns | {:?} = {:?}",
        a.xy(),
        normalize.xy()
    );

    println!(
        "FVector2::distance | {span_distance} ns | {:?} {:?} = {distance}",
        a.xy(),
        b.xy()
    );
}
