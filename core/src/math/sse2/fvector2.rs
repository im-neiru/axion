// Imports from core
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Imports from std
use std::mem::MaybeUninit;

/// `FVector2` is a structure that represents a 2D vector with `f32` components.
/// It encapsulates two floating-point values and is used for various purposes in graphical applications
/// including points, vectors, and texture coordinates.
///
/// This structure is implemented with SIMD (Single Instruction, Multiple Data) which is a type of parallel computing
/// involving vectors that allows for multiple data points to be processed at once, resulting in performance improvements.
#[derive(Clone, Copy, Debug)]
pub struct FVector2 {
    pub x: f32,
    pub y: f32,
}

union UnionCast {
    v2: (FVector2, FVector2),
    m128: __m128,
}

impl Default for FVector2 {
    #[inline(always)]
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

impl FVector2 {
    /// A constant `FVector2` instance with both `x` and `y` components set to 0.0.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to 0.0.
    pub const ZERO: Self = Self::splat(0.0);

    /// A constant `FVector2` instance with both `x` and `y` components set to 1.0.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to 1.0.
    pub const ONE: Self = Self::splat(1.0);

    /// A constant `FVector2` instance with both `x` and `y` components set to -1.0.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to -1.0.
    pub const NEG_ONE: Self = Self::splat(-1.0);

    /// A constant `FVector2` instance with both `x` and `y` components set to the minimum finite value representable by `f32`.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to the minimum finite value representable by `f32`.
    pub const MIN: Self = Self::splat(f32::MIN);

    /// A constant `FVector2` instance with both `x` and `y` components set to the maximum finite value representable by `f32`.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to the maximum finite value representable by `f32`.
    pub const MAX: Self = Self::splat(f32::MAX);

    /// A constant `FVector2` instance with both `x` and `y` components set to a NaN (Not-a-Number) value.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to a NaN (Not-a-Number) value.
    pub const NAN: Self = Self::splat(f32::NAN);

    /// A constant `FVector2` instance with both `x` and `y` components set to positive infinity.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to positive infinity.
    pub const INFINITY: Self = Self::splat(f32::INFINITY);

    /// A constant `FVector2` instance with both `x` and `y` components set to negative infinity.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to negative infinity.
    pub const NEG_INFINITY: Self = Self::splat(f32::NEG_INFINITY);

    /// A constant `FVector2` instance with both `x` and `y` components set to the smallest positive value representable by `f32`.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to the smallest positive value representable by `f32`.
    pub const EPSILON: Self = Self::splat(f32::EPSILON);

    /// Constructs a new `FVector2`.
    ///
    /// # Arguments
    ///
    /// * `x` - A float that holds the x component of the vector.
    /// * `y` - A float that holds the y component of the vector.
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Create a new instance of `FVector2` with both `x` and `y` components set to the given `value`.
    ///
    /// This function creates a new `FVector2` instance with both `x` and `y` components
    /// initialized to the specified `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to set for both `x` and `y` components of the `FVector2`.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance with both `x` and `y` components set to `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::splat(5.0);
    /// assert_eq!(vector.x, 5.0);
    /// assert_eq!(vector.y, 5.0);
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self { x: value, y: value }
    }

    /// Returns the x and y components of the vector as a tuple.
    #[inline]
    pub const fn xy(self) -> (f32, f32) {
        (self.x, self.y)
    }

    /// Returns the y and x components of the vector as a tuple.
    #[inline]
    pub const fn yx(self) -> (f32, f32) {
        (self.y, self.x)
    }

    /// Returns the x component of the vector twice as a tuple.
    #[inline]
    pub const fn xx(self) -> (f32, f32) {
        (self.x, self.x)
    }

    /// Returns the y component of the vector twice as a tuple.
    #[inline]
    pub const fn yy(self) -> (f32, f32) {
        (self.y, self.y)
    }

    /// Returns the dot product of the vector and another vector.
    ///
    /// # Arguments
    ///
    /// * `other` - Another vector to calculate the dot product with.
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn dot(self, other: Self) -> f32 {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::default()),
            }
            .m128;
            let b = UnionCast {
                v2: (other, Self::default()),
            }
            .m128;

            let mut dot_product: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, b);
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
    pub fn length(self) -> f32 {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::default()),
            }
            .m128;

            let mut length: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
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
    pub fn length_sq(self) -> f32 {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::default()),
            }
            .m128;

            let mut length_sq: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
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
    /// use axion::math::*;
    /// let v = FVector2::new(3.0, 4.0);
    /// let inv_length = v.length_inv();
    /// println!("{}", inv_length); // prints: 0.2
    /// ```

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn length_inv(self) -> f32 {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::default()),
            }
            .m128;

            let mut length_inv: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
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
    /// use axion::math::*;
    /// let vector1 = FVector2::new(1.0, 2.0);
    /// let vector2 = FVector2::new(4.0, 6.0);
    /// let distance = vector1.distance(vector2);
    /// println!("Distance: {}", distance);
    /// ```
    ///
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn distance(self, other: Self) -> f32 {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::default()),
            }
            .m128;
            let b = UnionCast {
                v2: (other, Self::default()),
            }
            .m128;

            let mut distance: f32 = MaybeUninit::uninit().assume_init();
            let sub = _mm_sub_ps(a, b);
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
    /// * `FVector2` - A new FVector2 that is the normalized version of `self`.
    /// # Example
    ///
    /// ```rust
    /// let v = FVector2::new(3.0, 4.0);
    /// let normalized_v = v.normalize();
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn normalize(self) -> Self {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::default()),
            }
            .m128;
            let mut normalized_vec: (Self, Self) =
                MaybeUninit::uninit().assume_init();

            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let root = _mm_sqrt_ps(sum);
            let recip = _mm_set1_ps(_mm_cvtss_f32(_mm_rcp_ps(root)));
            let normalized = _mm_mul_ps(a, recip);

            _mm_store_ps(
                (&mut normalized_vec as *mut (Self, Self)) as *mut f32,
                normalized,
            );
            normalized_vec.0
        }
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;

            let b = UnionCast {
                v2: (Self::ZERO, Self::ZERO),
            }
            .m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) == 0xf
        }
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;

            let cmp_result = _mm_cmpunord_ps(a, a);

            _mm_movemask_ps(cmp_result) != 0
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() || self.y.is_finite()
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

    let ref_time = SystemTime::now();
    let is_zero = FVector2::ZERO.is_zero();

    let span_is_zero = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let is_nan = FVector2::NAN.is_nan();

    let span_is_nan = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let is_finite = FVector2::INFINITY.is_finite();

    let span_is_finite = SystemTime::now()
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

    println!(
        "FVector2::is_nan | {span_is_nan} ns | {:?}  = {is_nan}",
        FVector2::NAN.xy(),
    );

    println!(
        "FVector2::is_nan | {span_is_zero} ns | {:?}  = {is_zero}",
        FVector2::ZERO.xy(),
    );

    println!(
        "FVector2::is_finite | {span_is_finite} ns | {:?}  = {is_finite}",
        FVector2::INFINITY.xy(),
    );
}
