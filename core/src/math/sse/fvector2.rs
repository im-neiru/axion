// Imports from core
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Imports from std
use std::{fmt, mem::MaybeUninit, ops};

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

    /// A constant `FVector2` representing the positive X-axis.
    ///
    /// This constant vector has a value of `(1.0, 0.0)`, representing the positive X-axis in 2D space.
    pub const AXIS_X: Self = Self { x: 1.0, y: 0.0 };

    /// A constant `FVector2` representing the positive Y-axis.
    ///
    /// This constant vector has a value of `(0.0, 1.0)`, representing the positive Y-axis in 2D space.
    pub const AXIS_Y: Self = Self { x: 0.0, y: 1.0 };

    /// A constant `FVector2` representing the negative X-axis.
    ///
    /// This constant vector has a value of `(-1.0, 0.0)`, representing the negative X-axis in 2D space.
    pub const NEG_AXIS_X: Self = Self { x: -1.0, y: 0.0 };

    /// A constant `FVector2` representing the negative Y-axis.
    ///
    /// This constant vector has a value of `(0.0, -1.0)`, representing the negative Y-axis in 2D space.
    pub const NEG_AXIS_Y: Self = Self { x: 0.0, y: -1.0 };

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

    /// Adds the `z` axis value to a `FVector2`, creating a new `FVector3`.
    ///
    /// This function takes a `FVector2` and a `z` value, and returns a new `FVector3`
    /// with the `x` and `y` components from the input `FVector2` and the provided `z` value.
    ///
    /// # Arguments
    ///
    /// * `self` - The input `FVector2` that will serve as the basis for the `x` and `y` components of the resulting `FVector3`.
    /// * `z` - The value to be used as the `z` component of the resulting `FVector3`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` with `x` and `y` components inherited from the input `FVector2`
    /// and the `z` component set to the provided `z` value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector2;
    ///
    /// let f2 = FVector2::new(1.0, 2.0);
    /// let f3 = f2.add_axis(3.0); // New FVector3
    ///
    /// assert_eq!(f3.x, 1.0);
    /// assert_eq!(f3.y, 2.0);
    /// assert_eq!(f3.z, 3.0);
    /// ```
    #[inline]
    pub const fn add_axis(self, z: f32) -> super::FVector3 {
        super::FVector3 {
            x: self.x,
            y: self.y,
            z,
        }
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
                v2: (self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (other, Self::ZERO),
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
                v2: (self, Self::ZERO),
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
                v2: (self, Self::ZERO),
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
                v2: (self, Self::ZERO),
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
                v2: (self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (other, Self::ZERO),
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

    /// Computes the squared Euclidean distance between two `FVector2` instances.
    ///
    /// This method calculates the squared Euclidean distance between `self` and `other`, which is a
    /// more efficient version of the Euclidean distance as it avoids the square root operation.
    ///
    /// # Arguments
    ///
    /// * `other` - The `FVector2` instance representing the other point in space.
    ///
    /// # Returns
    ///
    /// The squared Euclidean distance between `self` and `other` as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let point1 = FVector2::new(2.0, 3.0);
    /// let point2 = FVector2::new(1.0, 2.0);
    /// let distance_sq = point1.distance_sq(point2);
    ///
    /// assert_eq!(distance_sq, 2.0);
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn distance_sq(self, other: Self) -> f32 {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (other, Self::ZERO),
            }
            .m128;

            let mut distance_sq: f32 = MaybeUninit::uninit().assume_init();
            let sub = _mm_sub_ps(a, b);
            let product = _mm_mul_ps(sub, sub);
            let sum = _mm_hadd_ps(product, product);

            _mm_store_ss(&mut distance_sq, sum);

            distance_sq
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
                v2: (self, Self::ZERO),
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

    /// Performs linear interpolation between two `FVector2` instances.
    ///
    /// Linear interpolation, or lerp, blends between two `FVector2` instances using a specified
    /// interpolation factor `end`. The result is a new `FVector2` where each component is interpolated
    /// between the corresponding components of `self` and `end` based on `end`.
    ///
    /// # Arguments
    ///
    /// * `end` - The target `FVector2` to interpolate towards.
    /// * `scalar` - The interpolation factor, typically in the range [0.0, 1.0], where:
    ///   - `scalar = 0.0` returns `self`.
    ///   - `scalar = 1.0` returns `end`.
    ///   - Values in between interpolate linearly.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance resulting from the linear interpolation of `self` and `end` with factor `end`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let start = FVector2::new(1.0, 2.0);
    /// let end = FVector2::new(3.0, 4.0);
    /// let interpolated = start.lerp(end, 0.5);
    ///
    /// assert_eq!(interpolated, FVector2::new(2.0, 3.0));
    /// ```
    #[inline]
    pub fn lerp(self, end: Self, scalar: f32) -> Self {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;

            let b = UnionCast {
                v2: (end, Self::ZERO),
            }
            .m128;

            UnionCast {
                m128: _mm_add_ps(
                    a,
                    _mm_mul_ps(_mm_sub_ps(b, a), _mm_set1_ps(scalar)),
                ),
            }
            .v2
            .0
        }
    }

    /// Clamps each component of the `FVector2` to be within the specified range.
    ///
    /// This function takes an `FVector2` instance and ensures that each component is
    /// within the specified `min` and `max` bounds. If a component is less than `min`,
    /// it will be set to `min`, and if it is greater than `max`, it will be set to `max`.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum bounds for each component.
    /// * `max` - The maximum bounds for each component.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance with each component clamped to the specified range.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::new(2.0, 5.0);
    /// let clamped = vector.clamp(FVector2::new(1.0, 3.0), FVector2::new(4.0, 6.0));
    /// assert_eq!(clamped, FVector2::new(2.0, 5.0));
    /// ```
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Returns an `FVector2` with each component set to the minimum of the corresponding components of `self` and `value`.
    ///
    /// This function compares each component of `self` and `value` and returns a new `FVector2`
    /// with each component set to the minimum of the corresponding components of `self` and `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The `FVector2` to compare with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance with each component set to the minimum of the corresponding components of `self` and `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(2.0, 5.0);
    /// let vector2 = FVector2::new(1.0, 3.0);
    /// let min_vector = vector1.min(vector2);
    /// assert_eq!(min_vector, FVector2::new(1.0, 3.0));
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn min(self, value: Self) -> Self {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;

            let b = UnionCast {
                v2: (value, Self::ZERO),
            }
            .m128;

            UnionCast {
                m128: _mm_min_ps(a, b),
            }
            .v2
            .0
        }
    }

    /// Returns an `FVector2` with each component set to the maximum of the corresponding components of `self` and `value`.
    ///
    /// This function compares each component of `self` and `value` and returns a new `FVector2`
    /// with each component set to the maximum of the corresponding components of `self` and `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The `FVector2` to compare with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance with each component set to the maximum of the corresponding components of `self` and `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(2.0, 5.0);
    /// let vector2 = FVector2::new(1.0, 3.0);
    /// let max_vector = vector1.max(vector2);
    /// assert_eq!(max_vector, FVector2::new(2.0, 5.0));
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn max(self, value: Self) -> Self {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;

            let b = UnionCast {
                v2: (value, Self::ZERO),
            }
            .m128;

            UnionCast {
                m128: _mm_max_ps(a, b),
            }
            .v2
            .0
        }
    }

    /// Checks if all components of the `FVector2` are equal to zero.
    ///
    /// This function compares each component of the `FVector2` with zero and returns
    /// `true` if all components are equal to zero, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if all components of the `FVector2` are equal to zero, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let zero_vector = FVector2::ZERO;
    /// let non_zero_vector = FVector2::new(2.0, 0.0);
    ///
    /// assert_eq!(zero_vector.is_zero(), true);
    /// assert_eq!(non_zero_vector.is_zero(), false);
    /// ```
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

    /// Checks if any component of the `FVector2` is NaN (Not-a-Number).
    ///
    /// This function checks each component of the `FVector2` for NaN and returns `true`
    /// if at least one component is NaN, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if any component of the `FVector2` is NaN, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let nan_vector = FVector2::new(f32::NAN, 2.0);
    /// let valid_vector = FVector2::new(1.0, 3.0);
    ///
    /// assert_eq!(nan_vector.is_nan(), true);
    /// assert_eq!(valid_vector.is_nan(), false);
    /// ```
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

    /// Checks if any component of the `FVector2` is finite.
    ///
    /// This function checks each component of the `FVector2` for finiteness and returns
    /// `true` if at least one component is finite, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if any component of the `FVector2` is finite, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let finite_vector = FVector2::new(2.0, 3.0);
    /// let infinite_vector = FVector2::new(f32::INFINITY, 2.0);
    ///
    /// assert_eq!(finite_vector.is_finite(), true);
    /// assert_eq!(infinite_vector.is_finite(), false);
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() || self.y.is_finite()
    }
}

impl PartialEq for FVector2 {
    /// Checks if two `FVector2` instances are equal.
    ///
    /// This implementation compares each component of two `FVector2` instances for equality
    /// and returns `true` if all components are equal, and `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to compare with.
    ///
    /// # Returns
    ///
    /// `true` if all components of the two `FVector2` instances are equal, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(2.0, 3.0);
    /// let vector2 = FVector2::new(2.0, 3.0);
    /// let vector3 = FVector2::new(1.0, 2.0);
    ///
    /// assert_eq!(vector1, vector2);
    /// assert_ne!(vector1, vector3);
    /// ```
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let a = UnionCast {
                v2: (*self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (*rhs, Self::ZERO),
            }
            .m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) == 0xf
        }
    }

    /// Checks if two `FVector2` instances are not equal.
    ///
    /// This implementation compares each component of two `FVector2` instances for inequality
    /// and returns `true` if at least one component is not equal, and `false` if all components
    /// are equal.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to compare with.
    ///
    /// # Returns
    ///
    /// `true` if at least one component of the two `FVector2` instances is not equal, `false` if all components are equal.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(2.0, 3.0);
    /// let vector2 = FVector2::new(1.0, 2.0);
    ///
    /// assert_ne!(vector1, vector2);
    /// ```

    #[inline]
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, rhs: &Self) -> bool {
        unsafe {
            let a = UnionCast {
                v2: (*self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (*rhs, Self::ZERO),
            }
            .m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) != 0xf
        }
    }
}

impl fmt::Display for FVector2 {
    /// Formats the `FVector2` as a string in the form *(x, y)*.
    ///
    /// This implementation allows you to format an `FVector2` instance as a string,
    /// where the `x` and `y` components are enclosed in parentheses and separated by a comma.
    ///
    /// # Arguments
    ///
    /// * `formatter` - A mutable reference to the `fmt::Formatter` used for formatting.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating whether the formatting was successful.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::new(2.0, 5.0);
    /// let formatted = format!("{}", vector);
    ///
    /// assert_eq!(formatted, "(2.0, 5.0)");
    /// ```
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}, {})", self.x, self.y)
    }
}

impl ops::Index<usize> for FVector2 {
    type Output = f32;

    /// Indexes the `FVector2` by a `usize` index.
    ///
    /// This implementation allows you to access the components of an `FVector2` using a
    /// `usize` index where `0` corresponds to `x` and `1` corresponds to `y`. It returns
    /// a reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access, where `0` represents `x` and `1` represents `y`.
    ///
    /// # Returns
    ///
    /// A reference to the specified component of the `FVector2`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not 0 or 1).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::new(2.0, 3.0);
    /// let x_component = vector[0];
    ///
    /// assert_eq!(x_component, 2.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::IndexMut<usize> for FVector2 {
    /// Mutable indexing for the `FVector2` by a `usize` index.
    ///
    /// This implementation allows you to mutably access and modify the components of an `FVector2`
    /// using a `usize` index where `0` corresponds to `x` and `1` corresponds to `y`. It returns
    /// a mutable reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access and modify, where `0` represents `x` and `1` represents `y`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the specified component of the `FVector2`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not 0 or 1).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(2.0, 3.0);
    /// vector[0] = 4.0;
    ///
    /// assert_eq!(vector[0], 4.0);
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::Index<u32> for FVector2 {
    type Output = f32;

    /// Indexes the `FVector2` by a `u32` index.
    ///
    /// This implementation allows you to access the components of an `FVector2` using a
    /// `u32` index where `0` corresponds to `x` and `1` corresponds to `y`. It returns
    /// a reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access, where `0` represents `x` and `1` represents `y`.
    ///
    /// # Returns
    ///
    /// A reference to the specified component of the `FVector2`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not 0 or 1).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::new(2.0, 3.0);
    /// let x_component = vector[0u32];
    ///
    /// assert_eq!(x_component, 2.0);
    /// ```
    #[inline]
    fn index(&self, index: u32) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::IndexMut<u32> for FVector2 {
    /// Mutable indexing for the `FVector2` by a `u32` index.
    ///
    /// This implementation allows you to mutably access and modify the components of an `FVector2`
    /// using a `u32` index where `0` corresponds to `x` and `1` corresponds to `y`. It returns
    /// a mutable reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access and modify, where `0` represents `x` and `1` represents `y`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the specified component of the `FVector2`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not 0 or 1).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(2.0, 3.0);
    /// vector[0u32] = 4.0;
    ///
    /// assert_eq!(vector[0u32], 4.0);
    /// ```
    #[inline]
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::Add<Self> for FVector2 {
    type Output = Self;

    /// Adds two `FVector2` instances component-wise.
    ///
    /// This implementation allows you to add two `FVector2` instances together
    /// component-wise, resulting in a new `FVector2` where each component is the
    /// sum of the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance resulting from the component-wise addition of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(2.0, 3.0);
    /// let vector2 = FVector2::new(1.0, 2.0);
    /// let result = vector1 + vector2;
    ///
    /// assert_eq!(result, FVector2::new(3.0, 5.0));
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (rhs, Self::ZERO),
            }
            .m128;

            UnionCast {
                m128: _mm_add_ps(a, b),
            }
            .v2
            .0
        }
    }
}

impl ops::Sub<Self> for FVector2 {
    type Output = Self;

    /// Subtracts one `FVector2` from another component-wise.
    ///
    /// This implementation allows you to subtract one `FVector2` from another
    /// component-wise, resulting in a new `FVector2` where each component is the
    /// difference between the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to subtract from `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance resulting from the component-wise subtraction of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(2.0, 3.0);
    /// let vector2 = FVector2::new(1.0, 2.0);
    /// let result = vector1 - vector2;
    ///
    /// assert_eq!(result, FVector2::new(1.0, 1.0));
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (rhs, Self::ZERO),
            }
            .m128;

            UnionCast {
                m128: _mm_sub_ps(a, b),
            }
            .v2
            .0
        }
    }
}

impl ops::Mul<Self> for FVector2 {
    type Output = Self;

    /// Multiplies two `FVector2` instances component-wise.
    ///
    /// This implementation allows you to multiply two `FVector2` instances together
    /// component-wise, resulting in a new `FVector2` where each component is the
    /// product of the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance resulting from the component-wise multiplication of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(2.0, 3.0);
    /// let vector2 = FVector2::new(1.0, 2.0);
    /// let result = vector1 * vector2;
    ///
    /// assert_eq!(result, FVector2::new(2.0, 6.0));
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (rhs, Self::ZERO),
            }
            .m128;

            UnionCast {
                m128: _mm_mul_ps(a, b),
            }
            .v2
            .0
        }
    }
}

impl ops::Div<Self> for FVector2 {
    type Output = Self;

    /// Divides one `FVector2` by another component-wise.
    ///
    /// This implementation allows you to divide one `FVector2` by another component-wise,
    /// resulting in a new `FVector2` where each component is the result of dividing the
    /// corresponding components of `self` by `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to divide `self` by.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance resulting from the component-wise division of `self` by `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(2.0, 6.0);
    /// let vector2 = FVector2::new(1.0, 2.0);
    /// let result = vector1 / vector2;
    ///
    /// assert_eq!(result, FVector2::new(2.0, 3.0));
    /// ```
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (rhs, Self::ZERO),
            }
            .m128;

            UnionCast {
                m128: _mm_div_ps(a, b),
            }
            .v2
            .0
        }
    }
}

impl ops::Rem<Self> for FVector2 {
    type Output = Self;

    /// Computes the remainder of dividing one `FVector2` by another component-wise.
    ///
    /// This implementation allows you to compute the remainder of dividing one `FVector2`
    /// by another component-wise, resulting in a new `FVector2` where each component is
    /// the remainder of dividing the corresponding components of `self` by `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to compute the remainder of division with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance resulting from the component-wise remainder of division of `self` by `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector1 = FVector2::new(7.0, 10.0);
    /// let vector2 = FVector2::new(3.0, 4.0);
    /// let result = vector1 % vector2;
    ///
    /// assert_eq!(result, FVector2::new(1.0, 2.0));
    /// ```
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast {
                v2: (self, Self::ZERO),
            }
            .m128;
            let b = UnionCast {
                v2: (rhs, Self::ZERO),
            }
            .m128;

            UnionCast {
                m128: _mm_sub_ps(
                    a,
                    _mm_mul_ps(b, _mm_floor_ps(_mm_div_ps(a, b))),
                ),
            }
            .v2
            .0
        }
    }
}

impl ops::AddAssign<Self> for FVector2 {
    /// Adds another `FVector2` to `self` in place.
    ///
    /// This implementation allows you to add another `FVector2` to `self` in place, modifying
    /// `self` to be the result of the addition.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to add to `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(2.0, 3.0);
    /// vector += FVector2::new(1.0, 2.0);
    ///
    /// assert_eq!(vector, FVector2::new(3.0, 5.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ops::SubAssign<Self> for FVector2 {
    /// Subtracts another `FVector2` from `self` in place.
    ///
    /// This implementation allows you to subtract another `FVector2` from `self` in place,
    /// modifying `self` to be the result of the subtraction.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to subtract from `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(2.0, 3.0);
    /// vector -= FVector2::new(1.0, 2.0);
    ///
    /// assert_eq!(vector, FVector2::new(1.0, 1.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ops::MulAssign<Self> for FVector2 {
    /// Multiplies `self` by another `FVector2` in place.
    ///
    /// This implementation allows you to multiply `self` by another `FVector2` in place,
    /// modifying `self` to be the result of the multiplication.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to multiply `self` with.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(2.0, 3.0);
    /// vector *= FVector2::new(1.0, 2.0);
    ///
    /// assert_eq!(vector, FVector2::new(2.0, 6.0));
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl ops::DivAssign<Self> for FVector2 {
    /// Divides `self` by another `FVector2` in place.
    ///
    /// This implementation allows you to divide `self` by another `FVector2` in place,
    /// modifying `self` to be the result of the division.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to divide `self` by.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(2.0, 6.0);
    /// vector /= FVector2::new(1.0, 2.0);
    ///
    /// assert_eq!(vector, FVector2::new(2.0, 3.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl ops::RemAssign<Self> for FVector2 {
    /// Computes the remainder of dividing `self` by another `FVector2` in place.
    ///
    /// This implementation allows you to compute the remainder of dividing `self` by another
    /// `FVector2` in place, modifying `self` to be the result of the remainder operation.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector2` to compute the remainder of division with `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(7.0, 10.0);
    /// vector %= FVector2::new(3.0, 4.0);
    ///
    /// assert_eq!(vector, FVector2::new(1.0, 2.0));
    /// ```
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
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

    println!("FVector2::dot | {span_dot} ns | {a} ⋅ {b} = {dot}",);

    println!("FVector2::length | {span_length} ns | {a} = {length}",);

    println!("FVector2::normalize | {span_normalize} ns | {a} = {normalize}",);

    println!("FVector2::distance | {span_distance} ns | {a} {b} = {distance}",);

    println!(
        "FVector2::is_nan | {span_is_nan} ns | {}  = {is_nan}",
        FVector2::NAN,
    );

    println!(
        "FVector2::is_nan | {span_is_zero} ns | {}  = {is_zero}",
        FVector2::ZERO,
    );

    println!(
        "FVector2::is_finite | {span_is_finite} ns | {}  = {is_finite}",
        FVector2::INFINITY,
    );

    {
        let start = FVector2::new(1.0, 2.0);
        let end = FVector2::new(3.0, 4.0);

        let ref_time = SystemTime::now();

        let lerp = start.lerp(end, 0.5);

        let span_lerp = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector2::lerp | {span_lerp} ns | {start} {end} = {lerp}");
    }

    {
        let a = FVector2::new(1.0, 2.0);
        let b = FVector2::new(1.0, 2.0);

        let ref_time = SystemTime::now();

        let eq = a == b;

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector2::eq | {span} ns | {a} == {b} = {eq}");
    }

    {
        let a = FVector2::new(1.0, 2.0);
        let b = FVector2::new(1.0, 2.0);

        let ref_time = SystemTime::now();

        let ne = a != b;

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector2::ne | {span} ns | {a} != {b} = {ne}");
    }

    {
        let ref_time = SystemTime::now();
        let distance_sq = a.distance_sq(b);

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector2::distance_sq | {span} ns | {a} {b} = {distance_sq}");
    }
}
