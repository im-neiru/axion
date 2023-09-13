// Imports from core
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Imports from std
use std::mem::MaybeUninit;
use std::ops;

use crate::math::FVector3;

union UnionCast {
    v3: (FVector3, f32),
    m128: __m128,
}

impl FVector3 {
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
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (other, 0.0) }.m128;

            let mut dot_product: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, b);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);

            _mm_store_ss(&mut dot_product, sum);

            dot_product
        }
    }

    /// Calculates the cross product of two `FVector3` instances.
    ///
    /// This method computes the cross product between `self` and `other`. The result is another
    /// `FVector3` that is orthogonal to both input vectors, representing the direction of the cross
    /// product. The cross product is commonly used for various geometric calculations.
    ///
    /// # Arguments
    ///
    /// * `other` - The `FVector3` instance representing the other vector.
    ///
    /// # Returns
    ///
    /// The cross product of `self` and `other` as a new `FVector3`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(1.0, 0.0, 0.0);
    /// let vector2 = FVector3::new(0.0, 1.0, 0.0);
    /// let cross_product = vector1.cross(vector2);
    ///
    /// assert_eq!(cross_product, FVector3::new(0.0, 0.0, 1.0));
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn cross(self, other: Self) -> Self {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (other, 0.0) }.m128;

            let a_yzx = _mm_shuffle_ps(a, a, super::mm_shuffle(3, 0, 2, 1));
            let b_yzx = _mm_shuffle_ps(b, b, super::mm_shuffle(3, 0, 2, 1));
            let c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));

            UnionCast {
                m128: _mm_shuffle_ps(c, c, super::mm_shuffle(3, 0, 2, 1)),
            }
            .v3
            .0
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
            let a = UnionCast { v3: (self, 0.0) }.m128;

            let mut length: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
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
            let a = UnionCast { v3: (self, 0.0) }.m128;

            let mut length_sq: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);

            _mm_store_ss(&mut length_sq, sum);

            length_sq
        }
    }

    //// Returns the inverse of the length (reciprocal of the magnitude) of the vector,
    /// If the length of the vector is `FVector3::ZERO`, this function will return an `f32::INFINITY`.
    ///
    ///
    /// The inverse length is calculated by taking the reciprocal of the length of the vector (`length` function).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// // Create a 3D vector
    /// let v = FVector3::new(3.0, 4.0, 5.0);
    ///
    /// // Calculate the inverse length of the 3D vector
    /// let inv_length = v.length_inv();
    ///
    /// // Print the result
    /// println!("Inverse Length: {}", inv_length); // prints: 0.1924500897298753
    /// ```

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn length_inv(self) -> f32 {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;

            let mut length_inv: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
            let root = _mm_sqrt_ps(sum);
            let recip = _mm_rcp_ps(root);

            _mm_store_ss(&mut length_inv, recip);

            length_inv
        }
    }

    /// Calculates the Euclidean distance between two 3D vectors.
    ///
    /// This function computes the Euclidean distance between two `FVector3`
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
    /// ```
    /// use axion::math::FVector3;
    ///
    /// // Create two 3D vectors
    /// let vector1 = FVector3::new(1.0, 2.0, 3.0);
    /// let vector2 = FVector3::new(4.0, 6.0, 8.0);
    ///
    /// // Calculate the distance between the two vectors
    /// let distance = vector1.distance(vector2);
    ///
    /// // Print the result
    /// println!("Distance between vector1 and vector2: {}", distance);
    /// ```

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn distance(self, other: Self) -> f32 {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (other, 0.0) }.m128;

            let mut distance: f32 = MaybeUninit::uninit().assume_init();
            let sub = _mm_sub_ps(a, b);
            let product = _mm_mul_ps(sub, sub);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
            let root = _mm_sqrt_ps(sum);

            _mm_store_ss(&mut distance, root);

            distance
        }
    }

    /// Computes the squared Euclidean distance between two `FVector3` instances.
    ///
    /// This method calculates the squared Euclidean distance between `self` and `other`, which is a
    /// more efficient version of the Euclidean distance as it avoids the square root operation.
    ///
    /// # Arguments
    ///
    /// * `other` - The `FVector3` instance representing the other point in 3D space.
    ///
    /// # Returns
    ///
    /// The squared Euclidean distance between `self` and `other` as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let point1 = FVector3::new(2.0, 3.0, 1.0);
    /// let point2 = FVector3::new(1.0, 2.0, 3.0);
    /// let distance_sq = point1.distance_sq(point2);
    ///
    /// // Check if the squared distance is approximately 10.0 within a small tolerance
    /// assert!((distance_sq - 10.0).abs() < 1e-6);
    /// ```

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn distance_sq(self, other: Self) -> f32 {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (other, 0.0) }.m128;

            let mut distance_sq: f32 = MaybeUninit::uninit().assume_init();
            let sub = _mm_sub_ps(a, b);
            let product = _mm_mul_ps(sub, sub);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);

            _mm_store_ss(&mut distance_sq, sum);

            distance_sq
        }
    }

    /// The `normalize` function normalizes a vector using SIMD instructions for efficiency.
    /// Be cautious when using this function with vectors at the origin `(0.0, 0.0, 0.0)`, as this would lead
    /// to a division by zero when calculating the reciprocal of the root. The result in such cases
    /// would be ``FVector3::NAN``. It is recommended to check if the vector is at the origin before
    /// calling this function.
    ///
    /// # Arguments
    ///
    /// * `self` - A FVector3 to be normalized.
    ///
    /// # Returns
    ///
    /// * `FVector3` - A new FVector3 that is the normalized version of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let v = FVector3::new(3.0, 4.0, 5.0); // Create a 3D vector
    /// let normalized_v = v.normalize(); // Normalize the 3D vector
    /// ```
    /// The `normalize` function normalizes a vector using SIMD instructions for efficiency.
    /// Be cautious when using this function with vectors at the origin `(0.0, 0.0, 0.0)`, as this would lead
    /// to a division by zero when calculating the reciprocal of the root. The result in such cases
    /// would be ``FVector3::NAN``. It is recommended to check if the vector is at the origin before
    /// calling this function.
    ///
    /// # Arguments
    ///
    /// * `self` - A FVector3 to be normalized.
    ///
    /// # Returns
    ///
    /// * `FVector3` - A new FVector3 that is the normalized version of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let v = FVector3::new(3.0, 4.0, 5.0); // Create a 3D vector
    /// let normalized_v = v.normalize(); // Normalize the 3D vector
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn normalize(self) -> Self {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let mut normalized_vec: (Self, Self) =
                MaybeUninit::uninit().assume_init();

            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
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

    /// Performs linear interpolation between two `FVector3` instances.
    ///
    /// Linear interpolation, or lerp, blends between two `FVector3` instances using a specified
    /// interpolation factor `scalar`. The result is a new `FVector3` where each component is interpolated
    /// between the corresponding components of `self` and `end` based on `scalar`.
    ///
    /// # Arguments
    ///
    /// * `end` - The target `FVector3` to interpolate towards.
    /// * `scalar` - The interpolation factor, typically in the range `0.0`, `1.0`, where:
    ///   - `scalar = 0.0` returns `self`.
    ///   - `scalar = 1.0` returns `end`.
    ///   - Values in between interpolate linearly.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance resulting from the linear interpolation of `self` and `end` with factor `scalar`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector3;
    ///
    /// let start = FVector3::new(1.0, 2.0, 3.0);
    /// let end = FVector3::new(3.0, 4.0, 5.0);
    /// let interpolated = start.lerp(end, 0.5);
    ///
    /// assert_eq!(interpolated, FVector3::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    pub fn lerp(self, end: Self, scalar: f32) -> Self {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;

            let b = UnionCast { v3: (end, 0.0) }.m128;

            UnionCast {
                m128: _mm_add_ps(
                    a,
                    _mm_mul_ps(_mm_sub_ps(b, a), _mm_set1_ps(scalar)),
                ),
            }
            .v3
            .0
        }
    }

    /// Clamps each component of the `FVector3` to be within the specified range.
    ///
    /// This function takes an `FVector3` instance and ensures that each component is
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
    /// A new `FVector3` instance with each component clamped to the specified range.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector = FVector3::new(2.0, 3.0, 5.0); // Fixed: Three components
    /// let clamped = vector.clamp(FVector3::new(1.0, 2.0, 3.0), FVector3::new(4.0, 5.0, 6.0)); // Fixed: Three components
    /// assert_eq!(clamped, FVector3::new(2.0, 3.0, 5.0));
    /// ```
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Returns an `FVector3` with each component set to the minimum of the corresponding components of `self` and `value`.
    ///
    /// This function compares each component of `self` and `value` and returns a new `FVector3`
    /// with each component set to the minimum of the corresponding components of `self` and `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The `FVector3` to compare with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance with each component set to the minimum of the corresponding components of `self` and `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(2.0, 3.0, 5.0); // Create a 3D vector
    /// let vector2 = FVector3::new(1.0, 2.0, 4.0); // Create another 3D vector
    /// let min_vector = vector1.min(vector2); // Find the component-wise minimum
    ///
    /// // The resulting vector should have each component as the minimum of the corresponding components of vector1 and vector2
    /// assert_eq!(min_vector, FVector3::new(1.0, 2.0, 4.0));
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn min(self, value: Self) -> Self {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;

            let b = UnionCast { v3: (value, 0.0) }.m128;

            UnionCast {
                m128: _mm_min_ps(a, b),
            }
            .v3
            .0
        }
    }

    /// Returns an `FVector3` with each component set to the maximum of the corresponding components of `self` and `value`.
    ///
    /// This function compares each component of `self` and `value` and returns a new `FVector3`
    /// with each component set to the maximum of the corresponding components of `self` and `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The `FVector3` to compare with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance with each component set to the maximum of the corresponding components of `self` and `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(2.0, 5.0, 8.0); // Create a 3D vector
    /// let vector2 = FVector3::new(1.0, 3.0, 10.0); // Create another 3D vector
    /// let max_vector = vector1.max(vector2); // Find the maximum component-wise
    ///
    /// assert_eq!(max_vector, FVector3::new(2.0, 5.0, 10.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn max(self, value: Self) -> Self {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;

            let b = UnionCast { v3: (value, 0.0) }.m128;

            UnionCast {
                m128: _mm_max_ps(a, b),
            }
            .v3
            .0
        }
    }

    /// Checks if all components of the `FVector3` are equal to zero.
    ///
    /// This function compares each component of the `FVector3` with zero and returns
    /// `true` if all components are equal to zero, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// Returns `true` if all components of the `FVector3` are equal to zero, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// // Create an `FVector3` with all components set to zero
    /// let zero_vector = FVector3::ZERO;
    ///
    /// // Create an `FVector3` with some non-zero components
    /// let non_zero_vector = FVector3::new(2.0, 0.0, 0.0);
    ///
    /// // Check if all components of the zero vector are equal to zero
    /// assert_eq!(zero_vector.is_zero(), true);
    ///
    /// // Check if all components of the non-zero vector are equal to zero
    /// assert_eq!(non_zero_vector.is_zero(), false);
    /// ```

    #[inline]
    pub fn is_zero(self) -> bool {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;

            let b = UnionCast {
                v3: (Self::ZERO, 0.0),
            }
            .m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) == 0xf
        }
    }

    /// Checks if any component of the `FVector3` is `f32::NAN`.
    ///
    /// This function checks each component of the `FVector3` for `f32::NAN` and returns `true`
    /// if at least one component is `f32::NAN`, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if any component of the `FVector3` is `f32::NAN`, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let nan_vector = FVector3::new(f32::NAN, 2.0, 1.0); // Create a 3D vector with NaN component
    /// let valid_vector = FVector3::new(1.0, 3.0, 2.0);     // Create a 3D vector with valid components
    ///
    /// assert_eq!(nan_vector.is_nan(), true);    // Check if any component of the nan_vector is NaN
    /// assert_eq!(valid_vector.is_nan(), false); // Check if any component of the valid_vector is NaN
    /// ```

    #[inline]
    pub fn is_nan(self) -> bool {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;

            let cmp_result = _mm_cmpunord_ps(a, a);

            _mm_movemask_ps(cmp_result) != 0
        }
    }
}

impl PartialEq for FVector3 {
    /// Checks if two `FVector3` instances are equal.
    ///
    /// This implementation compares each component of two `FVector3` instances for equality
    /// and returns `true` if all components are equal, and `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to compare with.
    ///
    /// # Returns
    ///
    /// `true` if all components of the two `FVector3` instances are equal, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector3::new(2.0, 3.0, 4.0); // Create another 3D vector with the same values
    /// let vector3 = FVector3::new(1.0, 2.0, 3.0); // Create a different 3D vector
    ///
    /// assert_eq!(vector1, vector2); // Check if vector1 is equal to vector2
    /// assert_ne!(vector1, vector3); // Check if vector1 is not equal to vector3
    /// ```
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let a = UnionCast { v3: (*self, 0.0) }.m128;
            let b = UnionCast { v3: (*rhs, 0.0) }.m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) == 0xf
        }
    }

    /// Checks if two `FVector3` instances are not equal.
    ///
    /// This implementation compares each component of two `FVector3` instances for inequality
    /// and returns `true` if at least one component is not equal, and `false` if all components
    /// are equal.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to compare with.
    ///
    /// # Returns
    ///
    /// `true` if at least one component of the two `FVector3` instances is not equal, `false` if all components are equal.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector3::new(1.0, 2.0, 4.0); // Create another 3D vector with differences in components
    ///
    /// assert_ne!(vector1, vector2); // Check if vector1 is not equal to vector2 due to differing components
    /// ```
    #[inline]
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, rhs: &Self) -> bool {
        unsafe {
            let a = UnionCast { v3: (*self, 0.0) }.m128;
            let b = UnionCast { v3: (*rhs, 0.0) }.m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) != 0xf
        }
    }
}

impl ops::Add<Self> for FVector3 {
    type Output = Self;

    /// Adds two `FVector3` instances component-wise.
    ///
    /// This implementation allows you to add two `FVector3` instances together
    /// component-wise, resulting in a new `FVector3` where each component is the
    /// sum of the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance resulting from the component-wise addition of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector3::new(1.0, 2.0, 1.0); // Create another 3D vector
    /// let result = vector1 + vector2; // Perform component-wise addition
    ///
    /// assert_eq!(result, FVector3::new(3.0, 5.0, 5.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (rhs, 0.0) }.m128;

            UnionCast {
                m128: _mm_add_ps(a, b),
            }
            .v3
            .0
        }
    }
}

impl ops::Sub<Self> for FVector3 {
    type Output = Self;

    /// Subtracts one `FVector3` from another component-wise.
    ///
    /// This implementation allows you to subtract one `FVector3` from another
    /// component-wise, resulting in a new `FVector3` where each component is the
    /// difference between the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to subtract from `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance resulting from the component-wise subtraction of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector3::new(1.0, 2.0, 1.0); // Create another 3D vector
    /// let result = vector1 - vector2; // Perform component-wise subtraction
    ///
    /// assert_eq!(result, FVector3::new(1.0, 1.0, 3.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (rhs, 0.0) }.m128;

            UnionCast {
                m128: _mm_sub_ps(a, b),
            }
            .v3
            .0
        }
    }
}

impl ops::Mul<Self> for FVector3 {
    type Output = Self;

    /// Multiplies two `FVector3` instances component-wise.
    ///
    /// This implementation allows you to multiply two `FVector3` instances together
    /// component-wise, resulting in a new `FVector3` where each component is the
    /// product of the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance resulting from the component-wise multiplication of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector3::new(1.0, 2.0, 2.0); // Create another 3D vector
    /// let result = vector1 * vector2; // Perform component-wise multiplication
    ///
    /// assert_eq!(result, FVector3::new(2.0, 6.0, 8.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (rhs, 0.0) }.m128;

            UnionCast {
                m128: _mm_mul_ps(a, b),
            }
            .v3
            .0
        }
    }
}

impl ops::Div<Self> for FVector3 {
    type Output = Self;

    /// Divides one `FVector3` by another component-wise.
    ///
    /// This implementation allows you to divide one `FVector3` by another component-wise,
    /// resulting in a new `FVector3` where each component is the result of dividing the
    /// corresponding components of `self` by `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to divide `self` by.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance resulting from the component-wise division of `self` by `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(2.0, 6.0, 8.0); // Create a 3D vector
    /// let vector2 = FVector3::new(1.0, 2.0, 4.0); // Create another 3D vector
    /// let result = vector1 / vector2; // Perform component-wise division
    ///
    /// assert_eq!(result, FVector3::new(2.0, 3.0, 2.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (rhs, 0.0) }.m128;

            UnionCast {
                m128: _mm_div_ps(a, b),
            }
            .v3
            .0
        }
    }
}

impl ops::Rem<Self> for FVector3 {
    type Output = Self;

    /// Computes the remainder of dividing one `FVector3` by another component-wise.
    ///
    /// This implementation allows you to compute the remainder of dividing one `FVector3`
    /// by another component-wise, resulting in a new `FVector3` where each component is
    /// the remainder of dividing the corresponding components of `self` by `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to compute the remainder of division with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance resulting from the component-wise remainder of division of `self` by `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector1 = FVector3::new(7.0, 10.0, 13.0); // Create a 3D vector
    /// let vector2 = FVector3::new(3.0, 4.0, 5.0);   // Create another 3D vector
    /// let result = vector1 % vector2;                // Perform component-wise remainder
    ///
    /// assert_eq!(result, FVector3::new(1.0, 2.0, 3.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v3: (self, 0.0) }.m128;
            let b = UnionCast { v3: (rhs, 0.0) }.m128;

            UnionCast {
                m128: _mm_sub_ps(
                    a,
                    _mm_mul_ps(b, _mm_floor_ps(_mm_div_ps(a, b))),
                ),
            }
            .v3
            .0
        }
    }
}

impl ops::AddAssign<Self> for FVector3 {
    /// Adds another `FVector3` to `self` in place.
    ///
    /// This implementation allows you to add another `FVector3` to `self` in place, modifying
    /// `self` to be the result of the addition.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to add to `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let mut vector = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// vector += FVector3::new(1.0, 2.0, 1.0);        // Perform in-place addition
    ///
    /// assert_eq!(vector, FVector3::new(3.0, 5.0, 5.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ops::SubAssign<Self> for FVector3 {
    /// Subtracts another `FVector3` from `self` in place.
    ///
    /// This implementation allows you to subtract another `FVector3` from `self` in place,
    /// modifying `self` to be the result of the subtraction.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to subtract from `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let mut vector = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// vector -= FVector3::new(1.0, 2.0, 1.0);        // Perform in-place subtraction
    ///
    /// assert_eq!(vector, FVector3::new(1.0, 1.0, 3.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ops::MulAssign<Self> for FVector3 {
    /// Multiplies `self` by another `FVector3` in place.
    ///
    /// This implementation allows you to multiply `self` by another `FVector3` in place,
    /// modifying `self` to be the result of the multiplication.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to multiply `self` with.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let mut vector = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// vector *= FVector3::new(1.0, 2.0, 2.0);        // Perform in-place multiplication
    ///
    /// assert_eq!(vector, FVector3::new(2.0, 6.0, 8.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl ops::DivAssign<Self> for FVector3 {
    /// Divides `self` by another `FVector3` in place.
    ///
    /// This implementation allows you to divide `self` by another `FVector3` in place,
    /// modifying `self` to be the result of the division.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to divide `self` by.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let mut vector = FVector3::new(2.0, 6.0, 8.0); // Create a 3D vector
    /// vector /= FVector3::new(1.0, 2.0, 4.0);        // Perform in-place division
    ///
    /// assert_eq!(vector, FVector3::new(2.0, 3.0, 2.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl ops::RemAssign<Self> for FVector3 {
    /// Computes the remainder of dividing `self` by another `FVector3` in place.
    ///
    /// This implementation allows you to compute the remainder of dividing `self` by another
    /// `FVector3` in place, modifying `self` to be the result of the remainder operation.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector3` to compute the remainder of division with `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let mut vector = FVector3::new(7.0, 10.0, 12.0); // Create a 3D vector
    /// vector %= FVector3::new(3.0, 4.0, 5.0);           // Perform in-place remainder
    ///
    /// assert_eq!(vector, FVector3::new(1.0, 2.0, 2.0));  // Check the result with 3D vectors
    /// ```
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

#[test]
fn test_fvector3_sse3() {
    use std::time::SystemTime;

    let a = FVector3::new(3.0, 4.0, 2.0);
    let b = FVector3::new(4.0, 3.2, 5.0);
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
    let is_zero = FVector3::ZERO.is_zero();

    let span_is_zero = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let is_nan = FVector3::NAN.is_nan();

    let span_is_nan = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let is_finite = FVector3::INFINITY.is_finite();

    let span_is_finite = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    // assert!(dot == 34.8, "FVector3::dot | Wrong output");
    // assert!(length == 5.0, "FVector3::length | Wrong output");

    println!("FVector3::dot | {span_dot} ns | {a} ⋅ {b} = {dot}",);

    println!("FVector3::length | {span_length} ns | {a} = {length}",);

    println!("FVector3::normalize | {span_normalize} ns | {a} = {normalize}",);

    println!("FVector3::distance | {span_distance} ns | {a} {b} = {distance}",);

    println!(
        "FVector3::is_nan | {span_is_nan} ns | {}  = {is_nan}",
        FVector3::NAN,
    );

    println!(
        "FVector3::is_nan | {span_is_zero} ns | {}  = {is_zero}",
        FVector3::ZERO,
    );

    println!(
        "FVector3::is_finite | {span_is_finite} ns | {}  = {is_finite}",
        FVector3::INFINITY,
    );

    {
        let start = FVector3::new(1.0, 2.0, 1.0);
        let end = FVector3::new(3.0, 4.0, 1.0);

        let ref_time = SystemTime::now();

        let lerp = start.lerp(end, 0.5);

        let span_lerp = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector3::lerp | {span_lerp} ns | {start} {end} = {lerp}");
    }

    {
        let a = FVector3::new(1.0, 2.0, 1.0);
        let b = FVector3::new(1.0, 2.0, 1.0);

        let ref_time = SystemTime::now();

        let eq = a == b;

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector3::eq | {span} ns | {a} == {b} = {eq}");
    }

    {
        let a = FVector3::new(1.0, 2.0, 1.0);
        let b = FVector3::new(1.0, 2.0, 1.0);

        let ref_time = SystemTime::now();

        let ne = a != b;

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector3::ne | {span} ns | {a} != {b} = {ne}");
    }

    {
        let ref_time = SystemTime::now();
        let distance_sq = a.distance_sq(b);

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector3::distance_sq | {span} ns | {a} {b} = {distance_sq}");
    }

    {
        let a = FVector3::new(1.0, 2.0, -1.0);
        let b = FVector3::new(1.0, 2.0, 3.0);

        let ref_time = SystemTime::now();

        let cross = a.cross(b);

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector3::cross | {span} ns | {a} × {b} = {cross}");
    }
}
