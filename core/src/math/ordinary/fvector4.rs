use std::ops;

use crate::math::FVector4;

impl FVector4 {
    /// Returns the dot product of the vector and another vector.
    ///
    /// # Arguments
    ///
    /// * `other` - Another vector to calculate the dot product with.
    ///
    /// # Returns
    ///
    /// The dot product of the two vectors as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let vector2 = FVector4::new(4.0, 3.0, 2.0, 1.0);
    /// let dot_product = vector1.dot(vector2);
    ///
    /// assert_eq!(dot_product, 20.0);
    /// ```
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x.mul_add(
            other.x,
            self.y
                .mul_add(other.y, self.z.mul_add(other.z, self.w * other.w)),
        )
    }

    /// Returns the length (magnitude) of the vector.
    ///
    /// The length of a vector is a non-negative number that describes the extent of the vector in space.
    ///
    /// # Returns
    ///
    /// The length of the vector as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(3.0, 4.0, 5.0, 0.0);
    /// let length = vector.length();
    ///
    /// assert_eq!(length, 7.071068);
    /// ```
    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    /// Returns the squared length of the vector.
    ///
    /// This function is often used when comparing lengths, as it avoids the computationally
    /// expensive square root operation.
    ///
    /// # Returns
    ///
    /// The squared length of the vector as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(3.0, 4.0, 5.0, 0.0);
    /// let length_sq = vector.length_sq();
    ///
    /// assert_eq!(length_sq, 50.0);
    /// ```
    #[inline]
    pub fn length_sq(self) -> f32 {
        self.x.mul_add(
            self.x,
            self.y
                .mul_add(self.y, self.z.mul_add(self.z, self.w * self.w)),
        )
    }

    /// Returns the inverse of the length (reciprocal of the magnitude) of the vector.
    ///
    /// If the length of the vector is `FVector4::ZERO`, this function will return `f32::INFINITY`.
    ///
    /// # Returns
    ///
    /// The inverse of the length of the vector as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(3.0, 4.0, 5.0, 0.0);
    /// let inv_length = vector.length_inv();
    ///
    /// assert_eq!(inv_length, 0.14142136);
    /// ```
    #[inline]
    pub fn length_inv(self) -> f32 {
        self.length_sq().sqrt().recip()
    }

    /// Calculates the Euclidean distance between two `FVector4` instances.
    ///
    /// This function computes the Euclidean distance between two `FVector4` instances.
    ///
    /// # Arguments
    ///
    /// * `other` - The second vector to which the distance is calculated.
    ///
    /// # Returns
    ///
    /// The calculated Euclidean distance between the two input vectors as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(1.0, 2.0, 3.0, 0.0);
    /// let vector2 = FVector4::new(4.0, 6.0, 8.0, 0.0);
    /// let distance = vector1.distance(vector2);
    ///
    /// assert_eq!(distance, 7.071068);
    /// ```
    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        self.distance_sq(other).sqrt()
    }

    /// Computes the squared Euclidean distance between two `FVector4` instances.
    ///
    /// This method calculates the squared Euclidean distance between `self` and `other`, which is a
    /// more efficient version of the Euclidean distance as it avoids the square root operation.
    ///
    /// # Arguments
    ///
    /// * `other` - The `FVector4` instance representing the other point in 4D space.
    ///
    /// # Returns
    ///
    /// The squared Euclidean distance between `self` and `other` as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let point1 = FVector4::new(2.0, 3.0, 1.0);
    /// let point2 = FVector4::new(1.0, 2.0, 3.0);
    /// let distance_sq = point1.distance_sq(point2);
    ///
    /// // Check if the squared distance is approximately 10.0 within a small tolerance
    /// assert!((distance_sq - 10.0).abs() < 1e-6);
    /// ```

    #[inline]
    pub fn distance_sq(self, other: Self) -> f32 {
        let x_diff = self.x - other.x;
        let y_diff = self.y - other.y;
        let z_diff = self.z - other.z;
        let w_diff = self.w - other.w;

        x_diff.mul_add(
            x_diff,
            y_diff.mul_add(y_diff, z_diff.mul_add(z_diff, w_diff * w_diff)),
        )
    }

    /// The `normalize` function normalizes a vector using SIMD instructions for efficiency.
    /// Be cautious when using this function with vectors at the origin `(0.0, 0.0, 0.0)`, as this would lead
    /// to a division by zero when calculating the reciprocal of the root. The result in such cases
    /// would be ``FVector4::NAN``. It is recommended to check if the vector is at the origin before
    /// calling this function.
    ///
    /// # Arguments
    ///
    /// * `self` - A FVector4 to be normalized.
    ///
    /// # Returns
    ///
    /// * `FVector4` - A new FVector4 that is the normalized version of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let v = FVector4::new(3.0, 4.0, 5.0); // Create a 3D vector
    /// let normalized_v = v.normalize(); // Normalize the 3D vector
    /// ```
    /// The `normalize` function normalizes a vector using SIMD instructions for efficiency.
    /// Be cautious when using this function with vectors at the origin `(0.0, 0.0, 0.0)`, as this would lead
    /// to a division by zero when calculating the reciprocal of the root. The result in such cases
    /// would be ``FVector4::NAN``. It is recommended to check if the vector is at the origin before
    /// calling this function.
    ///
    /// # Arguments
    ///
    /// * `self` - A FVector4 to be normalized.
    ///
    /// # Returns
    ///
    /// * `FVector4` - A new FVector4 that is the normalized version of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let v = FVector4::new(3.0, 4.0, 5.0); // Create a 3D vector
    /// let normalized_v = v.normalize(); // Normalize the 3D vector
    /// ```
    #[inline]
    pub fn normalize(self) -> Self {
        let length_inv = self.length_inv();

        Self {
            x: self.x * length_inv,
            y: self.y * length_inv,
            z: self.z * length_inv,
            w: self.w * length_inv,
        }
    }

    /// Project the vector onto another vector.
    ///
    /// This function calculates the projection of the current vector onto
    /// the `normal` vector. To ensure accurate results, normalize the `normal`
    /// vector before passing it to this function. Failure to normalize `normal`
    /// may lead to incorrect or undesirable results.
    ///
    /// # Arguments
    ///
    /// - `self`: The vector to be projected.
    /// - `normal`: The normalized vector onto which the projection is made.
    ///
    /// # Returns
    ///
    /// A new `FVector4` representing the projection of the current vector onto `normal`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vec1 = FVector4::new(1.0, 2.0, 3.0, 5.0);
    /// let normalized = FVector4::new(4.0, 5.0, 6.0, 5.0).normalize();
    ///
    /// let projection = vec1.project(normalized);
    /// println!("Projection: {:?}", projection);
    /// ```
    ///
    /// Note: Ensure that the `normal` vector is normalized before passing it to this function.
    #[inline]
    pub fn project(self, normal: Self) -> Self {
        let dot_product = self.dot(normal);

        Self {
            x: dot_product * normal.x,
            y: dot_product * normal.y,
            z: dot_product * normal.z,
            w: dot_product * normal.w,
        }
    }

    /// Performs linear interpolation between two `FVector4` instances.
    ///
    /// Linear interpolation, or lerp, blends between two `FVector4` instances using a specified
    /// interpolation factor `scalar`. The result is a new `FVector4` where each component is interpolated
    /// between the corresponding components of `self` and `end` based on `scalar`.
    ///
    /// # Arguments
    ///
    /// * `end` - The target `FVector4` to interpolate towards.
    /// * `scalar` - The interpolation factor, typically in the range `0.0`, `1.0`, where:
    ///   - `scalar = 0.0` returns `self`.
    ///   - `scalar = 1.0` returns `end`.
    ///   - Values in between interpolate linearly.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the linear interpolation of `self` and `end` with factor `scalar`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let start = FVector4::new(1.0, 2.0, 3.0, 0.0);
    /// let end = FVector4::new(3.0, 4.0, 5.0, 0.0);
    /// let interpolated = start.lerp(end, 0.5);
    ///
    /// assert_eq!(interpolated, FVector4::new(2.0, 3.0, 4.0, 0.0));
    /// ```
    #[inline]
    pub fn lerp(self, end: Self, scalar: f32) -> Self {
        let x = self.x - end.x;
        let y = self.y - end.y;
        let z = self.z - end.z;
        let w = self.w - end.w;

        Self {
            x: x.mul_add(scalar, self.x),
            y: y.mul_add(scalar, self.y),
            z: z.mul_add(scalar, self.z),
            w: w.mul_add(scalar, self.w),
        }
    }

    /// Clamps each component of the vector to a specified range.
    ///
    /// This method clamps each component of the vector to the specified `min` and `max` values. If a component is less
    /// than `min`, it is set to `min`. If it is greater than `max`, it is set to `max`. Otherwise, it remains unchanged.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value for each component.
    /// * `max` - The maximum value for each component.
    ///
    /// # Returns
    ///
    /// A new `FVector4` with components clamped to the `min` and `max` range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(2.0, 3.0, 1.0, 5.0);
    /// let clamped = vector.clamp(FVector4::new(1.0, 2.0, 1.0, 3.0), FVector4::new(3.0, 4.0, 2.0, 4.0));
    ///
    /// assert_eq!(clamped, FVector4::new(2.0, 3.0, 2.0, 4.0));
    /// ```
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Calculates the element-wise minimum between the vector and another vector.
    ///
    /// This method computes the element-wise minimum between each component of the vector and the corresponding component
    /// of the `value` vector. The result is a new `FVector4` where each component is the minimum of the original vector
    /// and the `value` vector.
    ///
    /// # Arguments
    ///
    /// * `value` - Another `FVector4` to calculate the element-wise minimum with.
    ///
    /// # Returns
    ///
    /// A new `FVector4` with components set to the minimum of the original vector and the `value` vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(2.0, 3.0, 1.0, 5.0);
    /// let min_vector = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let result = vector.min(min_vector);
    ///
    /// assert_eq!(result, FVector4::new(1.0, 2.0, 1.0, 4.0));
    /// ```
    #[inline]
    pub fn min(self, value: Self) -> Self {
        Self {
            x: self.x.min(value.x),
            y: self.y.min(value.y),
            z: self.z.min(value.z),
            w: self.w.min(value.w),
        }
    }

    /// Calculates the element-wise maximum between the vector and another vector.
    ///
    /// This method computes the element-wise maximum between each component of the vector and the corresponding component
    /// of the `value` vector. The result is a new `FVector4` where each component is the maximum of the original vector
    /// and the `value` vector.
    ///
    /// # Arguments
    ///
    /// * `value` - Another `FVector4` to calculate the element-wise maximum with.
    ///
    /// # Returns
    ///
    /// A new `FVector4` with components set to the maximum of the original vector and the `value` vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(2.0, 3.0, 1.0, 5.0);
    /// let max_vector = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let result = vector.max(max_vector);
    ///
    /// assert_eq!(result, FVector4::new(2.0, 3.0, 3.0, 5.0));
    /// ```
    #[inline]
    pub fn max(self, value: Self) -> Self {
        Self {
            x: self.x.max(value.x),
            y: self.y.max(value.y),
            z: self.z.max(value.z),
            w: self.w.max(value.w),
        }
    }

    /// Checks if all components of the vector are equal to zero.
    ///
    /// This method returns `true` if all components of the vector are exactly zero, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if all components of the vector are zero; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let zero_vector = FVector4::new(0.0, 0.0, 0.0, 0.0);
    /// let non_zero_vector = FVector4::new(1.0, 0.0, 0.0, 0.0);
    ///
    /// assert_eq!(zero_vector.is_zero(), true);
    /// assert_eq!(non_zero_vector.is_zero(), false);
    /// ```
    #[inline]
    pub fn is_zero(self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0 && self.w == 0.0
    }

    /// Checks if any component of the vector is NaN (Not-a-Number).
    ///
    /// This method returns `true` if any component of the vector is NaN (Not-a-Number), and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if any component of the vector is NaN; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let nan_vector = FVector4::new(1.0, f32::NAN, 3.0, 4.0);
    /// let non_nan_vector = FVector4::new(2.0, 3.0, 4.0, 5.0);
    ///
    /// assert_eq!(nan_vector.is_nan(), true);
    /// assert_eq!(non_nan_vector.is_nan(), false);
    /// ```
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() && self.y.is_nan() && self.z.is_nan() && self.w.is_nan()
    }
}

impl PartialEq for FVector4 {
    /// Checks if two `FVector4` instances are equal.
    ///
    /// This implementation compares each component of two `FVector4` instances for equality
    /// and returns `true` if all components are equal, and `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to compare with.
    ///
    /// # Returns
    ///
    /// `true` if all components of the two `FVector4` instances are equal, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector4::new(2.0, 3.0, 4.0); // Create another 3D vector with the same values
    /// let vector3 = FVector4::new(1.0, 2.0, 3.0); // Create a different 3D vector
    ///
    /// assert_eq!(vector1, vector2); // Check if vector1 is equal to vector2
    /// assert_ne!(vector1, vector3); // Check if vector1 is not equal to vector3
    /// ```
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z && self.w == rhs.w
    }

    /// Checks if two `FVector4` instances are not equal.
    ///
    /// This implementation compares each component of two `FVector4` instances for inequality
    /// and returns `true` if at least one component is not equal, and `false` if all components
    /// are equal.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to compare with.
    ///
    /// # Returns
    ///
    /// `true` if at least one component of the two `FVector4` instances is not equal, `false` if all components are equal.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector4::new(1.0, 2.0, 4.0); // Create another 3D vector with differences in components
    ///
    /// assert_ne!(vector1, vector2); // Check if vector1 is not equal to vector2 due to differing components
    /// ```
    #[inline]
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, rhs: &Self) -> bool {
        self.x != rhs.x || self.y != rhs.y || self.z != rhs.z || self.w != rhs.w
    }
}

impl ops::Add<Self> for FVector4 {
    type Output = Self;

    /// Adds two `FVector4` instances component-wise.
    ///
    /// This method performs component-wise addition of two `FVector4` instances, resulting in a new `FVector4` where each component
    /// is the sum of the corresponding components from `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The second `FVector4` to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the addition of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let vector2 = FVector4::new(2.0, 3.0, 4.0, 5.0);
    /// let result = vector1 + vector2;
    ///
    /// assert_eq!(result, FVector4::new(3.0, 5.0, 7.0, 9.0));
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl ops::Sub<Self> for FVector4 {
    type Output = Self;

    /// Subtracts one `FVector4` from another component-wise.
    ///
    /// This implementation allows you to subtract one `FVector4` from another
    /// component-wise, resulting in a new `FVector4` where each component is the
    /// difference between the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to subtract from `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the component-wise subtraction of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector4::new(1.0, 2.0, 1.0); // Create another 3D vector
    /// let result = vector1 - vector2; // Perform component-wise subtraction
    ///
    /// assert_eq!(result, FVector4::new(1.0, 1.0, 3.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl ops::Mul<Self> for FVector4 {
    type Output = Self;

    /// Multiplies two `FVector4` instances component-wise.
    ///
    /// This method performs component-wise multiplication of two `FVector4` instances, resulting in a new `FVector4` where each component
    /// is the product of the corresponding components from `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The second `FVector4` to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the multiplication of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0, 5.0);
    /// let vector2 = FVector4::new(3.0, 2.0, 1.0, 2.0);
    /// let result = vector1 * vector2;
    ///
    /// assert_eq!(result, FVector4::new(6.0, 6.0, 4.0, 10.0));
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
            w: self.w * rhs.w,
        }
    }
}

impl ops::Div<Self> for FVector4 {
    type Output = Self;

    /// Divides two `FVector4` instances component-wise.
    ///
    /// This method performs component-wise division of `self` by `rhs`, resulting in a new `FVector4` where each component
    /// is the quotient of the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` to divide `self` by.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the division of `self` by `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(6.0, 6.0, 4.0, 10.0);
    /// let vector2 = FVector4::new(3.0, 2.0, 1.0, 2.0);
    /// let result = vector1 / vector2;
    ///
    /// assert_eq!(result, FVector4::new(2.0, 3.0, 4.0, 5.0));
    /// ```
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
            w: self.w / rhs.w,
        }
    }
}

impl ops::Rem<Self> for FVector4 {
    type Output = Self;

    /// Computes the remainder of dividing one `FVector4` by another component-wise.
    ///
    /// This implementation allows you to compute the remainder of dividing one `FVector4`
    /// by another component-wise, resulting in a new `FVector4` where each component is
    /// the remainder of dividing the corresponding components of `self` by `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to compute the remainder of division with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the component-wise remainder of division of `self` by `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(7.0, 10.0, 13.0); // Create a 3D vector
    /// let vector2 = FVector4::new(3.0, 4.0, 5.0);   // Create another 3D vector
    /// let result = vector1 % vector2;                // Perform component-wise remainder
    ///
    /// assert_eq!(result, FVector4::new(1.0, 2.0, 3.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x % rhs.x,
            y: self.y % rhs.y,
            z: self.z % rhs.z,
            w: self.w % rhs.w,
        }
    }
}

impl ops::AddAssign<Self> for FVector4 {
    /// Adds another `FVector4` to `self` in-place.
    ///
    /// This method performs component-wise addition of another `FVector4` to `self`, updating `self` with the result.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` to add to `self` in-place.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let mut vector1 = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let vector2 = FVector4::new(2.0, 4.0, 5.0, 6.0);
    ///
    /// vector1 += vector2;
    ///
    /// assert_eq!(vector1, FVector4::new(3.0, 6.0, 8.0, 10.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl ops::SubAssign<Self> for FVector4 {
    /// Subtracts another `FVector4` from `self` in-place.
    ///
    /// This method performs component-wise subtraction of another `FVector4` from `self`, updating `self` with the result.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` to subtract from `self` in-place.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let mut vector1 = FVector4::new(3.0, 6.0, 8.0, 10.0);
    /// let vector2 = FVector4::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// vector1 -= vector2;
    ///
    /// assert_eq!(vector1, FVector4::new(2.0, 4.0, 5.0, 6.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

impl ops::MulAssign<Self> for FVector4 {
    /// Multiplies `self` by another `FVector4` in place.
    ///
    /// This implementation allows you to multiply `self` by another `FVector4` in place,
    /// modifying `self` to be the result of the multiplication.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to multiply `self` with.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let mut vector = FVector4::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// vector *= FVector4::new(1.0, 2.0, 2.0);        // Perform in-place multiplication
    ///
    /// assert_eq!(vector, FVector4::new(2.0, 6.0, 8.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
            w: self.w * rhs.w,
        }
    }
}

impl ops::DivAssign<Self> for FVector4 {
    /// Calculates the component-wise remainder of `self` by another `FVector4`.
    ///
    /// This method performs component-wise remainder of `self` by another `FVector4`, producing a new `FVector4` with
    /// each component being the remainder of the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` by which to perform component-wise remainder.
    ///
    /// # Returns
    ///
    /// A new `FVector4` resulting from the component-wise remainder operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(10.0, 15.0, 8.0, 7.0);
    /// let vector2 = FVector4::new(3.0, 4.0, 3.0, 2.0);
    ///
    /// let result = vector1 % vector2;
    ///
    /// assert_eq!(result, FVector4::new(1.0, 3.0, 2.0, 1.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
            w: self.w / rhs.w,
        }
    }
}

impl ops::RemAssign<Self> for FVector4 {
    /// Calculates the component-wise remainder of `self` by another `FVector4` in-place.
    ///
    /// This method performs component-wise remainder of `self` by another `FVector4`, updating `self` with the result.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` by which to perform component-wise remainder.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let mut vector1 = FVector4::new(10.0, 15.0, 8.0, 7.0);
    /// let vector2 = FVector4::new(3.0, 4.0, 3.0, 2.0);
    ///
    /// vector1 %= vector2;
    ///
    /// assert_eq!(vector1, FVector4::new(1.0, 3.0, 2.0, 1.0));
    /// ```
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = Self {
            x: self.x % rhs.x,
            y: self.y % rhs.y,
            z: self.z % rhs.z,
            w: self.w % rhs.w,
        }
    }
}
