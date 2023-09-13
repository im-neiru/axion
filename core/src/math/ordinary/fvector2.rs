use std::ops;

use crate::math::FVector2;

impl FVector2 {
    /// Returns the dot product of the vector and another vector.
    ///
    /// # Arguments
    ///
    /// * `other` - Another vector to calculate the dot product with.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x.mul_add(other.x, self.y * other.y)
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
    pub fn length(self) -> f32 {
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
    pub fn length_sq(self) -> f32 {
        self.x.mul_add(self.x, self.y * self.y)
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
    pub fn length_inv(self) -> f32 {
        self.length_sq().sqrt().recip()
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
    pub fn distance(self, other: Self) -> f32 {
        self.distance_sq(other).sqrt()
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
    pub fn distance_sq(self, other: Self) -> f32 {
        let x = self.x - other.x;
        let y = self.y - other.y;

        x.mul_add(x, y * y)
    }

    /// The `normalize` function normalizes a vector.
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
    pub fn normalize(self) -> Self {
        let length_inv = self.length_inv();

        Self {
            x: self.x * length_inv,
            y: self.y * length_inv,
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
    /// A new `FVector2` representing the projection of the current vector onto `normal`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector2;
    ///
    /// let vec1 = FVector2::new(1.0, 2.0);
    /// let normalized = FVector2::new(4.0, 5.0).normalize();
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
        let x = self.x - end.x;
        let y = self.y - end.y;

        Self {
            x: x.mul_add(scalar, self.x),
            y: y.mul_add(scalar, self.y),
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
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
        }
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
    pub fn min(self, value: Self) -> Self {
        Self {
            x: self.x.min(value.x),
            y: self.y.min(value.y),
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
    pub fn max(self, value: Self) -> Self {
        Self {
            x: self.x.max(value.x),
            y: self.y.max(value.y),
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
        self.x == 0.0 && self.y == 0.0
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
        self.x.is_finite() && self.y.is_finite()
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
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
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
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
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
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
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
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
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
        Self {
            x: self.x % rhs.x,
            y: self.y % rhs.y,
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
        *self = Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
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
        *self = Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
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
        *self = Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
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
        *self = Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
        }
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
        *self = Self {
            x: self.x % rhs.x,
            y: self.y % rhs.y,
        }
    }
}
