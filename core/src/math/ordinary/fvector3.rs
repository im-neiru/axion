use std::ops;

#[derive(Clone, Copy, Debug)]
pub struct FVector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl FVector3 {
    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to 0.0.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to 0.0.
    pub const ZERO: Self = Self::splat(0.0);

    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to 1.0.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to 1.0.
    pub const ONE: Self = Self::splat(1.0);

    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to -1.0.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to -1.0.
    pub const NEG_ONE: Self = Self::splat(-1.0);

    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to the minimum finite value representable by `f32`.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to the minimum finite value representable by `f32`.
    pub const MIN: Self = Self::splat(f32::MIN);

    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to the maximum finite value representable by `f32`.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to the maximum finite value representable by `f32`.
    pub const MAX: Self = Self::splat(f32::MAX);

    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to a NaN (Not-a-Number) value.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to a NaN (Not-a-Number) value.
    pub const NAN: Self = Self::splat(f32::NAN);

    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to positive infinity.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to positive infinity.
    pub const INFINITY: Self = Self::splat(f32::INFINITY);

    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to negative infinity.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to negative infinity.
    pub const NEG_INFINITY: Self = Self::splat(f32::NEG_INFINITY);

    /// A constant `FVector3` instance with all `x`, `y` and `z` components set to the smallest positive value representable by `f32`.
    ///
    /// This constant represents an `FVector3` with all `x`, `y` and `z` components
    /// initialized to the smallest positive value representable by `f32`.
    pub const EPSILON: Self = Self::splat(f32::EPSILON);

    /// A constant `FVector3` representing the positive X-axis.
    ///
    /// This constant vector has a value of (1.0, 0.0, 0.0), representing the positive X-axis in 3D space.
    pub const AXIS_X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// A constant `FVector3` representing the positive Y-axis.
    ///
    /// This constant vector has a value of (0.0, 1.0, 0.0), representing the positive Y-axis in 3D space.
    pub const AXIS_Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// A constant `FVector3` representing the positive Z-axis.
    ///
    /// This constant vector has a value of (0.0, 0.0, 1.0), representing the positive Z-axis in 3D space.
    pub const AXIS_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// A constant `FVector3` representing the negative X-axis.
    ///
    /// This constant vector has a value of (-1.0, 0.0, 0.0), representing the negative X-axis in 3D space.
    pub const NEG_AXIS_X: Self = Self {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    };

    /// A constant `FVector3` representing the negative Y-axis.
    ///
    /// This constant vector has a value of (0.0, -1.0, 0.0), representing the negative Y-axis in 3D space.
    pub const NEG_AXIS_Y: Self = Self {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    };

    /// A constant `FVector3` representing the negative Z-axis.
    ///
    /// This constant vector has a value of (0.0, 0.0, -1.0), representing the negative Z-axis in 3D space.
    pub const NEG_AXIS_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };

    /// Constructs a new `FVector3`.
    ///
    /// # Arguments
    ///
    /// * `x` - A float that holds the x component of the vector.
    /// * `y` - A float that holds the y component of the vector.
    /// * `z` - A float that holds the z component of the vector.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Converts the `FVector3` into a `FVector4` by adding a fourth component `w`.
    ///
    /// This method creates a new `FVector4` by taking the `x`, `y`, and `z` components of the
    /// `FVector3` and adding a fourth component `w`. This can be useful when working with
    /// homogeneous coordinates, where the fourth component typically represents a weight or scale.
    ///
    /// # Arguments
    ///
    /// * `w` - The value of the fourth component to add to the resulting `FVector4`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` with `x`, `y`, `z`, and `w` components.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector3 = FVector3::new(2.0, 3.0, 4.0);
    /// let vector4 = vector3.add_axis(1.0);
    ///
    /// assert_eq!(vector4, FVector4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
    #[inline]
    pub const fn add_axis(self, w: f32) -> super::FVector4 {
        super::FVector4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }

    /// Create a new instance of `FVector3` with all `x`, `y` and `z` components set to the given `value`.
    ///
    /// This function creates a new `FVector3` instance with all `x`, `y` and `z` components
    /// initialized to the specified `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to set for all `x`, `y` and `z` components of the `FVector3`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` instance with all `x`, `y` and `z` components set to `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector = FVector3::splat(5.0);
    /// assert_eq!(vector.x, 5.0);
    /// assert_eq!(vector.y, 5.0);
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }

    /// Returns the x and y components of the vector as a tuple.
    #[inline]
    pub const fn xy(self) -> (f32, f32) {
        (self.x, self.y)
    }

    /// Returns the x and z components of the vector as a tuple.
    #[inline]
    pub const fn xz(self) -> (f32, f32) {
        (self.x, self.z)
    }

    /// Returns the y and x components of the vector as a tuple.
    #[inline]
    pub const fn yx(self) -> (f32, f32) {
        (self.y, self.x)
    }

    /// Returns the y and z components of the vector as a tuple.
    #[inline]
    pub const fn yz(self) -> (f32, f32) {
        (self.y, self.z)
    }

    /// Returns the z and x components of the vector as a tuple.
    #[inline]
    pub const fn zx(self) -> (f32, f32) {
        (self.z, self.x)
    }

    /// Returns the z and y components of the vector as a tuple.
    #[inline]
    pub const fn zy(self) -> (f32, f32) {
        (self.z, self.y)
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

    /// Returns the z component of the vector twice as a tuple.
    #[inline]
    pub const fn zz(self) -> (f32, f32) {
        (self.z, self.z)
    }

    /// Returns the x, y, and z components of the vector as a tuple.
    #[inline]
    pub const fn xyz(self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }

    /// Returns the z, y, and x components of the vector as a tuple.
    #[inline]
    pub const fn zyx(self) -> (f32, f32, f32) {
        (self.z, self.y, self.x)
    }

    /// Returns the x, z, and y components of the vector as a tuple.
    #[inline]
    pub const fn xzy(self) -> (f32, f32, f32) {
        (self.x, self.z, self.y)
    }

    /// Returns the y, z, and x components of the vector as a tuple.
    #[inline]
    pub const fn yzx(self) -> (f32, f32, f32) {
        (self.y, self.z, self.x)
    }

    /// Returns the y, x, and z components of the vector as a tuple.
    #[inline]
    pub const fn yxz(self) -> (f32, f32, f32) {
        (self.y, self.x, self.z)
    }

    /// Returns the z, x, and y components of the vector as a tuple.
    #[inline]
    pub const fn zxy(self) -> (f32, f32, f32) {
        (self.z, self.x, self.y)
    }

    /// Returns the x component of the vector three times as a tuple.
    #[inline]
    pub const fn xxx(self) -> (f32, f32, f32) {
        (self.x, self.x, self.x)
    }

    /// Returns the y component of the vector three times as a tuple.
    #[inline]
    pub const fn yyy(self) -> (f32, f32, f32) {
        (self.y, self.y, self.y)
    }

    /// Returns the z component of the vector three times as a tuple.
    #[inline]
    pub const fn zzz(self) -> (f32, f32, f32) {
        (self.z, self.z, self.z)
    }

    /// Returns the x and y components of the vector as an FVector2.
    #[inline]
    pub const fn axis_xy(self) -> super::FVector2 {
        super::FVector2 {
            x: self.x,
            y: self.y,
        }
    }

    /// Returns the x and z components of the vector as an FVector2.
    #[inline]
    pub const fn axis_xz(self) -> super::FVector2 {
        super::FVector2 {
            x: self.x,
            y: self.z,
        }
    }

    /// Returns the y and x components of the vector as an FVector2.
    #[inline]
    pub const fn axis_yx(self) -> super::FVector2 {
        super::FVector2 {
            x: self.y,
            y: self.x,
        }
    }

    /// Returns the y and z components of the vector as an FVector2.
    #[inline]
    pub const fn axis_yz(self) -> super::FVector2 {
        super::FVector2 {
            x: self.y,
            y: self.z,
        }
    }

    /// Returns the z and x components of the vector as an FVector2.
    #[inline]
    pub const fn axis_zx(self) -> super::FVector2 {
        super::FVector2 {
            x: self.z,
            y: self.x,
        }
    }

    /// Returns the z and y components of the vector as an FVector2.
    #[inline]
    pub const fn axis_zy(self) -> super::FVector2 {
        super::FVector2 {
            x: self.z,
            y: self.y,
        }
    }

    /// Returns the x component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_xx(self) -> super::FVector2 {
        super::FVector2 {
            x: self.x,
            y: self.x,
        }
    }

    /// Returns the y component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_yy(self) -> super::FVector2 {
        super::FVector2 {
            x: self.y,
            y: self.y,
        }
    }

    /// Returns the z component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_zz(self) -> super::FVector2 {
        super::FVector2 {
            x: self.z,
            y: self.z,
        }
    }

    /// Returns the z, y, and x components of the vector as an FVector3.
    #[inline]
    pub const fn axis_zyx(self) -> FVector3 {
        FVector3 {
            x: self.z,
            y: self.y,
            z: self.x,
        }
    }

    /// Returns the x, z, and y components of the vector as an FVector3.
    #[inline]
    pub const fn axis_xzy(self) -> FVector3 {
        FVector3 {
            x: self.x,
            y: self.z,
            z: self.y,
        }
    }

    /// Returns the y, z, and x components of the vector as an FVector3.
    #[inline]
    pub const fn axis_yzx(self) -> FVector3 {
        FVector3 {
            x: self.y,
            y: self.z,
            z: self.x,
        }
    }

    /// Returns the y, x, and z components of the vector as an FVector3.
    #[inline]
    pub const fn axis_yxz(self) -> FVector3 {
        FVector3 {
            x: self.y,
            y: self.x,
            z: self.z,
        }
    }

    /// Returns the z, x, and y components of the vector as an FVector3.
    #[inline]
    pub const fn axis_zxy(self) -> FVector3 {
        FVector3 {
            x: self.z,
            y: self.x,
            z: self.y,
        }
    }

    /// Returns the x component of the vector three times as an FVector3.
    #[inline]
    pub const fn axis_xxx(self) -> FVector3 {
        FVector3 {
            x: self.x,
            y: self.x,
            z: self.x,
        }
    }

    /// Returns the y component of the vector three times as an FVector3.
    #[inline]
    pub const fn axis_yyy(self) -> FVector3 {
        FVector3 {
            x: self.y,
            y: self.y,
            z: self.y,
        }
    }

    /// Returns the z component of the vector three times as an FVector3.
    #[inline]
    pub const fn axis_zzz(self) -> FVector3 {
        FVector3 {
            x: self.z,
            y: self.z,
            z: self.z,
        }
    }

    /// Returns the dot product of the vector and another vector.
    ///
    /// # Arguments
    ///
    /// * `other` - Another vector to calculate the dot product with.
    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x
            .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
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
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y.mul_add(other.z, -self.z * other.y),
            y: -self.x.mul_add(other.z, -self.z * other.x),
            z: self.x.mul_add(other.y, -self.x * other.y),
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
        self.x
            .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z))
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
    pub fn length_inv(self) -> f32 {
        self.length_sq().sqrt().recip()
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
    pub fn distance(self, other: Self) -> f32 {
        self.distance_sq(other).sqrt()
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
    pub fn distance_sq(self, other: Self) -> f32 {
        let x_diff = self.x - other.x;
        let y_diff = self.y - other.y;
        let z_diff = self.z - other.z;

        x_diff.mul_add(x_diff, y_diff.mul_add(y_diff, z_diff * z_diff))
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
    pub fn normalize(self) -> Self {
        let length_inv = self.length_inv();

        Self {
            x: self.x * length_inv,
            y: self.y * length_inv,
            z: self.z * length_inv,
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
        let x = self.x - end.x;
        let y = self.y - end.y;
        let z = self.z - end.z;

        Self {
            x: x.mul_add(scalar, self.x),
            y: y.mul_add(scalar, self.y),
            z: z.mul_add(scalar, self.z),
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
    pub fn min(self, value: Self) -> Self {
        Self {
            x: self.x.min(value.x),
            y: self.y.min(value.y),
            z: self.z.min(value.z),
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
    pub fn max(self, value: Self) -> Self {
        Self {
            x: self.x.max(value.x),
            y: self.y.max(value.y),
            z: self.z.max(value.z),
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
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0
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
        self.x.is_nan() && self.y.is_nan() && self.z.is_nan()
    }

    /// Checks if any component of the `FVector3` is finite.
    ///
    /// This function checks each component of the `FVector3` for finiteness and returns
    /// `true` if at least one component is finite, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// Returns `true` if any component of the `FVector3` is finite, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// // Create an `FVector3` with all finite components
    /// let finite_vector = FVector3::new(2.0, 3.0, -1.0);
    ///
    /// // Create an `FVector3` with at least one infinite component
    /// let infinite_vector = FVector3::new(f32::INFINITY, 2.0, 0.0);
    ///
    /// // Check if any component of the finite vector is finite
    /// assert_eq!(finite_vector.is_finite(), true);
    ///
    /// // Check if any component of the infinite vector is finite
    /// assert_eq!(infinite_vector.is_finite(), false);
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() || self.y.is_finite() || self.z.is_finite()
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
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
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
        self.x != rhs.x || self.y != rhs.y || self.z != rhs.z
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
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
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
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
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
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
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
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
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
        Self {
            x: self.x % rhs.x,
            y: self.y % rhs.y,
            z: self.z % rhs.z,
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
        *self = Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
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
        *self = Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
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
        *self = Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
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
        *self = Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
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
        *self = Self {
            x: self.x % rhs.x,
            y: self.y % rhs.y,
            z: self.z % rhs.z,
        }
    }
}
