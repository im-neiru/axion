use std::ops;

use crate::math::{Quaternion, Vector3};

impl Quaternion {
    /// Transforms a `Vector3` by this `Quaternion`.
    ///
    /// This method applies the rotation represented by the `Quaternion` to the provided
    /// `Vector3`, returning a new `Vector3` with the transformed coordinates.
    ///
    /// # Parameters
    ///
    /// * `self`: The `Quaternion` representing the rotation.
    /// * `vector`: The `Vector3` to be transformed.
    ///
    /// # Returns
    ///
    /// A new `Vector3` representing the result of transforming the input `Vector3` by this `Quaternion`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::{Quaternion, vec3};
    ///
    /// let quaternion = Quaternion::identity();
    /// let vector = vec3(1.0, 0.0, 0.0);
    ///
    /// let transformed_vector = quaternion.transform_vector3(vector);
    /// ```
    ///
    /// In this example, an identity `Quaternion` is used to transform a `Vector3`.
    /// Since it's an identity, the transformed `Vector3` will be the same as the input.
    #[inline]
    pub fn transform_vector3(self, vector: Vector3) -> Self {
        Self {
            w: self.x.mul_add(
                -vector.x,
                self.y.mul_add(-vector.y, -self.z * vector.z),
            ),

            x: self.w.mul_add(
                vector.x,
                self.y.mul_add(vector.z, -self.z * vector.y),
            ),
            y: self.w.mul_add(
                vector.y,
                self.x.mul_add(-vector.z, self.z * vector.x),
            ),
            z: self.w.mul_add(
                vector.z,
                self.x.mul_add(vector.y, self.y * -vector.x),
            ),
        }
    }
}

impl ops::Add for Quaternion {
    type Output = Self;

    /// Adds two quaternions element-wise.
    ///
    /// This method returns a new quaternion whose components are the element-wise sum of the components
    /// of the two input quaternions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use axion::math::Quaternion;
    ///
    /// let quat1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let quat2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let sum = quat1 + quat2;
    /// assert_eq!(sum, Quaternion::new(6.0, 8.0, 10.0, 12.0));
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl ops::Sub for Quaternion {
    type Output = Self;

    /// Subtracts one quaternion from another element-wise.
    ///
    /// This method returns a new quaternion whose components are the element-wise difference between
    /// the components of the two input quaternions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use axion::math::Quaternion;
    ///
    /// let quat1 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let quat2 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let difference = quat1 - quat2;
    /// assert_eq!(difference, Quaternion::new(4.0, 4.0, 4.0, 4.0));
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl ops::Mul for Quaternion {
    type Output = Self;

    /// Multiplies this quaternion by another quaternion using the Hamilton product.
    ///
    /// # Parameters
    ///
    /// * `rhs`: A reference to the quaternion to multiply with.
    ///
    /// # Returns
    ///
    /// A new quaternion that is the result of the Hamilton product.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use axion::math::Quaternion;
    ///
    /// let quat1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let quat2 = Quaternion::new(5.0, 6.0, 7.0, 8.0);
    /// let result = quat1 * quat2;
    /// assert_eq!(result, Quaternion::new(-60.0, 12.0, 30.0, 24.0));
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            w: self.w.mul_add(
                rhs.w,
                self.x
                    .mul_add(-rhs.x, self.y.mul_add(-rhs.y, -self.z * rhs.z)),
            ),
            x: self.w.mul_add(
                rhs.x,
                self.x
                    .mul_add(rhs.w, self.y.mul_add(rhs.z, -self.z * rhs.y)),
            ),
            y: self.w.mul_add(
                rhs.y,
                self.x
                    .mul_add(-rhs.z, self.y.mul_add(rhs.w, self.z * rhs.x)),
            ),
            z: self.w.mul_add(
                rhs.z,
                self.x
                    .mul_add(rhs.y, self.y.mul_add(-rhs.x, self.z * rhs.w)),
            ),
        }
    }
}
