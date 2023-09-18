use crate::math::{degrees, Radians, Vector2, Vector3};
use std::fmt;
use std::ops;
/// A quaternion representing a rotation in 3D space.
///
/// Quaternions are a mathematical concept used to represent rotations in 3D space.
///
/// Quaternions have four components: `x`, `y`, `z`, and `w`, where:
///
/// - `x`, `y`, and `z` represent the axis of rotation.
/// - `w` is a scalar component.
///
/// Together, these components encode a 3D rotation.
///
/// # Example
///
/// ```rust
/// use axion::math::Quaternion;
///
/// // Create a new quaternion representing a 90-degree rotation around the Z-axis.
/// let quat = Quaternion { x: 0.0, y: 0.0, z: 1.0, w: 1.0 };
/// ```
#[derive(Copy, Clone)]
pub struct Quaternion {
    /// The `x` component of the quaternion.
    pub x: f32,

    /// The `y` component of the quaternion.
    pub y: f32,

    /// The `z` component of the quaternion.
    pub z: f32,

    /// The `w` component of the quaternion.
    pub w: f32,
}

/// Creates a new `Quaternion` with the specified components.
///
/// # Arguments
///
/// * `x`: The `x` component of the quaternion.
/// * `y`: The `y` component of the quaternion.
/// * `z`: The `z` component of the quaternion.
/// * `w`: The `w` component of the quaternion.
///
/// # Returns
///
/// A `Quaternion` representing a rotation in 3D space.
///
/// # Examples
///
/// ```rust
/// use axion::math::{Quaternion, quat};
///
/// // Create a new quaternion representing a 90-degree rotation around the Z-axis.
/// let quat = quat(0.0, 0.0, 1.0, 1.0);
/// ```
#[inline]
pub const fn quat(x: f32, y: f32, z: f32, w: f32) -> Quaternion {
    Quaternion { x, y, z, w }
}

impl Quaternion {
    /// Creates a new `Quaternion` for a rotation around a specified axis.
    ///
    /// This method constructs a quaternion representing a rotation of `theta` radians
    /// around the given `axis`.
    ///
    /// # Arguments
    ///
    /// * `axis`: The 3D vector representing the rotation axis.
    /// * `theta`: The angle in radians for the rotation around the axis.
    ///
    /// # Returns
    ///
    /// A `Quaternion` representing the specified rotation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use axion::math::{Quaternion, Vector3, Radians};
    ///
    /// // Create a quaternion for a 90-degree rotation around the X-axis.
    /// let axis = Vector3::new(1.0, 0.0, 0.0);
    /// let theta = Radians::PI_HALF;
    /// let quat = Quaternion::from_axis(axis, theta);
    /// println!("{quat}");
    /// ```
    #[inline]
    pub fn from_axis(axis: Vector3, theta: Radians) -> Self {
        let Vector2 { x: cos, y: sin } = (theta / 2.0).normal();
        let normal = axis.normalize();

        Self {
            w: cos,
            x: sin * normal.x,
            y: sin * normal.y,
            z: sin * normal.z,
        }
    }

    #[inline]
    pub fn vector3(self) -> Vector3 {
        Vector3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    #[inline]
    pub fn conjugate(self) -> Quaternion {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl fmt::Display for Quaternion {
    /// Formats the quaternion as a human-readable string in the format:
    ///
    /// `w + x𝑖 + y𝑗 + z𝑘`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use axion::math::{Quaternion, quat};
    ///
    /// let quat = quat(2.0, 3.0, 4.0, 1.0);
    /// let formatted = format!("{quat:.1}");
    /// assert_eq!(formatted, "1.0 + 2.0𝑖 + 3.0𝑗 + 4.0𝑘");
    /// ```
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.w, formatter)?;
        formatter.write_str(" + ")?;
        fmt::Display::fmt(&self.x, formatter)?;
        formatter.write_str("𝑖 + ")?;
        fmt::Display::fmt(&self.y, formatter)?;
        formatter.write_str("𝑗 + ")?;
        fmt::Display::fmt(&self.z, formatter)?;
        formatter.write_str("𝑘")
    }
}
