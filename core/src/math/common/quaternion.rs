use crate::math::{Radians, Vector3};

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

impl Quaternion {}
