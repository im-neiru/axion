/// A struct representing spherical angles.
///
/// This struct is used to hold spherical angles, typically representing
/// direction in a three-dimensional space. It includes two angles: azimuth and polar.
///
/// - `azimuth`: The azimuthal angle, which measures the angle in the horizontal plane
///   measured from the positive x-axis, typically in units like Radians, Degrees, or Turns.
/// - `polar`: The polar angle, which measures the angle from the positive z-axis,
///   typically in units like Radians, Degrees, or Turns.
///
/// This struct is generic over the type `T`, allowing you to use different angular units
/// to represent the angles, such as `Radians`, `Degrees`, or `Turns`.
///
/// # Examples
///
/// ```
/// use axion::math::{SphericalAngles, Radians};
///
/// let angles: SphericalAngles<Radians> = SphericalAngles {
///     azimuth: Radians::new(1.0),   // Azimuth angle in radians
///     polar: Radians::new(0.5),     // Polar angle in radians
/// };
/// ```
///
#[derive(Clone, Copy)]
pub struct SphericalAngles<T: super::Angle> {
    /// The azimuthal angle, typically in angular units like Radians, Degrees, or Turns.
    pub azimuth: T,
    /// The polar angle, typically in angular units like Radians, Degrees, or Turns.
    pub polar: T,
}
