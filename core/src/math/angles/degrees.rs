use std::fmt;

/// A type representing an angle measured in degrees.
///
/// `Degrees` is a simple wrapper around an `f32` value, providing a clear
/// and type-safe way to work with angles in degrees.
#[derive(Clone, Copy)]
pub struct Degrees(pub(in crate::math) f32);

/// Creates a new `Degrees` instance with the specified angle in degrees.
///
/// # Parameters
///
/// * `degrees`: The angle value in degrees.
///
/// # Examples
///
/// ```
/// use axion::math::Degrees;
///
/// let right_angle = Degrees::new(90.0);
/// ```
#[inline]
pub const fn degrees(deg: f32) -> Degrees {
    Degrees(deg)
}

impl Default for Degrees {
    /// Creates a new `Degrees` instance with the default value of `0.0` degrees.
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

impl Degrees {
    /// A constant representing a right angle, equal to `90.0` degrees.
    pub const RIGHT_ANGLE: Self = Self(90.0);

    /// A constant representing a straight angle, equal to `180.0` degrees.
    pub const STRAIGHT_ANGLE: Self = Self(180.0);

    /// A constant representing a full rotation, equal to `360.0` degrees.
    pub const FULL_ROTATION: Self = Self(360.0);

    /// Creates a new `Degrees` instance with the specified angle in degrees.
    ///
    /// # Parameters
    ///
    /// * `degrees`: The angle value in degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let right_angle = Degrees::new(90.0);
    /// ```
    #[inline]
    pub const fn new(degrees: f32) -> Self {
        Self(degrees)
    }
}

impl Degrees {
    /// Conversion factor from degrees to radians (tau represents a full turn).
    const FACTOR_DEG_TO_RAD: f32 = std::f32::consts::TAU / 360.0;

    /// Conversion factor from degrees to turns.
    const FACTOR_DEG_TO_TR: f32 = 1.0 / 360.0;

    /// Converts an angle in degrees to radians.
    ///
    /// This method multiplies the angle in degrees by the conversion factor
    /// `f32::consts::TAU / 360.0` to obtain the equivalent angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degree;
    ///
    /// let thirty_degrees = Degree::new(30.0);
    /// let radians = thirty_degrees.into_radians();
    /// ```
    #[inline]
    pub fn into_radians(self) -> super::Radians {
        super::Radians(self.0 * Self::FACTOR_DEG_TO_RAD)
    }

    /// Converts an angle in degrees to turns.
    ///
    /// This method multiplies the angle in degrees by the conversion factor
    /// `1.0 / 360.0` to obtain the equivalent angle in turns.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degree;
    ///
    /// let ninety_degrees = Degree::new(90.0);
    /// let turns = ninety_degrees.into_turns();
    /// ```
    #[inline]
    pub fn into_turns(self) -> super::Turns {
        super::Turns(self.0 * Self::FACTOR_DEG_TO_TR)
    }
}

impl fmt::Debug for Degrees {
    /// Formats the `Degrees` value for debugging purposes.
    ///
    /// This implementation formats the value with the associated "deg" unit
    /// to make it clear that it represents an angle in degrees.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)?; // Delegate formatting of the inner f32 value
        formatter.write_str("deg") // Append "deg" unit
    }
}

impl fmt::Display for Degrees {
    /// Formats the `Degrees` value for display purposes.
    ///
    /// This implementation formats the value with the associated "deg" unit
    /// to make it clear that it represents an angle in degrees.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)?; // Delegate formatting of the inner f32 value
        formatter.write_str("deg") // Append "deg" unit
    }
}
