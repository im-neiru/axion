use std::fmt;

/// A type representing an angle measured in radians.
///
/// `Radians` is a simple wrapper around a `f32` value, providing a clear
/// and type-safe way to work with angles in radians.
///
/// # Examples
///
/// ```
/// use axion::math::Radians;
///
/// let pi_over_2 = Radians::new(std::f32::consts::FRAC_PI_2);
/// let angle = pi_over_2.0; // Access the inner f32 value
/// ```
#[derive(Clone, Copy)]
pub struct Radians(pub(in crate::math) f32);

/// Creates a new `Radians` instance with the specified angle in radians.
///
/// # Parameters
///
/// * `rad`: The angle value in radians.
///
/// # Examples
///
/// ```
/// use axion::math::Radians;
///
/// let pi_radians = Radians::radians(std::f32::consts::PI);
/// ```
#[inline]
pub const fn radians(rad: f32) -> Radians {
    Radians(rad)
}

impl Default for Radians {
    /// Creates a new `Radians` instance with the default value of `0.0` radians.
    #[inline]
    fn default() -> Self {
        Self(Default::default())
    }
}

impl Radians {
    /// The mathematical constant π (pi) represented in radians.
    ///
    /// This constant represents the value of π (pi) in radians and can be used
    /// when working with angles in trigonometric calculations.
    pub const PI: Radians = Radians(std::f32::consts::PI);

    /// Half of the mathematical constant π (pi) represented in radians.
    ///
    /// This constant represents the value of π (pi) divided by 2 in radians.
    /// It is often used in various trigonometric calculations.
    pub const PI_HALF: Radians = Radians(std::f32::consts::FRAC_PI_2);

    /// The mathematical constant τ (tau) represented in radians.
    ///
    /// Tau (τ) is a mathematical constant equal to 2π, and this constant
    /// represents that value in radians. It is sometimes used as a more
    /// intuitive alternative to 2π in certain mathematical contexts.
    pub const TAU: Radians = Radians(std::f32::consts::TAU);

    /// Creates a new `Radians` instance with the specified angle in radians.
    ///
    /// # Parameters
    ///
    /// * `radians`: The angle value in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_over_2 = Radians::new(std::f32::consts::FRAC_PI_2);
    /// ```
    #[inline]
    pub const fn new(radians: f32) -> Self {
        Self(radians)
    }
}
impl Radians {
    /// Conversion factor from radians to degrees.
    pub(super) const FACTOR_RAD_TO_DEG: f32 = 360.0 / std::f32::consts::TAU;

    /// Conversion factor from radians to turns (tau represents a full turn).
    pub(super) const FACTOR_RAD_TO_TR: f32 = 1.0 / std::f32::consts::TAU;

    // Converts an angle in radians to degrees.
    ///
    /// This method multiplies the angle in radians by the conversion factor
    /// `360.0 / f32::consts::TAU` to obtain the equivalent angle in degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_radians = Radians::new(std::f32::consts::PI);
    /// let degrees = pi_radians.into_degrees();
    /// ```
    #[inline]
    pub fn into_degrees(self) -> super::Degrees {
        super::Degrees(self.0 * Self::FACTOR_RAD_TO_DEG)
    }

    /// Converts an angle in radians to turns.
    ///
    /// This method multiplies the angle in radians by the conversion factor
    /// `1.0 / f32::consts::TAU` to obtain the equivalent angle in turns.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_radians = Radians::new(std::f32::consts::PI);
    /// let turns = pi_radians.into_turns();
    /// ```
    #[inline]
    pub fn into_turns(self) -> super::Turns {
        super::Turns(self.0 * Self::FACTOR_RAD_TO_TR)
    }
}

impl fmt::Debug for Radians {
    /// Formats the `Radians` value for debugging purposes.
    ///
    /// This implementation formats the value with the associated "rad" unit
    /// to make it clear that it represents an angle in radians.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)?; // Delegate formatting of the inner f32 value
        formatter.write_str("rad") // Append "rad" unit
    }
}

impl fmt::Display for Radians {
    /// Formats the `Radians` value for display purposes.
    ///
    /// This implementation formats the value with the associated "rad" unit
    /// to make it clear that it represents an angle in radians.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)?; // Delegate formatting of the inner f32 value
        formatter.write_str("rad") // Append "rad" unit
    }
}
