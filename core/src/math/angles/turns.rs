use std::fmt;

/// A type representing an angle measured in turns.
///
/// `Turns` is a simple wrapper around an `f32` value, providing a clear
/// and type-safe way to work with angles in turns.
#[derive(Clone, Copy)]
pub struct Turns(pub(in crate::math) f32);

/// Creates a new `Turns` instance with the specified angle in turns.
///
/// # Parameters
///
/// * `turns`: The angle value in turns.
///
/// # Examples
///
/// ```
/// use axion::math::Turns;
///
/// let half_turn = Turns::new(0.5);
/// ```
#[inline]
pub const fn turns(turns: f32) -> Turns {
    Turns(turns)
}

impl Default for Turns {
    /// Creates a new `Turns` instance with the default value of `0.0` turns.
    fn default() -> Self {
        Self(Default::default())
    }
}

impl Turns {
    /// A constant representing a full turn, equal to `1.0` turns.
    pub const FULL_TURN: Self = Self(1.0);

    /// A constant representing half a turn, equal to `0.5` turns.
    pub const HALF_TURN: Self = Self(0.5);

    /// Creates a new `Turns` instance with the specified angle in turns.
    ///
    /// # Parameters
    ///
    /// * `turns`: The angle value in turns.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let half_turn = Turns::new(0.5);
    /// ```
    #[inline]
    pub const fn new(turns: f32) -> Self {
        Self(turns)
    }
}

impl Turns {
    /// Conversion factor from turns to degrees.
    const FACTOR_TR_TO_DEG: f32 = 360.0;

    /// Conversion factor from turns to radians (tau represents a full turn).
    const FACTOR_TR_TO_RAD: f32 = std::f32::consts::TAU;

    /// Converts an angle in turns to degrees.
    ///
    /// This method multiplies the angle in turns by the conversion factor
    /// `360.0` to obtain the equivalent angle in degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let half_turn = Turns::new(0.5);
    /// let degrees = half_turn.into_degrees();
    /// ```
    #[inline]
    pub fn into_degrees(self) -> super::Degrees {
        super::Degrees(self.0 * Self::FACTOR_TR_TO_DEG)
    }

    /// Converts an angle in turns to radians.
    ///
    /// This method multiplies the angle in turns by the conversion factor
    /// `f32::consts::TAU` to obtain the equivalent angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let half_turn = Turns::new(0.5);
    /// let radians = half_turn.into_radians();
    /// ```
    #[inline]
    pub fn into_radians(self) -> super::Radians {
        super::Radians(self.0 * Self::FACTOR_TR_TO_RAD)
    }
}

impl fmt::Debug for Turns {
    /// Formats the `Turns` value for debugging purposes.
    ///
    /// This implementation formats the value with the associated "tr" unit
    /// to make it clear that it represents an angle in turns.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)?; // Delegate formatting of the inner f32 value
        formatter.write_str("tr") // Append "tr" unit
    }
}

impl fmt::Display for Turns {
    /// Formats the `Turns` value for display purposes.
    ///
    /// This implementation formats the value with the associated "tr" unit
    /// to make it clear that it represents an angle in turns.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)?; // Delegate formatting of the inner f32 value
        formatter.write_str("tr") // Append "tr" unit
    }
}
