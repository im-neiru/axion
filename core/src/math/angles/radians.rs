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

impl Radians {
    /// Returns the cosine of the angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_half = Radians::PI_HALF;
    /// let cos = pi_half.cos();
    /// ```
    #[inline]
    pub fn cos(self) -> f32 {
        (self.0).cos()
    }

    /// Returns the sine of the angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_half = Radians::PI_HALF;
    /// let sin = pi_half.sin();
    /// ```
    #[inline]
    pub fn sin(self) -> f32 {
        (self.0).sin()
    }

    /// Returns the tangent of the angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_half = Radians::PI_HALF;
    /// let tan = pi_half.tan();
    /// ```
    #[inline]
    pub fn tan(self) -> f32 {
        (self.0).tan()
    }

    /// Returns the secant of the angle in radians.
    ///
    /// The secant is the reciprocal of the cosine.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_half = Radians::PI_HALF;
    /// let sec = pi_half.sec();
    /// ```
    #[inline]
    pub fn sec(self) -> f32 {
        (self.0).cos().recip()
    }

    /// Returns the cosecant of the angle in radians.
    ///
    /// The cosecant is the reciprocal of the sine.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_half = Radians::PI_HALF;
    /// let csc = pi_half.csc();
    /// ```
    #[inline]
    pub fn csc(self) -> f32 {
        (self.0).sin().recip()
    }

    /// Returns the cotangent of the angle in radians.
    ///
    /// The cotangent is the reciprocal of the tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi_half = Radians::PI_HALF;
    /// let cot = pi_half.cot();
    /// ```
    #[inline]
    pub fn cot(self) -> f32 {
        (self.0).tan().recip()
    }

    /// Returns a unit vector in the direction of the angle in radians.
    ///
    /// The `normal` method computes the cosine and sine of the angle in radians
    /// and returns a `axion::math::FVector2` representing the unit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let pi = Radians::PI;
    /// let normal_vector = pi.normal();
    /// ```
    #[inline]
    pub fn normal(self) -> crate::math::FVector2 {
        let (cos, sin) = (self.0.cos(), self.0.sin());
        crate::math::FVector2 { x: cos, y: sin }
    }

    /// Computes the arccosine of the `scalar` value in radians.
    ///
    /// The result is in the range \[*0* rad, *π* rad\], or NaN if the `scalar` is
    /// outside the range \[`-1.0`, `1.0`\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let scalar = 0.5;
    /// let angle = Radians::acos(scalar);
    /// ```
    #[inline]
    pub fn acos(scalar: f32) -> Self {
        Self(scalar.acos())
    }

    /// Computes the arcsine of the `scalar` value in radians.
    ///
    /// The result is in the range \[*-½π* rad, *½π* rad\], or `NaN` if the `scalar` is
    /// outside the range \[`-1.0`, `1.0`\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let scalar = 0.5;
    /// let angle = Radians::asin(scalar);
    /// ```
    #[inline]
    pub fn asin(scalar: f32) -> Self {
        Self(scalar.asin())
    }

    /// Computes the arc tangent of the `scalar` value in radians.
    ///
    /// The result is in the range \[*-½π* rad, *½π* rad\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let scalar = 0.5;
    /// let angle = Radians::atan(scalar);
    /// ```

    #[inline]
    pub fn atan(scalar: f32) -> Self {
        Self(scalar.atan())
    }

    /// Computes the four-quadrant arctangent of the `y` and `x` coordinates
    /// in radians, which corresponds to the vector direction.
    ///
    /// The result has the following range:
    ///
    ///   - When `x` = `0.0` and `y` = `0.0`: *0* rad
    ///   - When `x` >= `0.0`: arctan(`y`/`x`) -> \[*-½π* rad, *½π* rad\]
    ///   - When `y` >= `0.0`: arctan(`y`/`x`) + π -> (*½π* rad, *π* rad]
    ///   - When `y` < `0.0`: arctan(`y`/`x`) - π -> (-πrad, *-½π* rad)
    ///
    /// # Parameters
    ///
    /// * `y`: The `y` coordinate of the vector.
    /// * `x`: The `x` coordinate of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Radians;
    ///
    /// let angle = Radians::atan2(1.0, 1.0);
    /// ```
    #[inline]
    pub fn atan2(y: f32, x: f32) -> Self {
        Self(f32::atan2(y, x))
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
