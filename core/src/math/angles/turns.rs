use std::{fmt, ops};

use super::Radians;

/// A type representing an angle measured in turns.
///
/// `Turns` is a simple wrapper around an `f32` value, providing a clear
/// and type-safe way to work with angles in turns.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
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

impl super::Angle for Turns {}
impl super::PrivateAngle for Turns {}

impl Default for Turns {
    /// Creates a new `Turns` instance with the default value of `0.0` turns.
    #[inline]
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

impl Turns {
    /// Returns the cosine of the angle in turns.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let quarter_turn = Turns::new(0.25);
    /// let cos = quarter_turn.cos();
    /// ```
    #[inline]
    pub fn cos(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).cos()
    }

    /// Returns the sine of the angle in turns.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let half_turn = Turns::new(0.5);
    /// let sin = half_turn.sin();
    /// ```
    #[inline]
    pub fn sin(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).sin()
    }

    /// Returns the tangent of the angle in turns.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let quater_turn = Turns::new(0.25);
    /// let tan = quater_turn.tan();
    /// ```
    #[inline]
    pub fn tan(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).tan()
    }

    /// Returns the secant of the angle in turns.
    ///
    /// The secant is the reciprocal of the cosine.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let half_turn = Turns::new(0.5);
    /// let sec = half_turn.sec();
    /// ```
    #[inline]
    pub fn sec(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).cos().recip()
    }

    /// Returns the cosecant of the angle in turns.
    ///
    /// The cosecant is the reciprocal of the sine.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let quater_turn = Turns::new(0.25);
    /// let csc = quater_turn.csc();
    /// ```
    #[inline]
    pub fn csc(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).sin().recip()
    }

    /// Returns the cotangent of the angle in turns.
    ///
    /// The cotangent is the reciprocal of the tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let quarter_turn = Turns::new(0.25);
    /// let cot = quarter_turn.cot();
    /// ```
    #[inline]
    pub fn cot(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).tan().recip()
    }

    /// Returns a unit vector in the direction of the angle in turns.
    ///
    /// The `normal` method computes the cosine and sine of the angle in turns
    /// and returns a `axion::math::Vector2` representing the unit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let quarter_turn = Turns::new(0.25);
    /// let normal_vector = quarter_turn.normal();
    /// ```
    #[inline]
    pub fn normal(self) -> crate::math::Vector2 {
        let (cos, sin) = {
            let radians = self.0 * Self::FACTOR_TR_TO_RAD;
            (radians.cos(), radians.sin())
        };

        crate::math::Vector2 { x: cos, y: sin }
    }

    /// Computes the arccosine of the `scalar` value in turns.
    ///
    /// The result is in the range \[`0 tr`, `½ tr`\], or NaN if the `scalar` is
    /// outside the range \[`-1.0`, `1.0`\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let scalar = 0.5;
    /// let angle = Turns::acos(scalar);
    /// ```
    #[inline]
    pub fn acos(scalar: f32) -> Self {
        Self(scalar.acos() * Radians::FACTOR_RAD_TO_TR)
    }

    /// Computes the arcsine of the `scalar` value in turns.
    ///
    /// The result is in the range \[`-¼ tr`, `¼ tr`\], or `NaN` if the `scalar` is
    /// outside the range \[`-1.0`, `1.0`\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let scalar = 0.5;
    /// let angle = Turns::asin(scalar);
    /// ```
    #[inline]
    pub fn asin(scalar: f32) -> Self {
        Self(scalar.asin() * Radians::FACTOR_RAD_TO_TR)
    }

    /// Computes the arc tangent of the `scalar` value in turns.
    ///
    /// The result is in the range \[`-¼ tr`, `¼ tr`\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let scalar = 0.5;
    /// let angle = Turns::atan(scalar);
    /// ```

    #[inline]
    pub fn atan(scalar: f32) -> Self {
        Self(scalar.atan() * Radians::FACTOR_RAD_TO_TR)
    }

    /// Computes the four-quadrant arctangent of the `y` and `x` coordinates
    /// in turns, which corresponds to the vector direction.
    ///
    /// The result has the following range:
    ///
    ///   - When `x` = `0.0` and `y` = `0.0`: `0 tr`
    ///   - When `x` >= `0.0`: arctan(`y`/`x`) -> \[-¼ tr, ¼ tr\]
    ///   - When `y` >= `0.0`: arctan(`y`/`x`) + π -> (¼ tr, ½ tr]
    ///   - When `y` < `0.0`: arctan(`y`/`x`) - π -> (-½ tr, -¼ tr)
    ///
    /// # Parameters
    ///
    /// * `y`: The `y` coordinate of the vector.
    /// * `x`: The `x` coordinate of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Turns;
    ///
    /// let angle = Turns::atan2(1.0, 1.0);
    /// ```
    #[inline]
    pub fn atan2(y: f32, x: f32) -> Self {
        Self(f32::atan2(y, x) * Radians::FACTOR_RAD_TO_TR)
    }

    /// Returns the hyperbolic cosine of the angle in turns.
    #[inline]
    pub fn cosh(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).cosh()
    }

    /// Returns the hyperbolic sine of the angle in turns.
    #[inline]
    pub fn sinh(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).sinh()
    }

    /// Returns the hyperbolic tangent of the angle in turns.
    #[inline]
    pub fn tanh(self) -> f32 {
        (self.0 * Self::FACTOR_TR_TO_RAD).tanh()
    }

    /// Returns the inverse hyperbolic cosine of the angle in turns.
    #[inline]
    pub fn acosh(self) -> Self {
        Self(self.0.acosh() * Radians::FACTOR_RAD_TO_TR)
    }

    /// Returns the inverse hyperbolic sine of the angle in turns.
    #[inline]
    pub fn asinh(self) -> Self {
        Self(self.0.asinh() * Radians::FACTOR_RAD_TO_TR)
    }

    /// Returns the inverse hyperbolic tangent of the angle in turns.
    #[inline]
    pub fn atanh(self) -> Self {
        Self(self.0.atanh() * Radians::FACTOR_RAD_TO_TR)
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

impl ops::Add for Turns {
    type Output = Self;

    /// Adds two `Turns` values together.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Turns` value.
    /// - `rhs`: The right-hand side `Turns` value to be added.
    ///
    /// # Returns
    ///
    /// A new `Turns` value resulting from the addition.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl ops::Sub for Turns {
    type Output = Self;

    /// Subtracts one `Turns` value from another.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Turns` value.
    /// - `rhs`: The right-hand side `Turns` value to be subtracted.
    ///
    /// # Returns
    ///
    /// A new `Turns` value resulting from the subtraction.
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl ops::Mul for Turns {
    type Output = Self;

    /// Multiplies two `Turns` values together.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Turns` value.
    /// - `rhs`: The right-hand side `Turns` value to be multiplied.
    ///
    /// # Returns
    ///
    /// A new `Turns` value resulting from the multiplication.
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl ops::Div for Turns {
    type Output = Self;

    /// Divides one `Turns` value by another.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Turns` value.
    /// - `rhs`: The right-hand side `Turns` value to be divided by.
    ///
    /// # Returns
    ///
    /// A new `Turns` value resulting from the division.
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl ops::Rem for Turns {
    type Output = Self;

    /// Computes the remainder of dividing one `Turns` value by another.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Turns` value.
    /// - `rhs`: The right-hand side `Turns` value to compute the remainder with.
    ///
    /// # Returns
    ///
    /// A new `Turns` value representing the remainder.
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl ops::Add<f32> for Turns {
    type Output = Self;

    /// Adds a `f32` value to a `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Turns` value.
    /// - `rhs`: The `f32` value to be added.
    ///
    /// # Returns
    ///
    /// A new `Turns` value resulting from the addition.
    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl ops::Sub<f32> for Turns {
    type Output = Self;

    /// Subtracts a `f32` value from a `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Turns` value.
    /// - `rhs`: The `f32` value to be subtracted.
    ///
    /// # Returns
    ///
    /// A new `Turns` value resulting from the subtraction.
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Self(self.0 - rhs)
    }
}

impl ops::Mul<f32> for Turns {
    type Output = Self;

    /// Multiplies a `Turns` value by a `f32` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Turns` value.
    /// - `rhs`: The `f32` value to be multiplied.
    ///
    /// # Returns
    ///
    /// A new `Turns` value resulting from the multiplication.
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl ops::Div<f32> for Turns {
    type Output = Self;

    /// Divides a `Turns` value by a `f32` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Turns` value.
    /// - `rhs`: The `f32` value to divide by.
    ///
    /// # Returns
    ///
    /// A new `Turns` value resulting from the division.
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl ops::Rem<f32> for Turns {
    type Output = Self;

    /// Computes the remainder of dividing a `Turns` value by a `f32` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Turns` value.
    /// - `rhs`: The `f32` value to compute the remainder with.
    ///
    /// # Returns
    ///
    /// A new `Turns` value representing the remainder.
    #[inline]
    fn rem(self, rhs: f32) -> Self::Output {
        Self(self.0 % rhs)
    }
}
impl ops::AddAssign for Turns {
    /// Adds another `Turns` value to this one and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `Turns` value to be added.
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Self(self.0 + rhs.0)
    }
}

impl ops::SubAssign for Turns {
    /// Subtracts another `Turns` value from this one and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `Turns` value to be subtracted.
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self(self.0 - rhs.0)
    }
}

impl ops::MulAssign for Turns {
    /// Multiplies this `Turns` value by another `Turns` value and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `Turns` value to be multiplied.
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self(self.0 * rhs.0)
    }
}

impl ops::DivAssign for Turns {
    /// Divides this `Turns` value by another `Turns` value and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `Turns` value to divide by.
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Self(self.0 / rhs.0)
    }
}

impl ops::RemAssign for Turns {
    /// Computes the remainder of dividing this `Turns` value by another `Turns` value and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `Turns` value to compute the remainder with.
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = Self(self.0 % rhs.0)
    }
}

impl ops::AddAssign<f32> for Turns {
    /// Adds a `f32` scalar to this `Turns` value and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `f32` scalar to be added.
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        *self = Self(self.0 + rhs)
    }
}

impl ops::SubAssign<f32> for Turns {
    /// Subtracts a `f32` scalar from this `Turns` value and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `f32` scalar to be subtracted.
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        *self = Self(self.0 - rhs)
    }
}

impl ops::MulAssign<f32> for Turns {
    /// Multiplies this `Turns` value by a `f32` scalar and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `f32` scalar to be multiplied.
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = Self(self.0 * rhs)
    }
}

impl ops::DivAssign<f32> for Turns {
    /// Divides this `Turns` value by a `f32` scalar and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `f32` scalar to divide by.
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = Self(self.0 / rhs)
    }
}

impl ops::RemAssign<f32> for Turns {
    /// Computes the remainder of dividing this `Turns` value by a `f32` scalar and assigns the result to this `Turns` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Turns` value.
    /// - `rhs`: The `f32` scalar to compute the remainder with.
    #[inline]
    fn rem_assign(&mut self, rhs: f32) {
        *self = Self(self.0 % rhs)
    }
}
