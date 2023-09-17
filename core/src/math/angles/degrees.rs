use std::fmt;
use std::ops;

use super::Radians;

/// A type representing an angle measured in degrees.
///
/// `Degrees` is a simple wrapper around an `f32` value, providing a clear
/// and type-safe way to work with angles in degrees.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Degrees(pub(in crate::math) f32);

impl super::Angle for Degrees {}
impl super::PrivateAngle for Degrees {}

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

    /// A constant representing a right angle, equal to `90.0` degrees.
    pub const NAN: Self = Self(f32::NAN);

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

impl Degrees {
    /// Returns the cosine of the angle in degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let thirty_degrees = Degrees::new(30.0);
    /// let cos = thirty_degrees.cos();
    /// ```
    #[inline]
    pub fn cos(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).cos()
    }

    /// Returns the sine of the angle in degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let sixty_degrees = Degrees::new(60.0);
    /// let sin = sixty_degrees.sin();
    /// ```
    #[inline]
    pub fn sin(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).sin()
    }

    /// Returns the tangent of the angle in degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let forty_five_degrees = Degrees::new(45.0);
    /// let tan = forty_five_degrees.tan();
    /// ```
    #[inline]
    pub fn tan(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).tan()
    }

    /// Returns the secant of the angle in degrees.
    ///
    /// The secant is the reciprocal of the cosine.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let sixty_degrees = Degrees::new(60.0);
    /// let sec = sixty_degrees.sec();
    /// ```
    #[inline]
    pub fn sec(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).cos().recip()
    }

    /// Returns the cosecant of the angle in degrees.
    ///
    /// The cosecant is the reciprocal of the sine.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let forty_five_degrees = Degrees::new(45.0);
    /// let csc = forty_five_degrees.csc();
    /// ```
    #[inline]
    pub fn csc(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).sin().recip()
    }

    /// Returns the cotangent of the angle in degrees.
    ///
    /// The cotangent is the reciprocal of the tangent.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let thirty_degrees = Degrees::new(30.0);
    /// let cot = thirty_degrees.cot();
    /// ```
    #[inline]
    pub fn cot(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).tan().recip()
    }

    /// Returns a unit vector in the direction of the angle in degrees.
    ///
    /// The `normal` method computes the cosine and sine of the angle in degrees
    /// and returns a `axion::math::FVector2` representing the unit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let thirty_degrees = Degrees::new(30.0);
    /// let normal_vector = thirty_degrees.normal();
    /// ```
    #[inline]
    pub fn normal(self) -> crate::math::FVector2 {
        let (cos, sin) = {
            let radians = self.0 * Self::FACTOR_DEG_TO_RAD;
            (radians.cos(), radians.sin())
        };

        crate::math::FVector2 { x: cos, y: sin }
    }

    /// Computes the arccosine of the `scalar` value in degrees.
    ///
    /// The result is in the range \[`0°`, `180°`\], or NaN if the `scalar` is
    /// outside the range \[`-1.0`, `1.0`\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let scalar = 0.5;
    /// let angle = Degrees::acos(scalar);
    /// ```
    #[inline]
    pub fn acos(scalar: f32) -> Self {
        Self(scalar.acos() * Radians::FACTOR_RAD_TO_DEG)
    }

    /// Computes the arcsine of the `scalar` value in degrees.
    ///
    /// The result is in the range \[`-90°`, `90°`\], or `NaN` if the `scalar` is
    /// outside the range \[`-1.0`, `1.0`\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let scalar = 0.5;
    /// let angle = Degrees::asin(scalar);
    /// ```
    #[inline]
    pub fn asin(scalar: f32) -> Self {
        Self(scalar.asin() * Radians::FACTOR_RAD_TO_DEG)
    }

    /// Computes the arc tangent of the `scalar` value in degrees.
    ///
    /// The result is in the range \[`-90°`, `90°`\].
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let scalar = 0.5;
    /// let angle = Degrees::atan(scalar);
    /// ```

    #[inline]
    pub fn atan(scalar: f32) -> Self {
        Self(scalar.atan() * Radians::FACTOR_RAD_TO_DEG)
    }

    /// Computes the four-quadrant arctangent of the `y` and `x` coordinates
    /// in degrees, which corresponds to the vector direction.
    ///
    /// The result has the following range:
    ///
    ///   - When `x` = `0.0` and `y` = `0.0`: `0°`
    ///   - When `x` >= `0.0`: arctan(`y`/`x`) -> \[-90°, 90°\]
    ///   - When `y` >= `0.0`: arctan(`y`/`x`) + π -> (90°, 180°]
    ///   - When `y` < `0.0`: arctan(`y`/`x`) - π -> (-180°, -90°)
    ///
    /// # Parameters
    ///
    /// * `y`: The `y` coordinate of the vector.
    /// * `x`: The `x` coordinate of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::Degrees;
    ///
    /// let angle = Degrees::atan2(1.0, 1.0);
    /// ```
    #[inline]
    pub fn atan2(y: f32, x: f32) -> Self {
        Self(f32::atan2(y, x) * Radians::FACTOR_RAD_TO_DEG)
    }

    /// Returns the hyperbolic cosine of the angle in degrees.
    #[inline]
    pub fn cosh(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).cosh()
    }

    /// Returns the hyperbolic sine of the angle in degrees.
    #[inline]
    pub fn sinh(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).sinh()
    }

    /// Returns the hyperbolic tangent of the angle in degrees.
    #[inline]
    pub fn tanh(self) -> f32 {
        (self.0 * Self::FACTOR_DEG_TO_RAD).tanh()
    }

    /// Returns the inverse hyperbolic cosine of the angle in degrees.
    #[inline]
    pub fn acosh(self) -> Self {
        Self(self.0.acosh() * Radians::FACTOR_RAD_TO_DEG)
    }

    /// Returns the inverse hyperbolic sine of the angle in degrees.
    #[inline]
    pub fn asinh(self) -> Self {
        Self(self.0.asinh() * Radians::FACTOR_RAD_TO_DEG)
    }

    /// Returns the inverse hyperbolic tangent of the angle in degrees.
    #[inline]
    pub fn atanh(self) -> Self {
        Self(self.0.atanh() * Radians::FACTOR_RAD_TO_DEG)
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

impl ops::Add for Degrees {
    type Output = Self;

    /// Adds two `Degrees` values together.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Degrees` value.
    /// - `rhs`: The right-hand side `Degrees` value to be added.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value resulting from the addition.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl ops::Sub for Degrees {
    type Output = Self;

    /// Subtracts one `Degrees` value from another.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Degrees` value.
    /// - `rhs`: The right-hand side `Degrees` value to be subtracted.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value resulting from the subtraction.
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl ops::Mul for Degrees {
    type Output = Self;

    /// Multiplies two `Degrees` values together.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Degrees` value.
    /// - `rhs`: The right-hand side `Degrees` value to be multiplied.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value resulting from the multiplication.
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl ops::Div for Degrees {
    type Output = Self;

    /// Divides one `Degrees` value by another.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Degrees` value.
    /// - `rhs`: The right-hand side `Degrees` value to be divided by.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value resulting from the division.
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl ops::Rem for Degrees {
    type Output = Self;

    /// Computes the remainder of dividing one `Degrees` value by another.
    ///
    /// # Parameters
    ///
    /// - `self`: The left-hand side `Degrees` value.
    /// - `rhs`: The right-hand side `Degrees` value to compute the remainder with.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value representing the remainder.
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

impl ops::Add<f32> for Degrees {
    type Output = Self;

    /// Adds a `f32` value to a `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Degrees` value.
    /// - `rhs`: The `f32` value to be added.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value resulting from the addition.
    #[inline]
    fn add(self, rhs: f32) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl ops::Sub<f32> for Degrees {
    type Output = Self;

    /// Subtracts a `f32` value from a `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Degrees` value.
    /// - `rhs`: The `f32` value to be subtracted.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value resulting from the subtraction.
    #[inline]
    fn sub(self, rhs: f32) -> Self::Output {
        Self(self.0 - rhs)
    }
}

impl ops::Mul<f32> for Degrees {
    type Output = Self;

    /// Multiplies a `Degrees` value by a `f32` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Degrees` value.
    /// - `rhs`: The `f32` value to be multiplied.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value resulting from the multiplication.
    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl ops::Div<f32> for Degrees {
    type Output = Self;

    /// Divides a `Degrees` value by a `f32` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Degrees` value.
    /// - `rhs`: The `f32` value to divide by.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value resulting from the division.
    #[inline]
    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl ops::Rem<f32> for Degrees {
    type Output = Self;

    /// Computes the remainder of dividing a `Degrees` value by a `f32` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The `Degrees` value.
    /// - `rhs`: The `f32` value to compute the remainder with.
    ///
    /// # ReDegrees
    ///
    /// A new `Degrees` value representing the remainder.
    #[inline]
    fn rem(self, rhs: f32) -> Self::Output {
        Self(self.0 % rhs)
    }
}
impl ops::AddAssign for Degrees {
    /// Adds another `Degrees` value to this one and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `Degrees` value to be added.
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Self(self.0 + rhs.0)
    }
}

impl ops::SubAssign for Degrees {
    /// Subtracts another `Degrees` value from this one and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `Degrees` value to be subtracted.
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self(self.0 - rhs.0)
    }
}

impl ops::MulAssign for Degrees {
    /// Multiplies this `Degrees` value by another `Degrees` value and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `Degrees` value to be multiplied.
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self(self.0 * rhs.0)
    }
}

impl ops::DivAssign for Degrees {
    /// Divides this `Degrees` value by another `Degrees` value and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `Degrees` value to divide by.
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Self(self.0 / rhs.0)
    }
}

impl ops::RemAssign for Degrees {
    /// Computes the remainder of dividing this `Degrees` value by another `Degrees` value and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `Degrees` value to compute the remainder with.
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = Self(self.0 % rhs.0)
    }
}

impl ops::AddAssign<f32> for Degrees {
    /// Adds a `f32` scalar to this `Degrees` value and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `f32` scalar to be added.
    #[inline]
    fn add_assign(&mut self, rhs: f32) {
        *self = Self(self.0 + rhs)
    }
}

impl ops::SubAssign<f32> for Degrees {
    /// Subtracts a `f32` scalar from this `Degrees` value and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `f32` scalar to be subtracted.
    #[inline]
    fn sub_assign(&mut self, rhs: f32) {
        *self = Self(self.0 - rhs)
    }
}

impl ops::MulAssign<f32> for Degrees {
    /// Multiplies this `Degrees` value by a `f32` scalar and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `f32` scalar to be multiplied.
    #[inline]
    fn mul_assign(&mut self, rhs: f32) {
        *self = Self(self.0 * rhs)
    }
}

impl ops::DivAssign<f32> for Degrees {
    /// Divides this `Degrees` value by a `f32` scalar and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `f32` scalar to divide by.
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        *self = Self(self.0 / rhs)
    }
}

impl ops::RemAssign<f32> for Degrees {
    /// Computes the remainder of dividing this `Degrees` value by a `f32` scalar and assigns the result to this `Degrees` value.
    ///
    /// # Parameters
    ///
    /// - `self`: The mutable reference to this `Degrees` value.
    /// - `rhs`: The `f32` scalar to compute the remainder with.
    #[inline]
    fn rem_assign(&mut self, rhs: f32) {
        *self = Self(self.0 % rhs)
    }
}
