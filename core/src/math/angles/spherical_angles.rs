use std::fmt;

use super::{Degrees, Radians, Turns};

/// A struct representing spherical angles.
///
/// This struct is used to hold spherical angles, typically representing
/// direction in a three-dimensional space. It includes two angles: azimuthal and polar.
///
/// - `azimuthal`: The azimuthal angle, which measures the angle in the horizontal plane
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
///     azimuthal: Radians::new(1.0),   // Azimuth angle in radians
///     polar: Radians::new(0.5),     // Polar angle in radians
/// };
/// ```
///
#[derive(Clone, Copy, Debug)]
pub struct SphericalAngles<T: super::Angle> {
    /// The azimuthal angle, typically in angular units like Radians, Degrees, or Turns.
    pub azimuthal: T,
    /// The polar angle, typically in angular units like Radians, Degrees, or Turns.
    pub polar: T,
}

impl<T: super::Angle> fmt::Display for SphericalAngles<T> {
    /// Formats `SphericalAngles` as a string in the format `"(azimuthal, polar)"`.
    ///
    /// This implementation formats `SphericalAngles` as a string with its azimuthal and polar
    /// angles enclosed in parentheses and separated by a comma and space.
    ///
    /// # Parameters
    ///
    /// - `self`: A reference to the `SphericalAngles` struct to be formatted.
    /// - `formatter`: A mutable reference to the `fmt::Formatter` used for formatting.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating whether the formatting operation was successful.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{SphericalAngles, Degrees};
    ///
    /// let angles = SphericalAngles {
    ///     azimuthal: Degrees::new(45.0),
    ///     polar: Degrees::new(30.0),
    /// };
    ///
    /// let formatted = format!("{}", angles);
    /// assert_eq!(formatted, "(45.0, 30.0)");
    /// ```
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("(")?;
        fmt::Display::fmt(&self.azimuthal, formatter)?;
        formatter.write_str(", ")?;
        fmt::Display::fmt(&self.polar, formatter)?;
        formatter.write_str(")")
    }
}

impl SphericalAngles<Radians> {
    /// Converts angles in radians to angles in degrees.
    ///
    /// This method takes angles in radians and returns equivalent angles in degrees.
    ///
    /// # Returns
    ///
    /// Spherical angles with azimuthal and polar angles represented in degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{SphericalAngles, Radians, Degrees};
    ///
    /// let angles_radians = SphericalAngles {
    ///     azimuthal: Radians::new(1.57),  // 90 degrees in radians
    ///     polar: Radians::new(0.785),  // 45 degrees in radians
    /// };
    ///
    /// let angles_degrees = angles_radians.into_degrees();
    /// ```
    #[inline]
    pub fn into_degrees(self) -> SphericalAngles<Degrees> {
        SphericalAngles {
            azimuthal: self.azimuthal.into_degrees(),
            polar: self.polar.into_degrees(),
        }
    }

    /// Converts angles in radians to angles in turns.
    ///
    /// This method takes angles in radians and returns equivalent angles in turns.
    ///
    /// # Returns
    ///
    /// Spherical angles with azimuthal and polar angles represented in turns.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{SphericalAngles, Radians, Turns};
    ///
    /// let angles_radians = SphericalAngles {
    ///     azimuthal: Radians::new(3.142),  // 180 degrees in radians
    ///     polar: Radians::new(1.571),   // 90 degrees in radians
    /// };
    ///
    /// let angles_turns = angles_radians.into_turns();
    /// ```
    #[inline]
    pub fn into_turns(self) -> SphericalAngles<Turns> {
        SphericalAngles {
            azimuthal: self.azimuthal.into_turns(),
            polar: self.polar.into_turns(),
        }
    }
}

impl SphericalAngles<Degrees> {
    /// Converts angles in degrees to angles in radians.
    ///
    /// This method takes angles in degrees and returns equivalent angles in radians.
    ///
    /// # Returns
    ///
    /// Spherical angles with azimuthal and polar angles represented in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{SphericalAngles, Degrees};
    ///
    /// let angles_degrees = SphericalAngles {
    ///     azimuthal: Degrees::new(90.0),
    ///     polar: Degrees::new(45.0),
    /// };
    ///
    /// let angles_radians = angles_degrees.into_radians();
    /// ```
    #[inline]
    pub fn into_radians(self) -> SphericalAngles<Radians> {
        SphericalAngles {
            azimuthal: self.azimuthal.into_radians(),
            polar: self.polar.into_radians(),
        }
    }

    /// Converts angles in degrees to angles in turns.
    ///
    /// This method takes angles in degrees and returns equivalent angles in turns.
    ///
    /// # Returns
    ///
    /// Spherical angles with azimuthal and polar angles represented in turns.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{SphericalAngles, Degrees, Turns};
    ///
    /// let angles_degrees = SphericalAngles {
    ///     azimuthal: Degrees::new(180.0),
    ///     polar: Degrees::new(90.0),
    /// };
    ///
    /// let angles_turns = angles_degrees.into_turns();
    /// ```
    #[inline]
    pub fn into_turns(self) -> SphericalAngles<Turns> {
        SphericalAngles {
            azimuthal: self.azimuthal.into_turns(),
            polar: self.polar.into_turns(),
        }
    }
}

impl SphericalAngles<Turns> {
    /// Converts angles in turns to angles in radians.
    ///
    /// This method takes angles in turns and returns equivalent angles in radians.
    ///
    /// # Returns
    ///
    /// Spherical angles with azimuthal and polar angles represented in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{SphericalAngles, Turns, Radians};
    ///
    /// let angles_turns = SphericalAngles {
    ///     azimuthal: Turns::new(0.5),
    ///     polar: Turns::new(1.0),
    /// };
    ///
    /// let angles_radians = angles_turns.into_radians();
    /// ```
    #[inline]
    pub fn into_radians(self) -> SphericalAngles<Radians> {
        SphericalAngles {
            azimuthal: self.azimuthal.into_radians(),
            polar: self.polar.into_radians(),
        }
    }

    /// Converts angles in turns to angles in degrees.
    ///
    /// This method takes angles in turns and returns equivalent angles in degrees.
    ///
    /// # Returns
    ///
    /// Spherical angles with azimuthal and polar angles represented in degrees.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{SphericalAngles, Turns, Degrees};
    ///
    /// let angles_turns = SphericalAngles {
    ///     azimuthal: Turns::new(0.25),
    ///     polar: Turns::new(0.75),
    /// };
    ///
    /// let angles_degrees = angles_turns.into_degrees();
    /// ```
    #[inline]
    pub fn into_degrees(self) -> SphericalAngles<Degrees> {
        SphericalAngles {
            azimuthal: self.azimuthal.into_degrees(),
            polar: self.polar.into_degrees(),
        }
    }
}
