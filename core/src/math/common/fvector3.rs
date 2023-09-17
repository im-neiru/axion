use std::fmt;
use std::ops;

use crate::math::Radians;
use crate::math::{SphericalAngles, Vector2, Vector4};

/// `Vector3` is a structure that represents a 3D vector with `f32` components.
/// It encapsulates three floating-point values and is used for various purposes in graphical applications
/// including points, vectors, and texture coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Vector3 {
    /// The X component of the vector.
    pub x: f32,
    /// The Y component of the vector.
    pub y: f32,
    /// The Z component of the vector.
    pub z: f32,
}

/// Convenience function for creating a 3D vector (Vector3).
///
/// This function is a convenient way to create a 3D vector (Vector3)
/// with the given components.
///
/// # Arguments
///
/// * `x` - The x-component of the vector.
/// * `y` - The y-component of the vector.
/// * `z` - The z-component of the vector.
///
/// # Returns
///
/// A new `Vector3` with the specified components.
///
/// # Example
///
/// ```
/// use axion::math::{Vector3, vec3};
///
/// let vector = vec3(1.0, 2.0, 3.0); // Create a 3D vector
/// ```
#[inline(always)]
pub const fn vec3(x: f32, y: f32, z: f32) -> Vector3 {
    Vector3 { x, y, z }
}

impl Vector3 {
    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to 0.0.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to 0.0.
    pub const ZERO: Self = Self::splat(0.0);

    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to 1.0.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to 1.0.
    pub const ONE: Self = Self::splat(1.0);

    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to -1.0.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to -1.0.
    pub const NEG_ONE: Self = Self::splat(-1.0);

    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to the minimum finite value representable by `f32`.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to the minimum finite value representable by `f32`.
    pub const MIN: Self = Self::splat(f32::MIN);

    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to the maximum finite value representable by `f32`.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to the maximum finite value representable by `f32`.
    pub const MAX: Self = Self::splat(f32::MAX);

    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to a NaN (Not-a-Number) value.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to a NaN (Not-a-Number) value.
    pub const NAN: Self = Self::splat(f32::NAN);

    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to positive infinity.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to positive infinity.
    pub const INFINITY: Self = Self::splat(f32::INFINITY);

    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to negative infinity.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to negative infinity.
    pub const NEG_INFINITY: Self = Self::splat(f32::NEG_INFINITY);

    /// A constant `Vector3` instance with all `x`, `y` and `z` components set to the smallest positive value representable by `f32`.
    ///
    /// This constant represents an `Vector3` with all `x`, `y` and `z` components
    /// initialized to the smallest positive value representable by `f32`.
    pub const EPSILON: Self = Self::splat(f32::EPSILON);

    /// A constant `Vector3` representing the positive X-axis.
    ///
    /// This constant vector has a value of (1.0, 0.0, 0.0), representing the positive X-axis in 3D space.
    pub const AXIS_X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// A constant `Vector3` representing the positive Y-axis.
    ///
    /// This constant vector has a value of (0.0, 1.0, 0.0), representing the positive Y-axis in 3D space.
    pub const AXIS_Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// A constant `Vector3` representing the positive Z-axis.
    ///
    /// This constant vector has a value of (0.0, 0.0, 1.0), representing the positive Z-axis in 3D space.
    pub const AXIS_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// A constant `Vector3` representing the negative X-axis.
    ///
    /// This constant vector has a value of (-1.0, 0.0, 0.0), representing the negative X-axis in 3D space.
    pub const NEG_AXIS_X: Self = Self {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    };

    /// A constant `Vector3` representing the negative Y-axis.
    ///
    /// This constant vector has a value of (0.0, -1.0, 0.0), representing the negative Y-axis in 3D space.
    pub const NEG_AXIS_Y: Self = Self {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    };

    /// A constant `Vector3` representing the negative Z-axis.
    ///
    /// This constant vector has a value of (0.0, 0.0, -1.0), representing the negative Z-axis in 3D space.
    pub const NEG_AXIS_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };

    /// Constructs a new `Vector3`.
    ///
    /// # Arguments
    ///
    /// * `x` - A float that holds the x component of the vector.
    /// * `y` - A float that holds the y component of the vector.
    /// * `z` - A float that holds the z component of the vector.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Converts the `Vector3` into a `Vector4` by adding a fourth component `w`.
    ///
    /// This method creates a new `Vector4` by taking the `x`, `y`, and `z` components of the
    /// `Vector3` and adding a fourth component `w`. This can be useful when working with
    /// homogeneous coordinates, where the fourth component typically represents a weight or scale.
    ///
    /// # Arguments
    ///
    /// * `w` - The value of the fourth component to add to the resulting `Vector4`.
    ///
    /// # Returns
    ///
    /// A new `Vector4` with `x`, `y`, `z`, and `w` components.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::Vector3;
    ///
    /// let vector3 = Vector3::new(2.0, 3.0, 4.0);
    /// let vector4 = vector3.add_axis(1.0);
    ///
    /// assert_eq!(vector4, Vector4::new(2.0, 3.0, 4.0, 1.0));
    /// ```
    #[inline]
    pub const fn add_axis(self, w: f32) -> Vector4 {
        Vector4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }

    /// Create a new instance of `Vector3` with all `x`, `y` and `z` components set to the given `value`.
    ///
    /// This function creates a new `Vector3` instance with all `x`, `y` and `z` components
    /// initialized to the specified `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to set for all `x`, `y` and `z` components of the `Vector3`.
    ///
    /// # Returns
    ///
    /// A new `Vector3` instance with all `x`, `y` and `z` components set to `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::Vector3;
    ///
    /// let vector = Vector3::splat(5.0);
    /// assert_eq!(vector.x, 5.0);
    /// assert_eq!(vector.y, 5.0);
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }

    /// Returns the x and y components of the vector as a tuple.
    #[inline]
    pub const fn xy(self) -> (f32, f32) {
        (self.x, self.y)
    }

    /// Returns the x and z components of the vector as a tuple.
    #[inline]
    pub const fn xz(self) -> (f32, f32) {
        (self.x, self.z)
    }

    /// Returns the y and x components of the vector as a tuple.
    #[inline]
    pub const fn yx(self) -> (f32, f32) {
        (self.y, self.x)
    }

    /// Returns the y and z components of the vector as a tuple.
    #[inline]
    pub const fn yz(self) -> (f32, f32) {
        (self.y, self.z)
    }

    /// Returns the z and x components of the vector as a tuple.
    #[inline]
    pub const fn zx(self) -> (f32, f32) {
        (self.z, self.x)
    }

    /// Returns the z and y components of the vector as a tuple.
    #[inline]
    pub const fn zy(self) -> (f32, f32) {
        (self.z, self.y)
    }

    /// Returns the x component of the vector twice as a tuple.
    #[inline]
    pub const fn xx(self) -> (f32, f32) {
        (self.x, self.x)
    }

    /// Returns the y component of the vector twice as a tuple.
    #[inline]
    pub const fn yy(self) -> (f32, f32) {
        (self.y, self.y)
    }

    /// Returns the z component of the vector twice as a tuple.
    #[inline]
    pub const fn zz(self) -> (f32, f32) {
        (self.z, self.z)
    }

    /// Returns the x, y, and z components of the vector as a tuple.
    #[inline]
    pub const fn xyz(self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }

    /// Returns the z, y, and x components of the vector as a tuple.
    #[inline]
    pub const fn zyx(self) -> (f32, f32, f32) {
        (self.z, self.y, self.x)
    }

    /// Returns the x, z, and y components of the vector as a tuple.
    #[inline]
    pub const fn xzy(self) -> (f32, f32, f32) {
        (self.x, self.z, self.y)
    }

    /// Returns the y, z, and x components of the vector as a tuple.
    #[inline]
    pub const fn yzx(self) -> (f32, f32, f32) {
        (self.y, self.z, self.x)
    }

    /// Returns the y, x, and z components of the vector as a tuple.
    #[inline]
    pub const fn yxz(self) -> (f32, f32, f32) {
        (self.y, self.x, self.z)
    }

    /// Returns the z, x, and y components of the vector as a tuple.
    #[inline]
    pub const fn zxy(self) -> (f32, f32, f32) {
        (self.z, self.x, self.y)
    }

    /// Returns the x component of the vector three times as a tuple.
    #[inline]
    pub const fn xxx(self) -> (f32, f32, f32) {
        (self.x, self.x, self.x)
    }

    /// Returns the y component of the vector three times as a tuple.
    #[inline]
    pub const fn yyy(self) -> (f32, f32, f32) {
        (self.y, self.y, self.y)
    }

    /// Returns the z component of the vector three times as a tuple.
    #[inline]
    pub const fn zzz(self) -> (f32, f32, f32) {
        (self.z, self.z, self.z)
    }

    /// Returns the x and y components of the vector as an `Vector2`.
    #[inline]
    pub const fn axis_xy(self) -> Vector2 {
        Vector2 {
            x: self.x,
            y: self.y,
        }
    }

    /// Returns the x and z components of the vector as an `Vector2`.
    #[inline]
    pub const fn axis_xz(self) -> Vector2 {
        Vector2 {
            x: self.x,
            y: self.z,
        }
    }

    /// Returns the y and x components of the vector as an `Vector2`.
    #[inline]
    pub const fn axis_yx(self) -> Vector2 {
        Vector2 {
            x: self.y,
            y: self.x,
        }
    }

    /// Returns the y and z components of the vector as an `Vector2`.
    #[inline]
    pub const fn axis_yz(self) -> Vector2 {
        Vector2 {
            x: self.y,
            y: self.z,
        }
    }

    /// Returns the z and x components of the vector as an `Vector2`.
    #[inline]
    pub const fn axis_zx(self) -> Vector2 {
        Vector2 {
            x: self.z,
            y: self.x,
        }
    }

    /// Returns the z and y components of the vector as an `Vector2`.
    #[inline]
    pub const fn axis_zy(self) -> Vector2 {
        Vector2 {
            x: self.z,
            y: self.y,
        }
    }

    /// Returns the x component of the vector twice as an `Vector2`.
    #[inline]
    pub const fn axis_xx(self) -> Vector2 {
        Vector2 {
            x: self.x,
            y: self.x,
        }
    }

    /// Returns the y component of the vector twice as an `Vector2`.
    #[inline]
    pub const fn axis_yy(self) -> Vector2 {
        Vector2 {
            x: self.y,
            y: self.y,
        }
    }

    /// Returns the z component of the vector twice as an `Vector2`.
    #[inline]
    pub const fn axis_zz(self) -> Vector2 {
        Vector2 {
            x: self.z,
            y: self.z,
        }
    }

    /// Returns the z, y, and x components of the vector as an `Vector3`.
    #[inline]
    pub const fn axis_zyx(self) -> Vector3 {
        Vector3 {
            x: self.z,
            y: self.y,
            z: self.x,
        }
    }

    /// Returns the x, z, and y components of the vector as an `Vector3`.
    #[inline]
    pub const fn axis_xzy(self) -> Vector3 {
        Vector3 {
            x: self.x,
            y: self.z,
            z: self.y,
        }
    }

    /// Returns the y, z, and x components of the vector as an `Vector3`.
    #[inline]
    pub const fn axis_yzx(self) -> Vector3 {
        Vector3 {
            x: self.y,
            y: self.z,
            z: self.x,
        }
    }

    /// Returns the y, x, and z components of the vector as an `Vector3`.
    #[inline]
    pub const fn axis_yxz(self) -> Vector3 {
        Vector3 {
            x: self.y,
            y: self.x,
            z: self.z,
        }
    }

    /// Returns the z, x, and y components of the vector as an `Vector3`.
    #[inline]
    pub const fn axis_zxy(self) -> Vector3 {
        Vector3 {
            x: self.z,
            y: self.x,
            z: self.y,
        }
    }

    /// Returns the x component of the vector three times as an `Vector3`.
    #[inline]
    pub const fn axis_xxx(self) -> Vector3 {
        Vector3 {
            x: self.x,
            y: self.x,
            z: self.x,
        }
    }

    /// Returns the y component of the vector three times as an `Vector3`.
    #[inline]
    pub const fn axis_yyy(self) -> Vector3 {
        Vector3 {
            x: self.y,
            y: self.y,
            z: self.y,
        }
    }

    /// Returns the z component of the vector three times as an `Vector3`.
    #[inline]
    pub const fn axis_zzz(self) -> Vector3 {
        Vector3 {
            x: self.z,
            y: self.z,
            z: self.z,
        }
    }

    /// Checks if any component of the `Vector3` is finite.
    ///
    /// This function checks each component of the `Vector3` for finiteness and returns
    /// `true` if at least one component is finite, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// Returns `true` if any component of the `Vector3` is finite, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::Vector3;
    ///
    /// // Create an `Vector3` with all finite components
    /// let finite_vector = Vector3::new(2.0, 3.0, -1.0);
    ///
    /// // Create an `Vector3` with at least one infinite component
    /// let infinite_vector = Vector3::new(f32::INFINITY, 2.0, 0.0);
    ///
    /// // Check if any component of the finite vector is finite
    /// assert_eq!(finite_vector.is_finite(), true);
    ///
    /// // Check if any component of the infinite vector is finite
    /// assert_eq!(infinite_vector.is_finite(), false);
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Calculates the spherical angles from this point to another point in radians.
    ///
    /// This method calculates the spherical angles from this point to another point
    /// specified by `other` in radians. It computes the azimuthal angle and the polar
    /// angle, which represent the direction from this point to the `other` point.
    ///
    /// # Parameters
    ///
    /// - `other`: The other `Vector3` point to calculate angles to.
    ///
    /// # Returns
    ///
    /// Spherical angles in radians representing the direction from this point to `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{Vector3, Radians};
    ///
    /// let point1 = Vector3::new(1.0, 0.0, 0.0);
    /// let point2 = Vector3::new(1.0, 1.0, 0.0);
    ///
    /// let angles = point1.angles_to(point2);
    /// ```
    #[inline]
    pub fn angles_to(self, other: Self) -> SphericalAngles<Radians> {
        let diff = other - self;
        let xz = diff.axis_xz();

        SphericalAngles {
            azimuthal: xz.azimuthal(),
            polar: Radians::atan2(diff.y, xz.length()),
        }
    }

    /// Calculates the spherical angles for this point in radians.
    ///
    /// This method calculates the spherical angles for this point in radians.
    /// It computes the azimuthal angle and the polar angle, which represent the
    /// direction from the origin to this point.
    ///
    /// # Returns
    ///
    /// Spherical angles in radians representing the direction from the origin to this point.
    ///
    /// # Examples
    ///
    /// ```
    /// use axion::math::{Vector3, Radians};
    ///
    /// let point = Vector3::new(1.0, 0.0, 0.0);
    ///
    /// let angles = point.spherical_angles();
    /// ```
    #[inline]
    pub fn spherical_angles(self) -> SphericalAngles<Radians> {
        let xz = self.axis_xz();

        SphericalAngles {
            azimuthal: xz.azimuthal(),
            polar: Radians::atan2(self.y, xz.length()),
        }
    }
}

impl Default for Vector3 {
    /// Creates a new `Vector3` with all components initialized to the `f32::default()`.
    /// # Example
    ///
    /// ```rust
    /// use axion::math::Vector3;
    ///
    /// let default_vector: Vector3 = Default::default();
    /// assert_eq!(default_vector, Vector3::new(f32::default(), f32::default(), f32::default()));
    /// ```
    #[inline(always)]
    fn default() -> Self {
        Self {
            x: f32::default(),
            y: f32::default(),
            z: f32::default(),
        }
    }
}

impl fmt::Display for Vector3 {
    /// Formats the `Vector3` as a string in the form *(x, y, z)*.
    ///
    /// This implementation allows you to format an `Vector3` instance as a string,
    /// where the `x`, `y`, and `z` components are enclosed in parentheses and separated by a comma.
    ///
    /// # Arguments
    ///
    /// * `formatter` - A mutable reference to the `fmt::Formatter` used for formatting.
    ///
    /// # Returns
    ///
    /// A `fmt::Result` indicating whether the formatting was successful.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::Vector3;
    ///
    /// let vector = Vector3::new(2.0, 5.0, 1.0); // Create a 3D vector
    /// let formatted = format!("{}", vector);
    ///
    /// assert_eq!(formatted, "(2.0, 5.0, 1.0)"); // Check the formatted string representation
    /// ```
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("(")?;
        self.x.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.y.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.z.fmt(formatter)?;
        formatter.write_str(")")
    }
}

impl ops::Index<usize> for Vector3 {
    type Output = f32;

    /// Indexes the `Vector3` by a `usize` index.
    ///
    /// This implementation allows you to access the components of an `Vector3` using a
    /// `usize` index where `0` corresponds to `x`, `1` corresponds to `y`, and `2` corresponds to `z`. It returns
    /// a reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access, where `0` represents `x`, `1` represents `y`, and `2` represents `z`.
    ///
    /// # Returns
    ///
    /// A reference to the specified component of the `Vector3`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not in 0 to 2).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::Vector3;
    ///
    /// let vector = Vector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let x_component = vector[0]; // Access the x component using index 0
    ///
    /// assert_eq!(x_component, 2.0); // Check if the accessed component is correct
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::IndexMut<usize> for Vector3 {
    /// Mutable indexing for the `Vector3` by a `usize` index.
    ///
    /// This implementation allows you to mutably access and modify the components of an `Vector3`
    /// using a `usize` index where `0` corresponds to `x`, `1` corresponds to `y`, and `2` corresponds to `z`. It returns
    /// a mutable reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access and modify, where `0` represents `x`, `1` represents `y`, and `2` represents `z`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the specified component of the `Vector3`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not in 0 to 2).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::Vector3;
    ///
    /// let mut vector = Vector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// vector[0] = 4.0; // Modify the x component using index 0
    ///
    /// assert_eq!(vector[0], 4.0); // Check if the modified component is correct
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::Index<u32> for Vector3 {
    type Output = f32;

    /// Indexes the `Vector3` by a `u32` index.
    ///
    /// This implementation allows you to access the components of an `Vector3` using a
    /// `u32` index where `0` corresponds to `x`, `1` corresponds to `y`, and `2` corresponds to `z`. It returns
    /// a reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access, where `0` represents `x`, `1` represents `y`, and `2` represents `z`.
    ///
    /// # Returns
    ///
    /// A reference to the specified component of the `Vector3`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not in 0 to 2).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::Vector3;
    ///
    /// let vector = Vector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let x_component = vector[0u32]; // Access the x component using a u32 index
    ///
    /// assert_eq!(x_component, 2.0); // Check if the accessed component is correct
    /// ```
    #[inline]
    fn index(&self, index: u32) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::IndexMut<u32> for Vector3 {
    /// Mutable indexing for the `Vector3` by a `u32` index.
    ///
    /// This implementation allows you to mutably access and modify the components of an `Vector3`
    /// using a `u32` index where `0` corresponds to `x`, `1` corresponds to `y`, and `2` corresponds to `z`. It returns
    /// a mutable reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access and modify, where `0` represents `x`, `1` represents `y`, and `2` represents `z`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the specified component of the `Vector3`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not in 0 to 2).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::Vector3;
    ///
    /// let mut vector = Vector3::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// vector[0u32] = 4.0; // Modify the x component using a u32 index
    ///
    /// assert_eq!(vector[0u32], 4.0); // Check if the modified component is correct
    /// ```
    #[inline]
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index out of bounds"),
        }
    }
}
