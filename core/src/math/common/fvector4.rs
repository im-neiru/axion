use std::fmt;
use std::ops;

use crate::math::{FVector2, FVector3};

/// `FVector4` is a structure that represents a 3D vector with `f32` components.
/// It encapsulates three floating-point values and is used for various purposes in graphical applications
/// including points, vectors, and texture coordinates.
#[derive(Clone, Copy, Debug)]
pub struct FVector4 {
    /// The X component of the vector.
    pub x: f32,
    /// The Y component of the vector.
    pub y: f32,
    /// The Z component of the vector.
    pub z: f32,
    /// The W component of the vector.
    pub w: f32,
}

/// Convenience function for creating a 4D vector (FVector4).
///
/// This function is a convenient way to create a 4D vector (FVector4)
/// with the given components.
///
/// # Arguments
///
/// * `x` - The x-component of the vector.
/// * `y` - The y-component of the vector.
/// * `z` - The z-component of the vector.
/// * `w` - The w-component of the vector.
///
/// # Returns
///
/// A new `FVector4` with the specified components.
///
/// # Example
///
/// ```
/// use axion::math::{FVector4, vec4};
///
/// let vector = vec4(1.0, 2.0, 3.0, 4.0); // Create a 4D vector
/// ```
#[inline(always)]
pub const fn vec4(x: f32, y: f32, z: f32, w: f32) -> FVector4 {
    FVector4 { x, y, z, w }
}

impl Default for FVector4 {
    /// Creates a new `FVector4` with all components initialized to the `f32::default()`.
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let default_vector: FVector4 = Default::default();
    /// assert_eq!(default_vector, FVector4::new(f32::default(), f32::default(), f32::default(), f32::default()));
    /// ```
    #[inline(always)]
    fn default() -> Self {
        Self {
            x: f32::default(),
            y: f32::default(),
            z: f32::default(),
            w: f32::default(),
        }
    }
}

impl FVector4 {
    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to 0.0.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to 0.0.
    pub const ZERO: Self = Self::splat(0.0);

    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to 1.0.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to 1.0.
    pub const ONE: Self = Self::splat(1.0);

    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to -1.0.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to -1.0.
    pub const NEG_ONE: Self = Self::splat(-1.0);

    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to the minimum finite value representable by `f32`.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to the minimum finite value representable by `f32`.
    pub const MIN: Self = Self::splat(f32::MIN);

    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to the maximum finite value representable by `f32`.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to the maximum finite value representable by `f32`.
    pub const MAX: Self = Self::splat(f32::MAX);

    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to a NaN (Not-a-Number) value.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to a NaN (Not-a-Number) value.
    pub const NAN: Self = Self::splat(f32::NAN);

    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to positive infinity.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to positive infinity.
    pub const INFINITY: Self = Self::splat(f32::INFINITY);

    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to negative infinity.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to negative infinity.
    pub const NEG_INFINITY: Self = Self::splat(f32::NEG_INFINITY);

    /// A constant `FVector4` instance with all `x`, `y`, `z` and `w` components set to the smallest positive value representable by `f32`.
    ///
    /// This constant represents an `FVector4` with all `x`, `y`, `z` and `w` components
    /// initialized to the smallest positive value representable by `f32`.
    pub const EPSILON: Self = Self::splat(f32::EPSILON);

    /// A constant `FVector4` representing the positive X-axis.
    ///
    /// This constant vector has a value of (1.0, 0.0, 0.0, 0.0), representing the positive X-axis in 4D space.
    pub const AXIS_X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// A constant `FVector4` representing the positive Y-axis.
    ///
    /// This constant vector has a value of (0.0, 1.0, 0.0, 0.0), representing the positive Y-axis in 4D space.
    pub const AXIS_Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
        w: 0.0,
    };

    /// A constant `FVector4` representing the positive Z-axis.
    ///
    /// This constant vector has a value of (0.0, 0.0, 1.0, 0.0), representing the positive Z-axis in 4D space.
    pub const AXIS_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
        w: 0.0,
    };

    /// A constant `FVector4` representing the positive W-axis.
    ///
    /// This constant vector has a value of (0.0, 0.0, 0.0, 1.0), representing the positive W-axis in 4D space.
    pub const AXIS_W: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    /// A constant `FVector4` representing the negative X-axis.
    ///
    /// This constant vector has a value of (-1.0, 0.0, 0.0, 0.0), representing the negative X-axis in 4D space.
    pub const NEG_AXIS_X: Self = Self {
        x: -1.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// A constant `FVector4` representing the negative Y-axis.
    ///
    /// This constant vector has a value of (0.0, -1.0, 0.0, 0.0), representing the negative Y-axis in 4D space.
    pub const NEG_AXIS_Y: Self = Self {
        x: 0.0,
        y: -1.0,
        z: 0.0,
        w: 0.0,
    };

    /// A constant `FVector4` representing the negative Z-axis.
    ///
    /// This constant vector has a value of (0.0, 0.0, -1.0, 0.0), representing the negative Z-axis in 4D space.
    pub const NEG_AXIS_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: -1.0,
        w: 0.0,
    };

    /// A constant `FVector4` representing the negative W-axis.
    ///
    /// This constant vector has a value of (0.0, 0.0, 0.0, -1.0), representing the negative W-axis in 4D space.
    pub const NEG_AXIS_W: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: -1.0,
    };

    /// Constructs a new `FVector4`.
    ///
    /// # Arguments
    ///
    /// * `x` - A float that holds the x component of the vector.
    /// * `y` - A float that holds the y component of the vector.
    /// * `z` - A float that holds the z component of the vector.
    /// * `w` - A float that holds the w component of the vector.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Create a new instance of `FVector4` with all `x`, `y`, `z` and `w` components set to the given `value`.
    ///
    /// This function creates a new `FVector4` instance with all `x`, `y`, `z` and `w` components
    /// initialized to the specified `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to set for all `x`, `y`, `z` and `w` components of the `FVector4`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance with all `x`, `y`, `z` and `w` components set to `value`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::splat(5.0);
    /// assert_eq!(vector.x, 5.0);
    /// assert_eq!(vector.y, 5.0);
    /// assert_eq!(vector.z, 5.0);
    /// assert_eq!(vector.w, 5.0);
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
            w: value,
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

    /// Returns the x and y components of the vector as an FVector2.
    #[inline]
    pub const fn axis_xy(self) -> FVector2 {
        FVector2 {
            x: self.x,
            y: self.y,
        }
    }

    /// Returns the x and z components of the vector as an FVector2.
    #[inline]
    pub const fn axis_xz(self) -> FVector2 {
        FVector2 {
            x: self.x,
            y: self.z,
        }
    }

    /// Returns the y and x components of the vector as an FVector2.
    #[inline]
    pub const fn axis_yx(self) -> FVector2 {
        FVector2 {
            x: self.y,
            y: self.x,
        }
    }

    /// Returns the y and z components of the vector as an FVector2.
    #[inline]
    pub const fn axis_yz(self) -> FVector2 {
        FVector2 {
            x: self.y,
            y: self.z,
        }
    }

    /// Returns the z and x components of the vector as an FVector2.
    #[inline]
    pub const fn axis_zx(self) -> FVector2 {
        FVector2 {
            x: self.z,
            y: self.x,
        }
    }

    /// Returns the z and y components of the vector as an FVector2.
    #[inline]
    pub const fn axis_zy(self) -> FVector2 {
        FVector2 {
            x: self.z,
            y: self.y,
        }
    }

    /// Returns the x component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_xx(self) -> FVector2 {
        FVector2 {
            x: self.x,
            y: self.x,
        }
    }

    /// Returns the y component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_yy(self) -> FVector2 {
        FVector2 {
            x: self.y,
            y: self.y,
        }
    }

    /// Returns the z component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_zz(self) -> FVector2 {
        FVector2 {
            x: self.z,
            y: self.z,
        }
    }

    /// Returns the z, y, and x components of the vector as an `FVector3`.
    #[inline]
    pub const fn axis_zyx(self) -> FVector3 {
        FVector3 {
            x: self.z,
            y: self.y,
            z: self.x,
        }
    }

    /// Returns the x, z, and y components of the vector as an `FVector3`.
    #[inline]
    pub const fn axis_xzy(self) -> FVector3 {
        FVector3 {
            x: self.x,
            y: self.z,
            z: self.y,
        }
    }

    /// Returns the y, z, and x components of the vector as an `FVector3`.
    #[inline]
    pub const fn axis_yzx(self) -> FVector3 {
        FVector3 {
            x: self.y,
            y: self.z,
            z: self.x,
        }
    }

    /// Returns the y, x, and z components of the vector as an `FVector3`.
    #[inline]
    pub const fn axis_yxz(self) -> FVector3 {
        FVector3 {
            x: self.y,
            y: self.x,
            z: self.z,
        }
    }

    /// Returns the z, x, and y components of the vector as an `FVector3`.
    #[inline]
    pub const fn axis_zxy(self) -> FVector3 {
        FVector3 {
            x: self.z,
            y: self.x,
            z: self.y,
        }
    }

    /// Returns the x component of the vector three times as an `FVector3`.
    #[inline]
    pub const fn axis_xxx(self) -> FVector3 {
        FVector3 {
            x: self.x,
            y: self.x,
            z: self.x,
        }
    }

    /// Returns the y component of the vector three times as an FVector4.
    #[inline]
    pub const fn axis_yyy(self) -> FVector3 {
        FVector3 {
            x: self.y,
            y: self.y,
            z: self.y,
        }
    }

    /// Returns the z component of the vector three times as an FVector4.
    #[inline]
    pub const fn axis_zzz(self) -> FVector3 {
        FVector3 {
            x: self.z,
            y: self.z,
            z: self.z,
        }
    }

    /// Checks if all components of the vector are finite numbers.
    ///
    /// This method returns `true` if all components of the vector are finite (neither infinite nor NaN), and `false`
    /// otherwise.
    ///
    /// # Returns
    ///
    /// `true` if all components of the vector are finite numbers; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let finite_vector = FVector4::new(1.0, 2.0, -3.0, 4.0);
    /// let non_finite_vector = FVector4::new(1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY);
    ///
    /// assert_eq!(finite_vector.is_finite(), true);
    /// assert_eq!(non_finite_vector.is_finite(), false);
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite()
            && self.y.is_finite()
            && self.z.is_finite()
            && self.w.is_finite()
    }
}

impl fmt::Display for FVector4 {
    /// Formats the `FVector4` as a string in the form *(x, y, z, w)*.
    ///
    /// This implementation allows you to format an `FVector4` instance as a string,
    /// where the `x`, `y`, `z` and `w` components are enclosed in parentheses and separated by a comma.
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
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(2.0, 5.0, 1.0, 1.2); // Create a 4D vector
    /// let formatted = format!("{}", vector);
    ///
    /// assert_eq!(formatted, "(2.0, 5.0, 1.0, 1.2)"); // Check the formatted string representation
    /// ```
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("(")?;
        self.x.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.y.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.z.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.w.fmt(formatter)?;
        formatter.write_str(")")
    }
}

impl ops::Index<usize> for FVector4 {
    type Output = f32;

    /// Indexes the `FVector3` by a `usize` index.
    ///
    /// This implementation allows you to access the components of an `FVector3` using a
    /// `usize` index where `0` corresponds to `x`, `1` corresponds to `y`, and `2` corresponds to `z`. It returns
    /// a reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access, where `0` represents `x`, `1` represents `y`, and `2` represents `z`.
    ///
    /// # Returns
    ///
    /// A reference to the specified component of the `FVector3`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not in 0 to 2).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
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

impl ops::IndexMut<usize> for FVector4 {
    /// Mutable indexing for the `FVector3` by a `usize` index.
    ///
    /// This implementation allows you to mutably access and modify the components of an `FVector3`
    /// using a `usize` index where `0` corresponds to `x`, `1` corresponds to `y`, and `2` corresponds to `z`. It returns
    /// a mutable reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access and modify, where `0` represents `x`, `1` represents `y`, and `2` represents `z`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the specified component of the `FVector3`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not in 0 to 2).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let mut vector = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
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

impl ops::Index<u32> for FVector4 {
    type Output = f32;

    /// Indexes the `FVector3` by a `u32` index.
    ///
    /// This implementation allows you to access the components of an `FVector3` using a
    /// `u32` index where `0` corresponds to `x`, `1` corresponds to `y`, and `2` corresponds to `z`. It returns
    /// a reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access, where `0` represents `x`, `1` represents `y`, and `2` represents `z`.
    ///
    /// # Returns
    ///
    /// A reference to the specified component of the `FVector3`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not in 0 to 2).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let vector = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
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

impl ops::IndexMut<u32> for FVector4 {
    /// Mutable indexing for the `FVector3` by a `u32` index.
    ///
    /// This implementation allows you to mutably access and modify the components of an `FVector3`
    /// using a `u32` index where `0` corresponds to `x`, `1` corresponds to `y`, and `2` corresponds to `z`. It returns
    /// a mutable reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access and modify, where `0` represents `x`, `1` represents `y`, and `2` represents `z`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the specified component of the `FVector3`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not in 0 to 2).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector3;
    ///
    /// let mut vector = FVector3::new(2.0, 3.0, 4.0); // Create a 3D vector
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
