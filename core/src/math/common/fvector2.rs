use std::fmt;
use std::ops;

use crate::math::{FVector2, FVector3};

impl FVector2 {
    /// A constant `FVector2` instance with both `x` and `y` components set to 0.0.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to 0.0.
    pub const ZERO: Self = Self::splat(0.0);

    /// A constant `FVector2` instance with both `x` and `y` components set to 1.0.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to 1.0.
    pub const ONE: Self = Self::splat(1.0);

    /// A constant `FVector2` instance with both `x` and `y` components set to -1.0.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to -1.0.
    pub const NEG_ONE: Self = Self::splat(-1.0);

    /// A constant `FVector2` instance with both `x` and `y` components set to the minimum finite value representable by `f32`.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to the minimum finite value representable by `f32`.
    pub const MIN: Self = Self::splat(f32::MIN);

    /// A constant `FVector2` instance with both `x` and `y` components set to the maximum finite value representable by `f32`.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to the maximum finite value representable by `f32`.
    pub const MAX: Self = Self::splat(f32::MAX);

    /// A constant `FVector2` instance with both `x` and `y` components set to a NaN (Not-a-Number) value.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to a NaN (Not-a-Number) value.
    pub const NAN: Self = Self::splat(f32::NAN);

    /// A constant `FVector2` instance with both `x` and `y` components set to positive infinity.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to positive infinity.
    pub const INFINITY: Self = Self::splat(f32::INFINITY);

    /// A constant `FVector2` instance with both `x` and `y` components set to negative infinity.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to negative infinity.
    pub const NEG_INFINITY: Self = Self::splat(f32::NEG_INFINITY);

    /// A constant `FVector2` instance with both `x` and `y` components set to the smallest positive value representable by `f32`.
    ///
    /// This constant represents an `FVector2` with both `x` and `y` components
    /// initialized to the smallest positive value representable by `f32`.
    pub const EPSILON: Self = Self::splat(f32::EPSILON);

    /// A constant `FVector2` representing the positive X-axis.
    ///
    /// This constant vector has a value of `(1.0, 0.0)`, representing the positive X-axis in 2D space.
    pub const AXIS_X: Self = Self { x: 1.0, y: 0.0 };

    /// A constant `FVector2` representing the positive Y-axis.
    ///
    /// This constant vector has a value of `(0.0, 1.0)`, representing the positive Y-axis in 2D space.
    pub const AXIS_Y: Self = Self { x: 0.0, y: 1.0 };

    /// A constant `FVector2` representing the negative X-axis.
    ///
    /// This constant vector has a value of `(-1.0, 0.0)`, representing the negative X-axis in 2D space.
    pub const NEG_AXIS_X: Self = Self { x: -1.0, y: 0.0 };

    /// A constant `FVector2` representing the negative Y-axis.
    ///
    /// This constant vector has a value of `(0.0, -1.0)`, representing the negative Y-axis in 2D space.
    pub const NEG_AXIS_Y: Self = Self { x: 0.0, y: -1.0 };

    /// Constructs a new `FVector2`.
    ///
    /// # Arguments
    ///
    /// * `x` - A float that holds the x component of the vector.
    /// * `y` - A float that holds the y component of the vector.
    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Adds the `z` axis value to a `FVector2`, creating a new `FVector3`.
    ///
    /// This function takes a `FVector2` and a `z` value, and returns a new `FVector3`
    /// with the `x` and `y` components from the input `FVector2` and the provided `z` value.
    ///
    /// # Arguments
    ///
    /// * `self` - The input `FVector2` that will serve as the basis for the `x` and `y` components of the resulting `FVector3`.
    /// * `z` - The value to be used as the `z` component of the resulting `FVector3`.
    ///
    /// # Returns
    ///
    /// A new `FVector3` with `x` and `y` components inherited from the input `FVector2`
    /// and the `z` component set to the provided `z` value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector2;
    ///
    /// let f2 = FVector2::new(1.0, 2.0);
    /// let f3 = f2.add_axis(3.0); // New FVector3
    ///
    /// assert_eq!(f3.x, 1.0);
    /// assert_eq!(f3.y, 2.0);
    /// assert_eq!(f3.z, 3.0);
    /// ```
    #[inline]
    pub const fn add_axis(self, z: f32) -> FVector3 {
        FVector3 {
            x: self.x,
            y: self.y,
            z,
        }
    }

    /// Create a new instance of `FVector2` with both `x` and `y` components set to the given `value`.
    ///
    /// This function creates a new `FVector2` instance with both `x` and `y` components
    /// initialized to the specified `value`.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to set for both `x` and `y` components of the `FVector2`.
    ///
    /// # Returns
    ///
    /// A new `FVector2` instance with both `x` and `y` components set to `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::splat(5.0);
    /// assert_eq!(vector.x, 5.0);
    /// assert_eq!(vector.y, 5.0);
    /// ```
    #[inline]
    pub const fn splat(value: f32) -> Self {
        Self { x: value, y: value }
    }

    /// Returns the x and y components of the vector as a tuple.
    #[inline]
    pub const fn xy(self) -> (f32, f32) {
        (self.x, self.y)
    }

    /// Returns the y and x components of the vector as a tuple.
    #[inline]
    pub const fn yx(self) -> (f32, f32) {
        (self.y, self.x)
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

    /// Returns the y and x components of the vector as an `FVector2`.
    #[inline]
    pub const fn axis_yx(self) -> Self {
        Self {
            x: self.y,
            y: self.x,
        }
    }

    /// Returns the x component of the vector twice as an `FVector2`.
    #[inline]
    pub const fn axis_xx(self) -> Self {
        Self {
            x: self.x,
            y: self.x,
        }
    }

    /// Returns the y component of the vector twice as an `FVector2`.
    #[inline]
    pub const fn axis_yy(self) -> Self {
        Self {
            x: self.y,
            y: self.y,
        }
    }

    /// Checks if any component of the `FVector2` is finite.
    ///
    /// This function checks each component of the `FVector2` for finiteness and returns
    /// `true` if at least one component is finite, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if any component of the `FVector2` is finite, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let finite_vector = FVector2::new(2.0, 3.0);
    /// let infinite_vector = FVector2::new(f32::INFINITY, 2.0);
    ///
    /// assert_eq!(finite_vector.is_finite(), true);
    /// assert_eq!(infinite_vector.is_finite(), false);
    /// ```
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl Default for FVector2 {
    /// Creates a new `FVector2` with all components initialized to the `f32::default()`.
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector2;
    ///
    /// let default_vector: FVector2 = Default::default();
    /// assert_eq!(default_vector, FVector2::new(f32::default(), f32::default()));
    /// ```
    #[inline(always)]
    fn default() -> Self {
        Self {
            x: f32::default(),
            y: f32::default(),
        }
    }
}

impl fmt::Display for FVector2 {
    /// Formats the `FVector2` as a string in the form *(x, y)*.
    ///
    /// This implementation allows you to format an `FVector2` instance as a string,
    /// where the `x` and `y` components are enclosed in parentheses and separated by a comma.
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
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::new(2.0, 5.0);
    /// let formatted = format!("{}", vector);
    ///
    /// assert_eq!(formatted, "(2.0, 5.0)");
    /// ```
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("(")?;
        self.x.fmt(formatter)?;
        formatter.write_str(", ")?;
        self.y.fmt(formatter)?;
        formatter.write_str(")")
    }
}

impl ops::Index<usize> for FVector2 {
    type Output = f32;

    /// Indexes the `FVector2` by a `usize` index.
    ///
    /// This implementation allows you to access the components of an `FVector2` using a
    /// `usize` index where `0` corresponds to `x` and `1` corresponds to `y`. It returns
    /// a reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access, where `0` represents `x` and `1` represents `y`.
    ///
    /// # Returns
    ///
    /// A reference to the specified component of the `FVector2`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not 0 or 1).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::new(2.0, 3.0);
    /// let x_component = vector[0];
    ///
    /// assert_eq!(x_component, 2.0);
    /// ```
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::IndexMut<usize> for FVector2 {
    /// Mutable indexing for the `FVector2` by a `usize` index.
    ///
    /// This implementation allows you to mutably access and modify the components of an `FVector2`
    /// using a `usize` index where `0` corresponds to `x` and `1` corresponds to `y`. It returns
    /// a mutable reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access and modify, where `0` represents `x` and `1` represents `y`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the specified component of the `FVector2`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not 0 or 1).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(2.0, 3.0);
    /// vector[0] = 4.0;
    ///
    /// assert_eq!(vector[0], 4.0);
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::Index<u32> for FVector2 {
    type Output = f32;

    /// Indexes the `FVector2` by a `u32` index.
    ///
    /// This implementation allows you to access the components of an `FVector2` using a
    /// `u32` index where `0` corresponds to `x` and `1` corresponds to `y`. It returns
    /// a reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access, where `0` represents `x` and `1` represents `y`.
    ///
    /// # Returns
    ///
    /// A reference to the specified component of the `FVector2`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not 0 or 1).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let vector = FVector2::new(2.0, 3.0);
    /// let x_component = vector[0u32];
    ///
    /// assert_eq!(x_component, 2.0);
    /// ```
    #[inline]
    fn index(&self, index: u32) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::IndexMut<u32> for FVector2 {
    /// Mutable indexing for the `FVector2` by a `u32` index.
    ///
    /// This implementation allows you to mutably access and modify the components of an `FVector2`
    /// using a `u32` index where `0` corresponds to `x` and `1` corresponds to `y`. It returns
    /// a mutable reference to the component at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index to access and modify, where `0` represents `x` and `1` represents `y`.
    ///
    /// # Returns
    ///
    /// A mutable reference to the specified component of the `FVector2`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds (not 0 or 1).
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector2;
    ///
    /// let mut vector = FVector2::new(2.0, 3.0);
    /// vector[0u32] = 4.0;
    ///
    /// assert_eq!(vector[0u32], 4.0);
    /// ```
    #[inline]
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index out of bounds"),
        }
    }
}
