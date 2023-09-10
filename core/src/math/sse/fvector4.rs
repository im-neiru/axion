// Imports from core
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// Imports from std
use std::{fmt, mem::MaybeUninit, ops};

/// `FVector4` is a structure that represents a 3D vector with `f32` components.
/// It encapsulates three floating-point values and is used for various purposes in graphical applications
/// including points, vectors, and texture coordinates.
///
/// This structure is implemented with SIMD (Single Instruction, Multiple Data) which is a type of parallel computing
/// involving vectors that allows for multiple data points to be processed at once, resulting in performance improvements.
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

/// A union used for type punning between `FVector4` and `__m128`.
///
/// This union allows for seamless conversion and manipulation between `FVector4` instances
/// and the `__m128` SIMD data type, enabling efficient vector operations.
///
/// # Safety
///
/// Care must be taken when using this union, as it involves type punning and could result in
/// undefined behavior if not used correctly.
///
/// # Example
///
/// ```
/// use axion::math::FVector4;
/// use axion::simd::UnionCast;
///
/// let vector = FVector4::new(1.0, 2.0, 3.0, 4.0);
///
/// let union_cast = UnionCast { v4: vector };
/// let simd_data: __m128 = unsafe { union_cast.m128 };
///
/// // Now you can perform SIMD operations on `simd_data`
/// ```
#[repr(C)]
union UnionCast {
    /// Represents the `FVector4` view of the union.
    v4: FVector4,
    /// Represents the `__m128` SIMD data view of the union.
    m128: __m128,
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
    pub const fn axis_xy(self) -> super::FVector2 {
        super::FVector2 {
            x: self.x,
            y: self.y,
        }
    }

    /// Returns the x and z components of the vector as an FVector2.
    #[inline]
    pub const fn axis_xz(self) -> super::FVector2 {
        super::FVector2 {
            x: self.x,
            y: self.z,
        }
    }

    /// Returns the y and x components of the vector as an FVector2.
    #[inline]
    pub const fn axis_yx(self) -> super::FVector2 {
        super::FVector2 {
            x: self.y,
            y: self.x,
        }
    }

    /// Returns the y and z components of the vector as an FVector2.
    #[inline]
    pub const fn axis_yz(self) -> super::FVector2 {
        super::FVector2 {
            x: self.y,
            y: self.z,
        }
    }

    /// Returns the z and x components of the vector as an FVector2.
    #[inline]
    pub const fn axis_zx(self) -> super::FVector2 {
        super::FVector2 {
            x: self.z,
            y: self.x,
        }
    }

    /// Returns the z and y components of the vector as an FVector2.
    #[inline]
    pub const fn axis_zy(self) -> super::FVector2 {
        super::FVector2 {
            x: self.z,
            y: self.y,
        }
    }

    /// Returns the x component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_xx(self) -> super::FVector2 {
        super::FVector2 {
            x: self.x,
            y: self.x,
        }
    }

    /// Returns the y component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_yy(self) -> super::FVector2 {
        super::FVector2 {
            x: self.y,
            y: self.y,
        }
    }

    /// Returns the z component of the vector twice as an FVector2.
    #[inline]
    pub const fn axis_zz(self) -> super::FVector2 {
        super::FVector2 {
            x: self.z,
            y: self.z,
        }
    }

    /// Returns the z, y, and x components of the vector as an FVector3.
    #[inline]
    pub const fn axis_zyx(self) -> super::FVector3 {
        super::FVector3 {
            x: self.z,
            y: self.y,
            z: self.x,
        }
    }

    /// Returns the x, z, and y components of the vector as an FVector3.
    #[inline]
    pub const fn axis_xzy(self) -> super::FVector3 {
        super::FVector3 {
            x: self.x,
            y: self.z,
            z: self.y,
        }
    }

    /// Returns the y, z, and x components of the vector as an FVector3.
    #[inline]
    pub const fn axis_yzx(self) -> super::FVector3 {
        super::FVector3 {
            x: self.y,
            y: self.z,
            z: self.x,
        }
    }

    /// Returns the y, x, and z components of the vector as an FVector3.
    #[inline]
    pub const fn axis_yxz(self) -> super::FVector3 {
        super::FVector3 {
            x: self.y,
            y: self.x,
            z: self.z,
        }
    }

    /// Returns the z, x, and y components of the vector as an FVector3.
    #[inline]
    pub const fn axis_zxy(self) -> super::FVector3 {
        super::FVector3 {
            x: self.z,
            y: self.x,
            z: self.y,
        }
    }

    /// Returns the x component of the vector three times as an FVector3.
    #[inline]
    pub const fn axis_xxx(self) -> super::FVector3 {
        super::FVector3 {
            x: self.x,
            y: self.x,
            z: self.x,
        }
    }

    /// Returns the y component of the vector three times as an FVector4.
    #[inline]
    pub const fn axis_yyy(self) -> super::FVector3 {
        super::FVector3 {
            x: self.y,
            y: self.y,
            z: self.y,
        }
    }

    /// Returns the z component of the vector three times as an FVector4.
    #[inline]
    pub const fn axis_zzz(self) -> super::FVector3 {
        super::FVector3 {
            x: self.z,
            y: self.z,
            z: self.z,
        }
    }

    /// Returns the dot product of the vector and another vector.
    ///
    /// # Arguments
    ///
    /// * `other` - Another vector to calculate the dot product with.
    ///
    /// # Returns
    ///
    /// The dot product of the two vectors as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let vector2 = FVector4::new(4.0, 3.0, 2.0, 1.0);
    /// let dot_product = vector1.dot(vector2);
    ///
    /// assert_eq!(dot_product, 20.0);
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn dot(self, other: Self) -> f32 {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: other }.m128;

            let mut dot_product: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, b);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);

            _mm_store_ss(&mut dot_product, sum);

            dot_product
        }
    }

    /// Returns the length (magnitude) of the vector.
    ///
    /// The length of a vector is a non-negative number that describes the extent of the vector in space.
    ///
    /// # Returns
    ///
    /// The length of the vector as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(3.0, 4.0, 5.0, 0.0);
    /// let length = vector.length();
    ///
    /// assert_eq!(length, 7.071068);
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn length(self) -> f32 {
        unsafe {
            let a = UnionCast { v4: self }.m128;

            let mut length: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
            let sum = _mm_hadd_ps(sum, sum);
            let root = _mm_sqrt_ps(sum);

            _mm_store_ss(&mut length, root);

            length
        }
    }

    /// Returns the squared length of the vector.
    ///
    /// This function is often used when comparing lengths, as it avoids the computationally
    /// expensive square root operation.
    ///
    /// # Returns
    ///
    /// The squared length of the vector as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(3.0, 4.0, 5.0, 0.0);
    /// let length_sq = vector.length_sq();
    ///
    /// assert_eq!(length_sq, 50.0);
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn length_sq(self) -> f32 {
        unsafe {
            let a = UnionCast { v4: self }.m128;

            let mut length_sq: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
            let sum = _mm_hadd_ps(sum, sum);

            _mm_store_ss(&mut length_sq, sum);

            length_sq
        }
    }

    /// Returns the inverse of the length (reciprocal of the magnitude) of the vector.
    ///
    /// If the length of the vector is `FVector4::ZERO`, this function will return `f32::INFINITY`.
    ///
    /// # Returns
    ///
    /// The inverse of the length of the vector as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(3.0, 4.0, 5.0, 0.0);
    /// let inv_length = vector.length_inv();
    ///
    /// assert_eq!(inv_length, 0.14142136);
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn length_inv(self) -> f32 {
        unsafe {
            let a = UnionCast { v4: self }.m128;

            let mut length_inv: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
            let sum = _mm_hadd_ps(sum, sum);
            let root = _mm_sqrt_ps(sum);
            let recip = _mm_rcp_ps(root);

            _mm_store_ss(&mut length_inv, recip);

            length_inv
        }
    }

    /// Calculates the Euclidean distance between two `FVector4` instances.
    ///
    /// This function computes the Euclidean distance between two `FVector4` instances.
    ///
    /// # Arguments
    ///
    /// * `other` - The second vector to which the distance is calculated.
    ///
    /// # Returns
    ///
    /// The calculated Euclidean distance between the two input vectors as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(1.0, 2.0, 3.0, 0.0);
    /// let vector2 = FVector4::new(4.0, 6.0, 8.0, 0.0);
    /// let distance = vector1.distance(vector2);
    ///
    /// assert_eq!(distance, 7.071068);
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn distance(self, other: Self) -> f32 {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: other }.m128;

            let mut distance: f32 = MaybeUninit::uninit().assume_init();
            let sub = _mm_sub_ps(a, b);
            let product = _mm_mul_ps(sub, sub);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
            let sum = _mm_hadd_ps(sum, sum);
            let root = _mm_sqrt_ps(sum);

            _mm_store_ss(&mut distance, root);

            distance
        }
    }

    /// Computes the squared Euclidean distance between two `FVector4` instances.
    ///
    /// This method calculates the squared Euclidean distance between `self` and `other`, which is a
    /// more efficient version of the Euclidean distance as it avoids the square root operation.
    ///
    /// # Arguments
    ///
    /// * `other` - The `FVector4` instance representing the other point in 4D space.
    ///
    /// # Returns
    ///
    /// The squared Euclidean distance between `self` and `other` as a `f32` value.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let point1 = FVector4::new(2.0, 3.0, 1.0);
    /// let point2 = FVector4::new(1.0, 2.0, 3.0);
    /// let distance_sq = point1.distance_sq(point2);
    ///
    /// // Check if the squared distance is approximately 10.0 within a small tolerance
    /// assert!((distance_sq - 10.0).abs() < 1e-6);
    /// ```

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn distance_sq(self, other: Self) -> f32 {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: other }.m128;

            let mut distance_sq: f32 = MaybeUninit::uninit().assume_init();
            let sub = _mm_sub_ps(a, b);
            let product = _mm_mul_ps(sub, sub);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
            let sum = _mm_hadd_ps(sum, sum);

            _mm_store_ss(&mut distance_sq, sum);

            distance_sq
        }
    }

    /// The `normalize` function normalizes a vector using SIMD instructions for efficiency.
    /// Be cautious when using this function with vectors at the origin `(0.0, 0.0, 0.0)`, as this would lead
    /// to a division by zero when calculating the reciprocal of the root. The result in such cases
    /// would be ``FVector4::NAN``. It is recommended to check if the vector is at the origin before
    /// calling this function.
    ///
    /// # Arguments
    ///
    /// * `self` - A FVector4 to be normalized.
    ///
    /// # Returns
    ///
    /// * `FVector4` - A new FVector4 that is the normalized version of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let v = FVector4::new(3.0, 4.0, 5.0); // Create a 3D vector
    /// let normalized_v = v.normalize(); // Normalize the 3D vector
    /// ```
    /// The `normalize` function normalizes a vector using SIMD instructions for efficiency.
    /// Be cautious when using this function with vectors at the origin `(0.0, 0.0, 0.0)`, as this would lead
    /// to a division by zero when calculating the reciprocal of the root. The result in such cases
    /// would be ``FVector4::NAN``. It is recommended to check if the vector is at the origin before
    /// calling this function.
    ///
    /// # Arguments
    ///
    /// * `self` - A FVector4 to be normalized.
    ///
    /// # Returns
    ///
    /// * `FVector4` - A new FVector4 that is the normalized version of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let v = FVector4::new(3.0, 4.0, 5.0); // Create a 3D vector
    /// let normalized_v = v.normalize(); // Normalize the 3D vector
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn normalize(self) -> Self {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let mut normalized_vec: (Self, Self) =
                MaybeUninit::uninit().assume_init();

            let product = _mm_mul_ps(a, a);
            let sum = _mm_hadd_ps(product, product);
            let sum = _mm_hadd_ps(sum, sum);
            let sum = _mm_hadd_ps(sum, sum);
            let root = _mm_sqrt_ps(sum);
            let recip = _mm_set1_ps(_mm_cvtss_f32(_mm_rcp_ps(root)));
            let normalized = _mm_mul_ps(a, recip);

            _mm_store_ps(
                (&mut normalized_vec as *mut (Self, Self)) as *mut f32,
                normalized,
            );
            normalized_vec.0
        }
    }

    /// Performs linear interpolation between two `FVector4` instances.
    ///
    /// Linear interpolation, or lerp, blends between two `FVector4` instances using a specified
    /// interpolation factor `scalar`. The result is a new `FVector4` where each component is interpolated
    /// between the corresponding components of `self` and `end` based on `scalar`.
    ///
    /// # Arguments
    ///
    /// * `end` - The target `FVector4` to interpolate towards.
    /// * `scalar` - The interpolation factor, typically in the range `0.0`, `1.0`, where:
    ///   - `scalar = 0.0` returns `self`.
    ///   - `scalar = 1.0` returns `end`.
    ///   - Values in between interpolate linearly.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the linear interpolation of `self` and `end` with factor `scalar`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let start = FVector4::new(1.0, 2.0, 3.0, 0.0);
    /// let end = FVector4::new(3.0, 4.0, 5.0, 0.0);
    /// let interpolated = start.lerp(end, 0.5);
    ///
    /// assert_eq!(interpolated, FVector4::new(2.0, 3.0, 4.0, 0.0));
    /// ```
    #[inline]
    pub fn lerp(self, end: Self, scalar: f32) -> Self {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: end }.m128;

            UnionCast {
                m128: _mm_add_ps(
                    a,
                    _mm_mul_ps(_mm_sub_ps(b, a), _mm_set1_ps(scalar)),
                ),
            }
            .v4
        }
    }

    /// Clamps each component of the vector to a specified range.
    ///
    /// This method clamps each component of the vector to the specified `min` and `max` values. If a component is less
    /// than `min`, it is set to `min`. If it is greater than `max`, it is set to `max`. Otherwise, it remains unchanged.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value for each component.
    /// * `max` - The maximum value for each component.
    ///
    /// # Returns
    ///
    /// A new `FVector4` with components clamped to the `min` and `max` range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(2.0, 3.0, 1.0, 5.0);
    /// let clamped = vector.clamp(FVector4::new(1.0, 2.0, 1.0, 3.0), FVector4::new(3.0, 4.0, 2.0, 4.0));
    ///
    /// assert_eq!(clamped, FVector4::new(2.0, 3.0, 2.0, 4.0));
    /// ```
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }

    /// Calculates the element-wise minimum between the vector and another vector.
    ///
    /// This method computes the element-wise minimum between each component of the vector and the corresponding component
    /// of the `value` vector. The result is a new `FVector4` where each component is the minimum of the original vector
    /// and the `value` vector.
    ///
    /// # Arguments
    ///
    /// * `value` - Another `FVector4` to calculate the element-wise minimum with.
    ///
    /// # Returns
    ///
    /// A new `FVector4` with components set to the minimum of the original vector and the `value` vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(2.0, 3.0, 1.0, 5.0);
    /// let min_vector = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let result = vector.min(min_vector);
    ///
    /// assert_eq!(result, FVector4::new(1.0, 2.0, 1.0, 4.0));
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn min(self, value: Self) -> Self {
        unsafe {
            let a = UnionCast { v4: self }.m128;

            let b = UnionCast { v4: value }.m128;

            UnionCast {
                m128: _mm_min_ps(a, b),
            }
            .v4
        }
    }

    /// Calculates the element-wise maximum between the vector and another vector.
    ///
    /// This method computes the element-wise maximum between each component of the vector and the corresponding component
    /// of the `value` vector. The result is a new `FVector4` where each component is the maximum of the original vector
    /// and the `value` vector.
    ///
    /// # Arguments
    ///
    /// * `value` - Another `FVector4` to calculate the element-wise maximum with.
    ///
    /// # Returns
    ///
    /// A new `FVector4` with components set to the maximum of the original vector and the `value` vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(2.0, 3.0, 1.0, 5.0);
    /// let max_vector = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let result = vector.max(max_vector);
    ///
    /// assert_eq!(result, FVector4::new(2.0, 3.0, 3.0, 5.0));
    /// ```
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    pub fn max(self, value: Self) -> Self {
        unsafe {
            let a = UnionCast { v4: self }.m128;

            let b = UnionCast { v4: value }.m128;

            UnionCast {
                m128: _mm_max_ps(a, b),
            }
            .v4
        }
    }

    /// Checks if all components of the vector are equal to zero.
    ///
    /// This method returns `true` if all components of the vector are exactly zero, and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if all components of the vector are zero; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let zero_vector = FVector4::new(0.0, 0.0, 0.0, 0.0);
    /// let non_zero_vector = FVector4::new(1.0, 0.0, 0.0, 0.0);
    ///
    /// assert_eq!(zero_vector.is_zero(), true);
    /// assert_eq!(non_zero_vector.is_zero(), false);
    /// ```
    #[inline]
    pub fn is_zero(self) -> bool {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: Self::ZERO }.m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) == 0xf
        }
    }

    /// Checks if any component of the vector is NaN (Not-a-Number).
    ///
    /// This method returns `true` if any component of the vector is NaN (Not-a-Number), and `false` otherwise.
    ///
    /// # Returns
    ///
    /// `true` if any component of the vector is NaN; otherwise, `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let nan_vector = FVector4::new(1.0, f32::NAN, 3.0, 4.0);
    /// let non_nan_vector = FVector4::new(2.0, 3.0, 4.0, 5.0);
    ///
    /// assert_eq!(nan_vector.is_nan(), true);
    /// assert_eq!(non_nan_vector.is_nan(), false);
    /// ```
    #[inline]
    pub fn is_nan(self) -> bool {
        unsafe {
            let a = UnionCast { v4: self }.m128;

            let cmp_result = _mm_cmpunord_ps(a, a);

            _mm_movemask_ps(cmp_result) != 0
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

impl PartialEq for FVector4 {
    /// Checks if two `FVector4` instances are equal.
    ///
    /// This implementation compares each component of two `FVector4` instances for equality
    /// and returns `true` if all components are equal, and `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to compare with.
    ///
    /// # Returns
    ///
    /// `true` if all components of the two `FVector4` instances are equal, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector4::new(2.0, 3.0, 4.0); // Create another 3D vector with the same values
    /// let vector3 = FVector4::new(1.0, 2.0, 3.0); // Create a different 3D vector
    ///
    /// assert_eq!(vector1, vector2); // Check if vector1 is equal to vector2
    /// assert_ne!(vector1, vector3); // Check if vector1 is not equal to vector3
    /// ```
    #[inline]
    fn eq(&self, rhs: &Self) -> bool {
        unsafe {
            let a = UnionCast { v4: *self }.m128;
            let b = UnionCast { v4: *rhs }.m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) == 0xf
        }
    }

    /// Checks if two `FVector4` instances are not equal.
    ///
    /// This implementation compares each component of two `FVector4` instances for inequality
    /// and returns `true` if at least one component is not equal, and `false` if all components
    /// are equal.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to compare with.
    ///
    /// # Returns
    ///
    /// `true` if at least one component of the two `FVector4` instances is not equal, `false` if all components are equal.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector4::new(1.0, 2.0, 4.0); // Create another 3D vector with differences in components
    ///
    /// assert_ne!(vector1, vector2); // Check if vector1 is not equal to vector2 due to differing components
    /// ```
    #[inline]
    #[allow(clippy::partialeq_ne_impl)]
    fn ne(&self, rhs: &Self) -> bool {
        unsafe {
            let a = UnionCast { v4: *self }.m128;
            let b = UnionCast { v4: *rhs }.m128;

            let cmp_result = _mm_cmpeq_ps(a, b);

            _mm_movemask_ps(cmp_result) != 0xf
        }
    }
}

impl fmt::Display for FVector4 {
    /// Checks if two `FVector4` instances (4D vectors) are equal.
    ///
    /// This implementation compares each component of two `FVector4` instances for equality
    /// and returns `true` if all components are equal, and `false` otherwise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` (4D vector) to compare with.
    ///
    /// # Returns
    ///
    /// `true` if all components of the two `FVector4` instances (4D vectors) are equal, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0, 5.0); // Create a 4D vector
    /// let vector2 = FVector4::new(2.0, 3.0, 4.0, 5.0); // Create another 4D vector with the same values
    /// let vector3 = FVector4::new(1.0, 2.0, 3.0, 4.0); // Create a different 4D vector
    ///
    /// assert_eq!(vector1, vector2); // Check if vector1 is equal to vector2
    /// assert_ne!(vector1, vector3); // Check if vector1 is not equal to vector3
    /// ```
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "({}, {}, {}, {})",
            self.x, self.y, self.z, self.w
        )
    }
}

impl ops::Index<usize> for FVector4 {
    type Output = f32;

    /// Checks if two `FVector4` instances (4D vectors) are not equal.
    ///
    /// This implementation compares each component of two `FVector4` instances for inequality
    /// and returns `true` if at least one component is not equal, and `false` if all components
    /// are equal.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` (4D vector) to compare with.
    ///
    /// # Returns
    ///
    /// `true` if at least one component of the two `FVector4` instances (4D vectors) is not equal, `false` if all components are equal.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0, 5.0); // Create a 4D vector
    /// let vector2 = FVector4::new(1.0, 2.0, 4.0, 5.0); // Create another 4D vector with differences in components
    ///
    /// assert_ne!(vector1, vector2); // Check if vector1 is not equal to vector2 due to differing components
    /// ```
    #[inline]
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::IndexMut<usize> for FVector4 {
    /// Gets the mutable reference to the component at the specified `usize` index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the component to retrieve. Valid values are 0, 1, 2, or 3.
    ///
    /// # Returns
    ///
    /// A mutable reference to the value of the component at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds (not in the range `0..=3`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let mut vector = FVector4::new(10.0, 15.0, 8.0, 7.0);
    ///
    /// vector[0] = 20.0;
    /// vector[1] = 25.0;
    ///
    /// assert_eq!(vector.x, 20.0);
    /// assert_eq!(vector.y, 25.0);
    /// ```
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::Index<u32> for FVector4 {
    type Output = f32;

    /// Gets the component at the specified `u32` index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the component to retrieve. Valid values are 0, 1, 2, or 3.
    ///
    /// # Returns
    ///
    /// The value of the component at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds (not in the range `0..=3`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector = FVector4::new(10.0, 15.0, 8.0, 7.0);
    ///
    /// let x_component = vector[0u32];
    /// let y_component = vector[1u32];
    ///
    /// assert_eq!(x_component, 10.0);
    /// assert_eq!(y_component, 15.0);
    /// ```
    #[inline]
    fn index(&self, index: u32) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::IndexMut<u32> for FVector4 {
    /// Gets the mutable reference to the component at the specified `u32` index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the component to retrieve. Valid values are 0, 1, 2, or 3.
    ///
    /// # Returns
    ///
    /// A mutable reference to the value of the component at the specified index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds (not in the range `0..=3`).
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let mut vector = FVector4::new(10.0, 15.0, 8.0, 7.0);
    ///
    /// vector[0u32] = 20.0;
    /// vector[1u32] = 25.0;
    ///
    /// assert_eq!(vector.x, 20.0);
    /// assert_eq!(vector.y, 25.0);
    /// ```
    #[inline]
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("index out of bounds"),
        }
    }
}

impl ops::Add<Self> for FVector4 {
    type Output = Self;

    /// Adds two `FVector4` instances component-wise.
    ///
    /// This method performs component-wise addition of two `FVector4` instances, resulting in a new `FVector4` where each component
    /// is the sum of the corresponding components from `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The second `FVector4` to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the addition of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let vector2 = FVector4::new(2.0, 3.0, 4.0, 5.0);
    /// let result = vector1 + vector2;
    ///
    /// assert_eq!(result, FVector4::new(3.0, 5.0, 7.0, 9.0));
    /// ```
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: rhs }.m128;

            UnionCast {
                m128: _mm_add_ps(a, b),
            }
            .v4
        }
    }
}

impl ops::Sub<Self> for FVector4 {
    type Output = Self;

    /// Subtracts one `FVector4` from another component-wise.
    ///
    /// This implementation allows you to subtract one `FVector4` from another
    /// component-wise, resulting in a new `FVector4` where each component is the
    /// difference between the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to subtract from `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the component-wise subtraction of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// let vector2 = FVector4::new(1.0, 2.0, 1.0); // Create another 3D vector
    /// let result = vector1 - vector2; // Perform component-wise subtraction
    ///
    /// assert_eq!(result, FVector4::new(1.0, 1.0, 3.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: rhs }.m128;

            UnionCast {
                m128: _mm_sub_ps(a, b),
            }
            .v4
        }
    }
}

impl ops::Mul<Self> for FVector4 {
    type Output = Self;

    /// Multiplies two `FVector4` instances component-wise.
    ///
    /// This method performs component-wise multiplication of two `FVector4` instances, resulting in a new `FVector4` where each component
    /// is the product of the corresponding components from `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The second `FVector4` to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the multiplication of `self` and `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(2.0, 3.0, 4.0, 5.0);
    /// let vector2 = FVector4::new(3.0, 2.0, 1.0, 2.0);
    /// let result = vector1 * vector2;
    ///
    /// assert_eq!(result, FVector4::new(6.0, 6.0, 4.0, 10.0));
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: rhs }.m128;

            UnionCast {
                m128: _mm_mul_ps(a, b),
            }
            .v4
        }
    }
}

impl ops::Div<Self> for FVector4 {
    type Output = Self;

    /// Divides two `FVector4` instances component-wise.
    ///
    /// This method performs component-wise division of `self` by `rhs`, resulting in a new `FVector4` where each component
    /// is the quotient of the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` to divide `self` by.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the division of `self` by `rhs`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(6.0, 6.0, 4.0, 10.0);
    /// let vector2 = FVector4::new(3.0, 2.0, 1.0, 2.0);
    /// let result = vector1 / vector2;
    ///
    /// assert_eq!(result, FVector4::new(2.0, 3.0, 4.0, 5.0));
    /// ```
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: rhs }.m128;

            UnionCast {
                m128: _mm_div_ps(a, b),
            }
            .v4
        }
    }
}

impl ops::Rem<Self> for FVector4 {
    type Output = Self;

    /// Computes the remainder of dividing one `FVector4` by another component-wise.
    ///
    /// This implementation allows you to compute the remainder of dividing one `FVector4`
    /// by another component-wise, resulting in a new `FVector4` where each component is
    /// the remainder of dividing the corresponding components of `self` by `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to compute the remainder of division with `self`.
    ///
    /// # Returns
    ///
    /// A new `FVector4` instance resulting from the component-wise remainder of division of `self` by `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(7.0, 10.0, 13.0); // Create a 3D vector
    /// let vector2 = FVector4::new(3.0, 4.0, 5.0);   // Create another 3D vector
    /// let result = vector1 % vector2;                // Perform component-wise remainder
    ///
    /// assert_eq!(result, FVector4::new(1.0, 2.0, 3.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        unsafe {
            let a = UnionCast { v4: self }.m128;
            let b = UnionCast { v4: rhs }.m128;

            UnionCast {
                m128: _mm_sub_ps(
                    a,
                    _mm_mul_ps(b, _mm_floor_ps(_mm_div_ps(a, b))),
                ),
            }
            .v4
        }
    }
}

impl ops::AddAssign<Self> for FVector4 {
    /// Adds another `FVector4` to `self` in-place.
    ///
    /// This method performs component-wise addition of another `FVector4` to `self`, updating `self` with the result.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` to add to `self` in-place.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let mut vector1 = FVector4::new(1.0, 2.0, 3.0, 4.0);
    /// let vector2 = FVector4::new(2.0, 4.0, 5.0, 6.0);
    ///
    /// vector1 += vector2;
    ///
    /// assert_eq!(vector1, FVector4::new(3.0, 6.0, 8.0, 10.0));
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl ops::SubAssign<Self> for FVector4 {
    /// Subtracts another `FVector4` from `self` in-place.
    ///
    /// This method performs component-wise subtraction of another `FVector4` from `self`, updating `self` with the result.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` to subtract from `self` in-place.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let mut vector1 = FVector4::new(3.0, 6.0, 8.0, 10.0);
    /// let vector2 = FVector4::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// vector1 -= vector2;
    ///
    /// assert_eq!(vector1, FVector4::new(2.0, 4.0, 5.0, 6.0));
    /// ```
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl ops::MulAssign<Self> for FVector4 {
    /// Multiplies `self` by another `FVector4` in place.
    ///
    /// This implementation allows you to multiply `self` by another `FVector4` in place,
    /// modifying `self` to be the result of the multiplication.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side `FVector4` to multiply `self` with.
    ///
    /// # Example
    ///
    /// ```
    /// use axion::math::FVector4;
    ///
    /// let mut vector = FVector4::new(2.0, 3.0, 4.0); // Create a 3D vector
    /// vector *= FVector4::new(1.0, 2.0, 2.0);        // Perform in-place multiplication
    ///
    /// assert_eq!(vector, FVector4::new(2.0, 6.0, 8.0)); // Check the result with 3D vectors
    /// ```
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl ops::DivAssign<Self> for FVector4 {
    /// Calculates the component-wise remainder of `self` by another `FVector4`.
    ///
    /// This method performs component-wise remainder of `self` by another `FVector4`, producing a new `FVector4` with
    /// each component being the remainder of the corresponding components of `self` and `rhs`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` by which to perform component-wise remainder.
    ///
    /// # Returns
    ///
    /// A new `FVector4` resulting from the component-wise remainder operation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let vector1 = FVector4::new(10.0, 15.0, 8.0, 7.0);
    /// let vector2 = FVector4::new(3.0, 4.0, 3.0, 2.0);
    ///
    /// let result = vector1 % vector2;
    ///
    /// assert_eq!(result, FVector4::new(1.0, 3.0, 2.0, 1.0));
    /// ```
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl ops::RemAssign<Self> for FVector4 {
    /// Calculates the component-wise remainder of `self` by another `FVector4` in-place.
    ///
    /// This method performs component-wise remainder of `self` by another `FVector4`, updating `self` with the result.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The `FVector4` by which to perform component-wise remainder.
    ///
    /// # Example
    ///
    /// ```rust
    /// use axion::math::FVector4;
    ///
    /// let mut vector1 = FVector4::new(10.0, 15.0, 8.0, 7.0);
    /// let vector2 = FVector4::new(3.0, 4.0, 3.0, 2.0);
    ///
    /// vector1 %= vector2;
    ///
    /// assert_eq!(vector1, FVector4::new(1.0, 3.0, 2.0, 1.0));
    /// ```
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

#[test]
fn test_fvector4_sse3() {
    use std::time::SystemTime;

    let a = FVector4::new(3.0, 4.0, 2.0, 1.0);
    let b = FVector4::new(4.0, 3.2, 5.0, 2.0);
    let w = a.dot(b); // warm up
    println!("Warm up dot = {w}");

    let ref_time = SystemTime::now();
    let dot = a.dot(b);

    let span_dot = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let length = a.length();

    let span_length = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let normalize = a.normalize();

    let span_normalize = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let distance = a.distance(b);

    let span_distance = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let is_zero = FVector4::ZERO.is_zero();

    let span_is_zero = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let is_nan = FVector4::NAN.is_nan();

    let span_is_nan = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let is_finite = FVector4::INFINITY.is_finite();

    let span_is_finite = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    // assert!(dot == 34.8, "FVector4::dot | Wrong output");
    // assert!(length == 5.0, "FVector4::length | Wrong output");

    println!("FVector4::dot | {span_dot} ns | {a} ⋅ {b} = {dot}",);

    println!("FVector4::length | {span_length} ns | {a} = {length}",);

    println!("FVector4::normalize | {span_normalize} ns | {a} = {normalize}",);

    println!("FVector4::distance | {span_distance} ns | {a} {b} = {distance}",);

    println!(
        "FVector4::is_nan | {span_is_nan} ns | {}  = {is_nan}",
        FVector4::NAN,
    );

    println!(
        "FVector4::is_nan | {span_is_zero} ns | {}  = {is_zero}",
        FVector4::ZERO,
    );

    println!(
        "FVector4::is_finite | {span_is_finite} ns | {}  = {is_finite}",
        FVector4::INFINITY,
    );

    {
        let start = FVector4::new(1.0, 2.0, 1.0, 2.0);
        let end = FVector4::new(3.0, 4.0, 1.0, 2.0);

        let ref_time = SystemTime::now();

        let lerp = start.lerp(end, 0.5);

        let span_lerp = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector4::lerp | {span_lerp} ns | {start} {end} = {lerp}");
    }

    {
        let a = FVector4::new(1.0, 2.0, 1.0, 2.0);
        let b = FVector4::new(1.0, 2.0, 1.0, 2.0);

        let ref_time = SystemTime::now();

        let eq = a == b;

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector4::eq | {span} ns | {a} == {b} = {eq}");
    }

    {
        let a = FVector4::new(1.0, 2.0, 1.0, 3.0);
        let b = FVector4::new(1.0, 2.0, 1.0, 2.0);

        let ref_time = SystemTime::now();

        let ne = a != b;

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector4::ne | {span} ns | {a} != {b} = {ne}");
    }

    {
        let ref_time = SystemTime::now();
        let distance_sq = a.distance_sq(b);

        let span = SystemTime::now()
            .duration_since(ref_time)
            .unwrap()
            .as_nanos();

        println!("FVector4::distance_sq | {span} ns | {a} {b} = {distance_sq}");
    }
}
