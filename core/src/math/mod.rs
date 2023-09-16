// Within this module, is the implementations for vectors, matrices, and more...

// Sub modules
mod angles;
mod common;
#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod sse;

#[cfg(not(feature = "simd"))]
mod ordinary;

// Public structs

pub use angles::{radians, Radians};

/// `FVector2` is a structure that represents a 2D vector with `f32` components.
/// It encapsulates two floating-point values and is used for various purposes in graphical applications
/// including points, vectors, and texture coordinates.
#[derive(Clone, Copy, Debug)]
pub struct FVector2 {
    /// The X component of the vector.
    pub x: f32,
    /// The Y component of the vector.
    pub y: f32,
}

/// `FVector3` is a structure that represents a 3D vector with `f32` components.
/// It encapsulates three floating-point values and is used for various purposes in graphical applications
/// including points, vectors, and texture coordinates.
#[derive(Clone, Copy, Debug)]
pub struct FVector3 {
    /// The X component of the vector.
    pub x: f32,
    /// The Y component of the vector.
    pub y: f32,
    /// The Z component of the vector.
    pub z: f32,
}

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

#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use sse::*;

// Public structs
#[cfg(not(feature = "simd"))]
pub use ordinary::*;

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

/// Convenience function for creating a 3D vector (FVector3).
///
/// This function is a convenient way to create a 3D vector (FVector3)
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
/// A new `FVector3` with the specified components.
///
/// # Example
///
/// ```
/// use axion::math::{FVector3, vec3};
///
/// let vector = vec3(1.0, 2.0, 3.0); // Create a 3D vector
/// ```
#[inline(always)]
pub const fn vec3(x: f32, y: f32, z: f32) -> FVector3 {
    FVector3 { x, y, z }
}

/// Convenience function for creating a 2D vector (FVector2).
///
/// This function is a convenient way to create a 2D vector (FVector2)
/// with the given components.
///
/// # Arguments
///
/// * `x` - The x-component of the vector.
/// * `y` - The y-component of the vector.
///
/// # Returns
///
/// A new `FVector2` with the specified components.
///
/// # Example
///
/// ```
/// use axion::math::{FVector2, vec2};
///
/// let vector = vec2(1.0, 2.0); // Create a 2D vector
/// ```
#[inline(always)]
pub const fn vec2(x: f32, y: f32) -> FVector2 {
    FVector2 { x, y }
}
