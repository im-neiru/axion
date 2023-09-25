// Within this module, is the SSE2 implementations for vectors, matrices, and more...

// Sub modules
mod vector2;
mod vector3;
mod vector4;

pub use vector2::*;
pub use vector3::*;
pub use vector4::*;

/// Create an SSE shuffle mask from component indices.
///
/// This function constructs an SSE shuffle mask based on the provided component indices `z`, `y`, `x`, and `w`.
///
/// # Arguments
///
/// * `z` - The index of the component from `z` to be placed in the most significant position of the result.
/// * `y` - The index of the component from `y` to be placed in the second most significant position of the result.
/// * `x` - The index of the component from `x` to be placed in the third most significant position of the result.
/// * `w` - The index of the component from `w` to be placed in the least significant position of the result.
/// ```
#[inline(always)]
const fn mm_shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
