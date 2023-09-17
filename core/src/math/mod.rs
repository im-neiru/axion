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

pub use angles::Angle;
pub use angles::SphericalAngles;
pub use angles::{degrees, Degrees};
pub use angles::{radians, Radians};
pub use angles::{turns, Turns};

pub use common::{vec2, FVector2};
pub use common::{vec3, FVector3};
pub use common::{vec4, FVector4};

#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use sse::*;

// Public structs
#[cfg(not(feature = "simd"))]
pub use ordinary::*;
