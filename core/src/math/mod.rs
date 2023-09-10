// Within this module, is the implementations for vectors, matrices, and more...

// Sub modules
#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod sse;

// Public structs
#[cfg(feature = "simd")]
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use sse::{FVector2, FVector3, FVector4};
