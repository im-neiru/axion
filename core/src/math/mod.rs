// Within this module, is the implementations for vectors, matrices, and more...

// Sub modules
mod sse2;
mod vector;

// Private trait
pub(crate) use vector::Vector;

// Public trait
pub use vector::Vector2;
