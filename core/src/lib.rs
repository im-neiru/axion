// Public modules

#[cfg(feature = "math")]
/// A module containing various mathematical types and utilities.
pub mod math;

// Exported macros
pub use axion_macro;

// Private module
mod error;

// Public struct & enums
pub use error::{Error, Result};
