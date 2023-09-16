// Sub modules
mod degrees;
mod radians;
mod turns;

mod spherical_angles;

//Public structs
pub use degrees::{degrees, Degrees};
pub use radians::{radians, Radians};
pub use turns::{turns, Turns};

pub use spherical_angles::SphericalAngles;

// Traits

/// Trait for `Radians`, `Degrees` and `Turns`
pub trait Angle:
    PrivateAngle + Default + Copy + Clone + std::fmt::Display + std::fmt::Debug
{
}

///Ensures that `Angle` is only implemented here
pub trait PrivateAngle {}
