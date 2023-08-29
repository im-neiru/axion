use std::fmt::Debug;

pub struct Error(ErrorKind);

pub type Result<T> = std::result::Result<T, self::Error>;

pub(crate) enum ErrorKind {
    #[allow(unused)]
    Unknown,
    MathError(MathError),
}

pub(crate) enum MathError {
    LengthHasNoInverse,
}

impl Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            ErrorKind::Unknown => "Unknown Error",
            ErrorKind::MathError(MathError::LengthHasNoInverse) => "Unable to calculate inverse of length. The vector is at the origin.",
        })
    }
}

impl MathError {
    #[inline]
    pub(crate) fn into_result<T>(self) -> self::Result<T> {
        Err(Error(ErrorKind::MathError(self)))
    }
}
