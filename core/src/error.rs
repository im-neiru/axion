use std::fmt::Debug;

pub struct Error(ErrorKind);

pub type Result<T> = std::result::Result<T, self::Error>;

pub(crate) enum ErrorKind {
    #[allow(unused)]
    Unknown,
    MathError(MathError),
}

pub(crate) enum MathError {}

impl Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            ErrorKind::Unknown => "Unknown Error",
            ErrorKind::MathError(_) => "Math Error",
        })
    }
}

impl MathError {
    #[inline]
    pub(crate) fn into_result<T>(self) -> self::Result<T> {
        Err(Error(ErrorKind::MathError(self)))
    }
}
