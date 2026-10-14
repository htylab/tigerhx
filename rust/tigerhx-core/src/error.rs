use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A file or case that does not exist (HTTP 404).
    NotFound,
    /// A bad request or input file (HTTP 400).
    Invalid,
    /// The user stopped the job.
    Cancelled,
    Other,
}

#[derive(Debug, Clone)]
pub struct Error {
    pub kind: Kind,
    pub message: String,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(kind: Kind, message: impl Into<String>) -> Self {
        Error { kind, message: message.into() }
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Error::new(Kind::NotFound, message)
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Error::new(Kind::Invalid, message)
    }
    pub fn other(message: impl Into<String>) -> Self {
        Error::new(Kind::Other, message)
    }
    pub fn cancelled() -> Self {
        Error::new(Kind::Cancelled, "stopped")
    }
    pub fn is_cancelled(&self) -> bool {
        self.kind == Kind::Cancelled
    }
    /// Prefix the message with what was being done, keeping the kind.
    pub fn context(self, what: impl fmt::Display) -> Self {
        Error { kind: self.kind, message: format!("{what}: {}", self.message) }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        let kind = if e.kind() == std::io::ErrorKind::NotFound { Kind::NotFound } else { Kind::Other };
        Error::new(kind, e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::invalid(e.to_string())
    }
}

impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::other(format!("ONNX Runtime: {e}"))
    }
}

/// Shorthand for `Err(Error::invalid(format!(...)))`.
#[macro_export]
macro_rules! invalid {
    ($($arg:tt)*) => { Err($crate::error::Error::invalid(format!($($arg)*))) };
}
