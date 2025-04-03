pub mod model;
pub mod utils;
pub mod vad;

pub use model::SileroVAD;
pub use vad::{VADIterator, SpeechTimestamps};

/// Supported languages for VAD
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Russian,
    English,
    German,
    Spanish,
}

/// Error types for the Silero VAD library
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Model loading error: {0}")]
    ModelLoad(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Audio processing error: {0}")]
    AudioProcessing(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),
}

/// Result type for the Silero VAD library
pub type Result<T> = std::result::Result<T, Error>; 