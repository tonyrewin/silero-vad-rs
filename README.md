# Silero VAD - Rust Implementation

This is a Rust implementation of the [Silero Voice Activity Detection (VAD) model](https://github.com/snakers4/silero-vad). The original model is written in Python and uses PyTorch, while this implementation uses Rust with the `tract-onnx` crate for ONNX model inference.

## Features

- Voice Activity Detection (VAD) using the Silero model
- Support for both 8kHz and 16kHz audio
- Streaming VAD with iterator interface
- Audio file I/O utilities
- Efficient ONNX model inference
- Thread-safe implementation

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
silero-vad = "0.1.0"
```

## Usage

### Basic VAD

```rust
use silero_vad::{SileroVAD, VADIterator};
use silero_vad::utils::{read_audio, save_audio};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the model
    let model = SileroVAD::new("path/to/silero_vad.onnx")?;
    
    // Create a VAD iterator
    let mut vad = VADIterator::new(
        model,
        0.5,  // threshold
        16000, // sampling rate
        100,   // min silence duration (ms)
        30,    // speech pad (ms)
    );

    // Read audio file
    let audio = read_audio("input.wav", 16000)?;
    
    // Get speech timestamps
    let timestamps = vad.get_speech_timestamps(
        &audio.view(),
        250,    // min speech duration (ms)
        f32::INFINITY, // max speech duration (s)
        100,    // min silence duration (ms)
        30,     // speech pad (ms)
    )?;

    // Process timestamps
    for ts in timestamps {
        println!("Speech detected from {:.2}s to {:.2}s", ts.start, ts.end);
    }

    Ok(())
}
```

### Streaming VAD

```rust
use silero_vad::{SileroVAD, VADIterator};
use ndarray::Array1;

fn process_stream() -> Result<(), Box<dyn std::error::Error>> {
    let model = SileroVAD::new("path/to/silero_vad.onnx")?;
    let mut vad = VADIterator::new(model, 0.5, 16000, 100, 30);

    // Process audio chunks
    let chunk_size = 512; // for 16kHz
    let audio_chunk = Array1::zeros(chunk_size);
    
    if let Some(ts) = vad.process_chunk(&audio_chunk.view())? {
        println!("Speech detected from {:.2}s to {:.2}s", ts.start, ts.end);
    }

    Ok(())
}
```

## Model Files

You need to download the ONNX model file from the [original repository](https://github.com/snakers4/silero-vad). The model supports both 8kHz and 16kHz audio sampling rates.

## Performance

The Rust implementation is designed to be efficient and thread-safe. It uses:
- `tract-onnx` for optimized ONNX model inference
- `ndarray` for efficient array operations
- Zero-copy operations where possible
- Minimal memory allocations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original Silero VAD implementation by [snakers4](https://github.com/snakers4)
- ONNX runtime by [tract-onnx](https://github.com/snakers4/tract-onnx) 