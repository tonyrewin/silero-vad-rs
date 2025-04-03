use silero_vad::{SileroVAD, VADIterator};
use silero_vad::utils::{read_audio, save_audio};
use std::path::Path;
use log::info;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    // Check if model exists, if not, download it
    let model_path = Path::new("models/silero_vad.onnx");
    if !model_path.exists() {
        println!("Model not found. Please run 'make download-model' first.");
        return Ok(());
    }

    // Load the model
    info!("Loading VAD model...");
    let model = SileroVAD::new(model_path)?;
    
    // Create a VAD iterator
    let mut vad = VADIterator::new(
        model,
        0.5,  // threshold
        16000, // sampling rate
        100,   // min silence duration (ms)
        30,    // speech pad (ms)
    );

    // Check if input file exists
    let input_path = Path::new("examples/input.wav");
    if !input_path.exists() {
        println!("Input file not found. Please provide a WAV file at examples/input.wav");
        return Ok(());
    }

    // Read audio file
    println!("Reading audio file...");
    let audio = read_audio(input_path, 16000)?;
    println!("Audio loaded: {} samples", audio.len());
    
    // Get speech timestamps
    println!("Detecting speech...");
    let timestamps = vad.get_speech_timestamps(
        &audio.view(),
        250,    // min speech duration (ms)
        f32::INFINITY, // max speech duration (s)
        100,    // min silence duration (ms)
        30,     // speech pad (ms)
    )?;

    // Process timestamps
    println!("Found {} speech segments:", timestamps.len());
    for (i, ts) in timestamps.iter().enumerate() {
        println!("  {}. {:.2}s - {:.2}s (duration: {:.2}s)", 
                 i+1, ts.start, ts.end, ts.end - ts.start);
    }

    // Save speech segments to separate files
    if !timestamps.is_empty() {
        println!("Saving speech segments...");
        for (i, ts) in timestamps.iter().enumerate() {
            let output_path = format!("examples/speech_{}.wav", i+1);
            let speech_audio = silero_vad::utils::collect_chunks(&[ts.clone()], &audio, 16000)?;
            save_audio(&output_path, &speech_audio, 16000)?;
            println!("  Saved segment {} to {}", i+1, output_path);
        }
    }

    Ok(())
} 