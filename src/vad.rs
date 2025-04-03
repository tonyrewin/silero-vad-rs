use crate::{Result, SileroVAD};
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};
use log::debug;

/// Speech timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechTimestamps {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
}

/// Iterator for processing audio in chunks
pub struct VADIterator {
    model: SileroVAD,
    threshold: f32,
    sampling_rate: u32,
    min_silence_duration_ms: u32,
    speech_pad_ms: u32,
    speech_start: Option<f32>,
    speech_end: Option<f32>,
    last_prob: f32,
}

impl VADIterator {
    /// Create a new VAD iterator
    pub fn new(
        model: SileroVAD,
        threshold: f32,
        sampling_rate: u32,
        min_silence_duration_ms: u32,
        speech_pad_ms: u32,
    ) -> Self {
        Self {
            model,
            threshold,
            sampling_rate,
            min_silence_duration_ms,
            speech_pad_ms,
            speech_start: None,
            speech_end: None,
            last_prob: 0.0,
        }
    }

    /// Reset the iterator state
    pub fn reset(&mut self) {
        self.speech_start = None;
        self.speech_end = None;
        self.last_prob = 0.0;
        self.model.reset_states(1);
    }

    /// Process a single audio chunk and return speech timestamps if detected
    pub fn process_chunk(&mut self, x: &ArrayView1<f32>) -> Result<Option<SpeechTimestamps>> {
        let prob = self.model.process_chunk(x, self.sampling_rate)?;
        let prob = prob[0];

        let mut result = None;
        let time_per_sample = 1.0 / self.sampling_rate as f32;
        let current_time = (x.len() as f32) * time_per_sample;

        if prob >= self.threshold {
            if self.speech_start.is_none() {
                self.speech_start = Some(current_time);
            }
            self.speech_end = Some(current_time);
        } else if self.speech_start.is_some() {
            let silence_duration = current_time - self.speech_end.unwrap();
            let silence_duration_ms = (silence_duration * 1000.0) as u32;

            if silence_duration_ms >= self.min_silence_duration_ms {
                let start = self.speech_start.unwrap();
                let end = self.speech_end.unwrap() + (self.speech_pad_ms as f32 / 1000.0);
                result = Some(SpeechTimestamps { start, end });
                self.reset();
            }
        }

        self.last_prob = prob;
        Ok(result)
    }

    /// Get speech timestamps for an entire audio file
    pub fn get_speech_timestamps(
        &mut self,
        audio: &ArrayView1<f32>,
        min_speech_duration_ms: u32,
        max_speech_duration_s: f32,
        _min_silence_duration_ms: u32,
        _speech_pad_ms: u32,
    ) -> Result<Vec<SpeechTimestamps>> {
        let mut timestamps = Vec::new();
        let chunk_size = if self.sampling_rate == 16000 { 512 } else { 256 };
        
        // Process audio chunks one at a time
        let mut i = 0;
        while i < audio.len() {
            let end = (i + chunk_size).min(audio.len());
            if end - i < chunk_size {
                break;
            }

            debug!("Processing chunk at position {}", i);
            
            // Process the chunk
            let window = audio.slice(ndarray::s![i..end]);
            if let Some(ts) = self.process_chunk(&window)? {
                if ts.end - ts.start >= min_speech_duration_ms as f32 / 1000.0
                    && ts.end - ts.start <= max_speech_duration_s
                {
                    timestamps.push(ts);
                }
            }

            i = end;
        }

        Ok(timestamps)
    }
} 