use crate::{Error, Result};
use ndarray::{Array1, Array2, ArrayView1};
use std::path::Path;
use ort::{
    execution_providers::{TensorRTExecutionProvider, CUDAExecutionProvider},
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use log::{info, debug};

/// Main Silero VAD model wrapper
pub struct SileroVAD {
    session: Session,
    context: Array2<f32>,
    last_sr: u32,
    last_batch_size: usize,
}

impl SileroVAD {
    /// Create a new Silero VAD model from an ONNX file
    pub fn new(model_path: &Path) -> Result<Self> {
        // Configure TensorRT provider
        let tensorrt_provider = TensorRTExecutionProvider::default()
            .with_device_id(0)  // Use the first GPU
            .build();
        
        // Configure CUDA provider as fallback
        let cuda_provider = CUDAExecutionProvider::default()
            .with_device_id(0)  // Use the first GPU
            .build();
        
        info!("Attempting to use TensorRT execution provider with CUDA fallback");
        
        // Load the model with optimizations and GPU support
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([tensorrt_provider, cuda_provider])?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        
        info!("Model loaded successfully with GPU support");

        Ok(Self {
            session,
            context: Array2::zeros((1, 64)),
            last_sr: 0,
            last_batch_size: 0,
        })
    }

    /// Reset the model's internal state
    pub fn reset_states(&mut self, batch_size: usize) {
        self.context = Array2::zeros((batch_size, 64));
    }

    /// Validate input audio chunk
    fn validate_input(&self, x: &ArrayView1<f32>, sr: u32) -> Result<()> {
        if sr != 16000 {
            return Err(Error::InvalidInput("Sampling rate must be 16kHz".into()));
        }
        if x.len() != 512 {
            return Err(Error::InvalidInput("Input chunk must be 512 samples".into()));
        }
        Ok(())
    }

    /// Process a single audio chunk
    pub fn process_chunk(&mut self, x: &ArrayView1<f32>, sr: u32) -> Result<Array1<f32>> {
        self.validate_input(x, sr)?;

        let batch_size = 1;
        if self.last_batch_size != batch_size {
            self.reset_states(batch_size);
        }

        if self.last_sr != 0 && self.last_sr != sr {
            self.reset_states(batch_size);
        }

        // Prepare input tensor
        let input = Array2::from_shape_fn((batch_size, x.len() + 64), |(i, j)| {
            if j < 64 {
                self.context[[i, j]]
            } else {
                x[j - 64]
            }
        });

        // Create input tensor
        let input_shape = input.shape().to_vec();
        let input_data = input.into_raw_vec();

        debug!("Processing input tensor of shape {:?}", input_shape);

        // Create input tensor with just the 'input' name
        let inputs = vec![
            ("input", Tensor::from_array((input_shape, input_data.clone()))?.into_dyn()),
        ];

        let outputs = self.session.run(inputs)?;
        
        // Update context from the last 64 elements of input_data
        let context_data = input_data[input_data.len()-64..].to_vec();
        self.context = Array2::from_shape_vec((batch_size, 64), context_data)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        
        self.last_sr = sr;
        self.last_batch_size = batch_size;

        // Return speech probability
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        Ok(Array1::from_vec(output_tensor.iter().cloned().collect::<Vec<f32>>()))
    }

    /// Process a batch of audio chunks
    pub fn process_batch(&mut self, batch: &[ArrayView1<f32>], sr: u32) -> Result<Vec<Array1<f32>>> {
        let batch_size = batch.len();
        if batch_size == 0 {
            return Ok(Vec::new());
        }

        // Validate all chunks in the batch
        for x in batch {
            self.validate_input(x, sr)?;
        }

        if self.last_batch_size != batch_size {
            self.reset_states(batch_size);
        }

        if self.last_sr != 0 && self.last_sr != sr {
            self.reset_states(batch_size);
        }

        // Prepare input tensor for the entire batch
        let mut input_data = Vec::with_capacity(batch_size * (batch[0].len() + 64));
        for i in 0..batch_size {
            // Add context for this batch item
            for j in 0..64 {
                input_data.push(self.context[[i, j]]);
            }
            // Add audio data for this batch item
            for j in 0..batch[i].len() {
                input_data.push(batch[i][j]);
            }
        }

        let input_shape = vec![batch_size, batch[0].len() + 64];
        debug!("Processing batch input tensor of shape {:?}", input_shape);

        // Create input tensor with just the 'input' name
        let inputs = vec![
            ("input", Tensor::from_array((input_shape, input_data.clone()))?.into_dyn()),
        ];

        let outputs = self.session.run(inputs)?;
        
        // Update context from the last 64 elements of each batch item
        let mut context_data = Vec::with_capacity(batch_size * 64);
        for i in 0..batch_size {
            let start = i * (batch[0].len() + 64) + batch[0].len();
            let end = start + 64;
            if end <= input_data.len() {
                context_data.extend_from_slice(&input_data[start..end]);
            } else {
                // If we don't have enough data, use zeros
                context_data.extend(std::iter::repeat(0.0).take(64));
            }
        }
        self.context = Array2::from_shape_vec((batch_size, 64), context_data)
            .map_err(|e| Error::InvalidInput(e.to_string()))?;
        
        self.last_sr = sr;
        self.last_batch_size = batch_size;

        // Return speech probabilities for each chunk
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            results.push(Array1::from_vec(vec![output_tensor[i]]));
        }
        Ok(results)
    }
} 