//! In-memory windowed dataset over a tokenized corpus, feeds burn's
//! `Learner`/`SupervisedTraining` training loop.
use burn::data::dataset::Dataset;
use std::sync::Arc;

#[derive(Clone)]
pub struct TokenWindowDataset {
    data: Arc<[i32]>,
    context_length: usize,
    len: usize,
    seed: u64,
}

impl TokenWindowDataset {
    /// `len` is the number of pseudo-random windows exposed as one "epoch".
    /// `data` must have more than `context_length` tokens for input for this to be useful.
    pub fn new(data: Arc<[i32]>, context_length: usize, len: usize, seed: u64) -> Self {
        assert!(
            data.len() > context_length,
            "corpus has {} tokens, need more than context_length {context_length} to work",
            data.len()
        );
        Self {
            data,
            context_length,
            len,
            seed,
        }
    }

    /// Use splitmix64 as a cheap deterministic mix from an index to a window
    /// start position. Don't stare at it too long its lifted from gpt2 source.
    fn window_start(&self, index: usize) -> usize {
        let mut x = (index as u64).wrapping_add(self.seed);
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        (x as usize) % (self.data.len() - self.context_length)
    }
}

/// One item is a `context_length` token window aka a .0 and the same window
/// shifted by one token, aka the next-token targets .1 further.
impl Dataset<(Vec<i32>, Vec<i32>)> for TokenWindowDataset {
    fn get(&self, index: usize) -> Option<(Vec<i32>, Vec<i32>)> {
        if index >= self.len {
            return None;
        }
        let start = self.window_start(index);
        let x = self.data[start..start + self.context_length].to_vec();
        let y = self.data[start + 1..start + self.context_length + 1].to_vec();
        Some((x, y))
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// Stack `TokenWindowDataset` items into batched tensors, uploading only the
/// batch at hand to the gpu. This is meant to run on consumer hardware.
#[derive(Clone, Default)]
pub struct TokenBatcher;

#[derive(Clone, Debug)]
pub struct TokenBatch<B: burn::tensor::backend::Backend> {
    pub x: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    pub y: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
}

impl<B: burn::tensor::backend::Backend>
    burn::data::dataloader::batcher::Batcher<B, (Vec<i32>, Vec<i32>), TokenBatch<B>>
    for TokenBatcher
{
    fn batch(&self, items: Vec<(Vec<i32>, Vec<i32>)>, device: &B::Device) -> TokenBatch<B> {
        use burn::tensor::{Tensor, TensorData};

        let batch_size = items.len();
        let context_length = items.first().map_or(0, |(x, _)| x.len());

        let mut xs = Vec::with_capacity(batch_size * context_length);
        let mut ys = Vec::with_capacity(batch_size * context_length);
        for (x, y) in items {
            xs.extend(x);
            ys.extend(y);
        }

        let x = Tensor::from_data(TensorData::new(xs, [batch_size, context_length]), device);
        let y = Tensor::from_data(TensorData::new(ys, [batch_size, context_length]), device);
        TokenBatch { x, y }
    }
}
