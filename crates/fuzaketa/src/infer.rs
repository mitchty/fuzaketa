//! Autoregressive text generations
use crate::{model::Model, tokenizer::Tokenizer};
use burn::{prelude::*, tensor::activation};
use rand::{
    SeedableRng,
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
};
use std::io::{self, Write};

/// Feed `prompt` through `model`, then sample `n_new_tokens` worth of tokens
/// one at a time, printing each as it's produced to make this look like
/// something better than it is.
pub fn infer<B: Backend>(
    model: &Model<B>,
    tokenizer: &impl Tokenizer,
    prompt: &str,
    n_new_tokens: usize,
    context_length: usize,
    seed: u64,
) {
    let device = <B as Backend>::Device::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ids = tokenizer.encode(prompt);

    println!("prompt: {prompt}");
    io::stdout().flush().ok();
    for _ in 0..n_new_tokens {
        let x = {
            let start = (ids.len() as isize - context_length as isize).max(0) as usize;
            let ids_sliced = &ids[start..];
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(ids_sliced.to_vec(), [1, ids_sliced.len()]),
                &device,
            )
        };
        let logits = model.forward(x);
        let n = logits.dims()[1];
        let last = logits.slice_dim(1, (n - 1)..n).flatten::<1>(0, 2);
        let probs = activation::softmax(last, 0)
            .into_data()
            .to_vec::<f32>()
            .expect("softmax output should be readable as f32");

        // Never sample the trailing unknown/catch-all token? K? k.
        let known = &probs[..probs.len() - 1];
        let distribution =
            WeightedIndex::new(known).expect("token probabilities should be valid weights");
        let prediction = distribution.sample(&mut rng) as i32;

        ids.push(prediction);
        print!("{}", tokenizer.decode(&[prediction]));
        io::stdout().flush().ok();
    }
    println!();
}
