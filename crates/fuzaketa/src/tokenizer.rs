//! Minimal character-level tokenizer, this is the stupidest option for a
//! tokenizer ever. But its easy to understand so it stays.
//!
//! Gpt-2 uses the BPE, aka Byte Pair Encoding tokenizer in the "real world".
//! Ref: https://huggingface.co/learn/llm-course/en/chapter6/5 for more details.
//!
//! Gpt-4o uses TikToken https://github.com/openai/tiktoken instead.
//!
//! As the better and more complex tokenizer approaches tend to involve a round
//! trip through the data, this toy won't implement anything advanced.
use std::collections::HashMap;

/// Token ids are `i32` throughout as that is what burn's `Int` tensors want
/// anyway, ref: `Backend = Wgpu<f32, i32>` and `Cuda<f32, i32>`, and more
/// importantly it halves peak memory versus `usize` when holding a
/// multi-billion-token corpus in host memory.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<i32>;
    fn decode(&self, ids: &[i32]) -> String;
    fn vocab_size(&self) -> usize;
}

/// Fixed-vocabulary character tokenizer. Any character outside of the `CHARS`
/// string maps to a catch-all token id that is dropped when decoding is
/// attempted, so it's only useful as an "unknown" input token.
pub struct CharTokenizer {
    ttoi: HashMap<char, i32>,
    itot: HashMap<i32, char>,
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// While this is the dumbest option for a tokenizer, we constrain our model
/// size to be 102 individual characters. This drastically reduces the model
/// size and makes it slightly faster for inference cause we yeet out all of the
/// non token data.
///
/// The key takeaway for a tokenizer, is this *is* the core of a training loop for an llm.
///
/// Lets explain using the word understanding.
///
/// Outside of just splitting on each char, we have many options, like "split at first vowel"
/// understanding = u nde rsta ndi ng
///
/// This would mean if we hit under in the corpus under = u nde r so would be
/// how we might split a string into tokens. This is still a very poor
/// tokenization strategy however. Lets say we "know" most of our corpus is
/// English. We could imagine a tokenizer that would by default try to use its
/// understanding of english itself.
///
/// That is: understanding = under stand ing might be a more "intelligent"
/// tokenization strategy for English. It however would entirely fail for German:
/// Wellenreiten = nothing in english so we'd have to have a fallback but Auf Deutsch we would do:
/// Wellenreiten = Wellen reiten for the English users, this is lit: wave riding
/// or surfing. German speakers ignore the specifics to where this fails for now
/// this is mostly to contrast the data inputs versus tokenization strategies.
///
/// Now the key takeaway I want to convey with tokenization, is the result to
/// the final model size. Tokens, how they are created, and how they show up
/// within a training corpus, are what directly leads to the "size" of the final
/// LLM. Aka a 7b parameter model is "really" describing a model that is trained
/// against roughly 7b tokens of some variety. BUT, every model is trained
/// against a different idea of what a "token" is. They are specific to the
/// model and how well it does, or does not perform.
impl CharTokenizer {
    const CHARS: &'static str = "\n abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~0123456789äüöÄÖÜß";

    pub fn new() -> CharTokenizer {
        let itot = HashMap::from_iter(
            Self::CHARS
                .chars()
                .enumerate()
                .map(|(id, char)| (id as i32, char)),
        );
        let ttoi = HashMap::from_iter(
            Self::CHARS
                .chars()
                .enumerate()
                .map(|(id, char)| (char, id as i32)),
        );
        CharTokenizer { ttoi, itot }
    }

    /// Keep only characters this tokenizer knows about, useful for cleaning a
    /// raw text corpus before training any data. In a real world data pipeline
    /// the training data would be scrubbed prior to training. Since this is not
    /// a real world pipeline we just cheat and consider only the tokens we
    /// defined above to be the Pinnochios we train the model against.
    pub fn filter_known(&self, text: &str) -> String {
        text.chars().filter(|c| self.ttoi.contains_key(c)).collect()
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<i32> {
        let unknown = self.ttoi.len() as i32;
        text.chars()
            .map(|char| self.ttoi.get(&char).copied().unwrap_or(unknown))
            .collect()
    }

    fn decode(&self, ids: &[i32]) -> String {
        ids.iter().filter_map(|id| self.itot.get(id)).collect()
    }

    fn vocab_size(&self) -> usize {
        // +1 for the unknown/catch-all id.
        self.ttoi.len() + 1
    }
}
