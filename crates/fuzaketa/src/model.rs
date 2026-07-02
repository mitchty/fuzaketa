//! Toy GPT-2-style decoder-only transformer.
use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig, loss::CrossEntropyLossConfig,
    },
    prelude::*,
    tensor::activation,
};

/// Weight init used throughout the model, matching GPT-2's N(0, 0.02).
fn normal_init() -> Initializer {
    Initializer::Normal {
        mean: 0.0,
        std: 0.02,
    }
}

/// Weight init for the output projection of each residual sub-block,
/// scaled down to account for accumulation over `n_layers` residual
/// additions for reason ref the GPT-2 paper, section 2.3
fn residual_init(n_layers: usize) -> Initializer {
    Initializer::Normal {
        mean: 0.0,
        std: 0.02 / (2.0 * n_layers as f64).sqrt(),
    }
}

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    dropout: Dropout,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
    linear: Linear<B>,
}

#[derive(Debug, Config)]
pub struct ModelConfig {
    pub context_length: usize,
    pub vocab_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    #[config(default = "0.2")]
    pub dropout: f64,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model)
                .with_initializer(normal_init())
                .init(device),
            positional_embedding: EmbeddingConfig::new(self.context_length, self.d_model)
                .with_initializer(normal_init())
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            blocks: (0..self.n_layers)
                .map(|_| {
                    BlockConfig::new(self.d_model, self.d_hidden, self.n_heads, self.n_layers)
                        .with_dropout(self.dropout)
                        .init(device)
                })
                .collect(),
            norm: LayerNormConfig::new(self.d_model).init(device),
            linear: LinearConfig::new(self.d_model, self.vocab_size)
                .with_initializer(normal_init())
                .init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = input.clone();

        let x = {
            let emb_tok = self.token_embedding.forward(x.clone());
            let emb_pos = {
                let [_, t] = input.dims();
                self.positional_embedding
                    .forward(Tensor::arange(0..(t as i64), &x.device()).unsqueeze())
            };
            emb_tok + emb_pos
        };
        let x = self.dropout.forward(x);
        let x = self.blocks.iter().fold(x, |x, block| block.forward(x));
        let x = self.norm.forward(x);

        self.linear.forward(x)
    }

    pub fn loss(&self, logits: Tensor<B, 3>, targets: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        let [b, t, c] = logits.dims();
        CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.reshape([b * t, c]), targets.reshape([b * t]))
    }
}

#[derive(Debug, Config)]
struct BlockConfig {
    d_model: usize,
    d_hidden: usize,
    n_heads: usize,
    n_layers: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl BlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            norm_1: LayerNormConfig::new(self.d_model).init(device),
            multi_head: MultiHeadAttentionConfig::new(self.d_model, self.n_heads, self.n_layers)
                .with_dropout(self.dropout)
                .init(device),
            norm_2: LayerNormConfig::new(self.d_model).init(device),
            pwff: PositionWiseFeedForwardConfig::new(self.d_model, self.d_hidden, self.n_layers)
                .with_dropout(self.dropout)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
struct Block<B: Backend> {
    norm_1: LayerNorm<B>,
    multi_head: MultiHeadAttention<B>,
    norm_2: LayerNorm<B>,
    pwff: PositionWiseFeedForward<B>,
}

impl<B: Backend> Block<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = input;
        let x = x.clone() + self.multi_head.forward(self.norm_1.forward(x));
        x.clone() + self.pwff.forward(self.norm_2.forward(x))
    }
}

#[derive(Debug, Config)]
struct MultiHeadAttentionConfig {
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl MultiHeadAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let d_k = self.d_model / self.n_heads;
        assert!(
            d_k * self.n_heads == self.d_model,
            "d_model must be divisible by n_heads"
        );
        MultiHeadAttention {
            n_heads: self.n_heads,
            d_k,
            query: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .with_initializer(normal_init())
                .init(device),
            key: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .with_initializer(normal_init())
                .init(device),
            value: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .with_initializer(normal_init())
                .init(device),
            out: LinearConfig::new(self.d_model, self.d_model)
                .with_bias(false)
                .with_initializer(residual_init(self.n_layers))
                .init(device),
            attn_dropout: DropoutConfig::new(self.dropout).init(),
            resid_dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Debug, Module)]
struct MultiHeadAttention<B: Backend> {
    n_heads: usize,
    d_k: usize,
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    attn_dropout: Dropout,
    out: Linear<B>,
    resid_dropout: Dropout,
}

impl<B: Backend> MultiHeadAttention<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, t, _] = input.dims();

        let q = self.query.forward(input.clone());
        let k = self.key.forward(input.clone());
        let v = self.value.forward(input);

        let q = q.reshape([b, t, self.n_heads, self.d_k]).swap_dims(1, 2);
        let k = k.reshape([b, t, self.n_heads, self.d_k]).swap_dims(1, 2);
        let v = v.reshape([b, t, self.n_heads, self.d_k]).swap_dims(1, 2);

        let x = q.matmul(k.transpose()).div_scalar((self.d_k as f32).sqrt());
        let x = {
            let mask = Tensor::<B, 2, Bool>::tril_mask([t, t], 0, &x.device());
            x.mask_fill(mask.unsqueeze(), f32::NEG_INFINITY)
        };
        let x = activation::softmax(x, 3);
        let x = self.attn_dropout.forward(x);
        let x = x.matmul(v);
        let x = x.swap_dims(1, 2).reshape([b, t, self.n_heads * self.d_k]);
        let x = self.resid_dropout.forward(x);

        self.out.forward(x)
    }
}

// Position feed-forward network, GPT-2 uses GELU
#[derive(Debug, Config)]
struct PositionWiseFeedForwardConfig {
    d_model: usize,
    d_hidden: usize,
    n_layers: usize,
    #[config(default = "0.2")]
    dropout: f64,
}

impl PositionWiseFeedForwardConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward {
            linear_1: LinearConfig::new(self.d_model, self.d_hidden)
                .with_initializer(normal_init())
                .init(device),
            gelu: Gelu::new(),
            linear_2: LinearConfig::new(self.d_hidden, self.d_model)
                .with_initializer(residual_init(self.n_layers))
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Debug, Module)]
struct PositionWiseFeedForward<B: Backend> {
    linear_1: Linear<B>,
    gelu: Gelu,
    linear_2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear_1.forward(input);
        let x = self.gelu.forward(x);
        let x = self.linear_2.forward(x);

        self.dropout.forward(x)
    }
}
