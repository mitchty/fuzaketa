// Cheat wgpu out of a silly recursion limit just so it compiles. This is demo
// quality code after all this isn't production.
#![recursion_limit = "512"]

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(not(feature = "cuda"))]
use burn::backend::Wgpu;
use burn::{
    backend::Autodiff,
    module::Module,
    optim::AdamWConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use clap::{Parser, Subcommand};
use fuzaketa::{
    TrainingConfig,
    model::ModelConfig,
    tokenizer::{CharTokenizer, Tokenizer},
};
use std::{fs, path::PathBuf, sync::Arc};

#[cfg(not(feature = "cuda"))]
type Backend = Wgpu<f32, i32>;
#[cfg(feature = "cuda")]
type Backend = Cuda<f32, i32>;

type TrainBackend = Autodiff<Backend>;

fn device() -> <Backend as burn::tensor::backend::Backend>::Device {
    #[cfg(not(feature = "cuda"))]
    {
        burn::backend::wgpu::WgpuDevice::default()
    }
    #[cfg(feature = "cuda")]
    {
        burn::backend::cuda::CudaDevice::default()
    }
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new toy GPT2ish model on some text corpus.
    Train {
        #[arg(short = 'o', long, value_name = "PATH")]
        output_path: Option<PathBuf>,
        #[arg(short = 'c', long, default_value_t = 64)]
        context_length: usize,
        #[arg(short = 'd', long, default_value_t = 64)]
        d_model: usize,
        #[arg(short = 'l', long, default_value_t = 2)]
        n_layers: usize,
        #[arg(long, default_value_t = 2)]
        n_heads: usize,
        #[arg(short = 'n', long, default_value_t = 50)]
        n_steps: usize,
        #[arg(short = 'b', long, default_value_t = 32)]
        batch_size: usize,
        #[arg(short = 'u', long, value_name = "N")]
        micro_batch_size: Option<usize>,
        #[arg(short = 'r', long, default_value_t = 0.003)]
        learning_rate: f64,
        #[arg(short = 's', long, default_value_t = 0)]
        seed: u64,
        #[arg(short = 't', long, default_value = "model/corpus.txt")]
        text_corpus: PathBuf,
        /// Only use the first N megabytes of the corpus for training.
        #[arg(short = 'm', long)]
        n_mega_bytes: Option<u64>,
        #[arg(short = 'N', long)]
        name: Option<String>,
        /// Don't save the trained model, which is only useful for quick debug
        /// runs unless you just like wasting power...
        #[arg(short = 'x', long)]
        no_save: bool,
    },

    Query {
        model_path: PathBuf,
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(short, long, default_value_t = 1000)]
        n_new_tokens: usize,
        #[arg(short, long, default_value_t = 0)]
        seed: u64,
    },
}

fn main() {
    tracing_subscriber::fmt::init();

    match Cli::parse().command {
        Commands::Train {
            output_path,
            context_length,
            d_model,
            n_layers,
            n_heads,
            n_steps,
            batch_size,
            micro_batch_size,
            learning_rate,
            seed,
            text_corpus,
            n_mega_bytes,
            name,
            no_save,
        } => {
            let tokenizer = CharTokenizer::new();

            println!(
                "loading {} of {text_corpus:?} as a training dataset",
                n_mega_bytes.map_or_else(|| "all".to_string(), |mb| format!("first {mb} MiB"))
            );

            // Scoped so the raw and filtered text buffers are freed as soon as
            // we've got token ids out of them, rather than sticking around in
            // host RAM for the entire training run doing jack and squat.
            let mut ids = {
                use std::io::Read;
                let mut file = fs::File::open(&text_corpus)
                    .unwrap_or_else(|e| panic!("failed to open {text_corpus:?}: {e}"))
                    .take(n_mega_bytes.map_or(u64::MAX, |mb| mb << 20));
                let mut text = String::new();
                file.read_to_string(&mut text)
                    .expect("corpus should be valid utf-8");
                let text = tokenizer.filter_known(&text);
                tokenizer.encode(&text)
            };
            // Kept host-side and not uploaded to the gpu so that corpus size is
            // bounded by host ram rather than GPU vram `Arc<[i32]>` so the
            // dataloader threads can share the corpus without cloning it and
            // wasting time with memcp()'s
            let val_ids: Arc<[i32]> = ids.split_off((0.9 * ids.len() as f32) as usize).into();
            let train_ids: Arc<[i32]> = ids.into();

            let device = device();

            let config = TrainingConfig {
                n_steps,
                batch_size,
                micro_batch_size: micro_batch_size.unwrap_or(batch_size),
                batches_per_step: 100,
                validation_size: 128,
                seed,
                learning_rate,
                model: ModelConfig {
                    context_length,
                    vocab_size: tokenizer.vocab_size(),
                    d_model,
                    d_hidden: 4 * d_model,
                    n_heads,
                    n_layers,
                    dropout: 0.2,
                },
                optimizer: AdamWConfig::new(),
            };

            let artifact_dir = output_path.unwrap_or_else(|| {
                let ts = std::time::UNIX_EPOCH.elapsed().unwrap().as_secs();
                let dir = match name.as_deref() {
                    Some(label) => format!("model/{label}-trained_{ts}"),
                    None => format!("model/trained_{ts}"),
                };
                PathBuf::from(dir)
            });
            fs::create_dir_all(&artifact_dir).ok();

            let model = fuzaketa::train::<TrainBackend>(
                &config,
                train_ids,
                val_ids,
                &device,
                &artifact_dir,
            );

            if !no_save {
                println!("store trained model to: {artifact_dir:?}");
                config
                    .save(artifact_dir.join("config.json"))
                    .expect("config should save sanely");
                model
                    .clone()
                    .save_file(artifact_dir.join("model"), &CompactRecorder::new())
                    .expect("model should save sanely");
            }

            println!("infer example text for funsies");
            fuzaketa::infer(
                &model,
                &tokenizer,
                "\n",
                200,
                config.model.context_length,
                seed,
            );
        }
        Commands::Query {
            model_path,
            prompt,
            n_new_tokens,
            seed,
        } => {
            let device = device();
            let tokenizer = CharTokenizer::new();
            let config = TrainingConfig::load(model_path.join("config.json"))
                .expect("config.json should exist; run train first");
            let model: fuzaketa::model::Model<Backend> = config.model.init(&device).load_record(
                CompactRecorder::new()
                    .load(model_path.join("model"), &device)
                    .expect("model weights should exist; run train first"),
            );

            fuzaketa::infer(
                &model,
                &tokenizer,
                &prompt.unwrap_or_else(|| "\n".into()),
                n_new_tokens,
                config.model.context_length,
                seed,
            );
        }
    }
}
