//! Training, wired onto burn's `Learner`/`SupervisedTraining` setup.
use crate::{
    dataset::{TokenBatch, TokenBatcher, TokenWindowDataset},
    model::{Model, ModelConfig},
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamWConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};
use std::{path::Path, sync::Arc};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Number of epochs to train our model on.
    ///
    /// Each epoch is `batches_per_step * batch_size` roughly randomly
    /// sampled context windows for the underlying training loop.
    pub n_steps: usize,
    /// Note this batch size with the test training data needs a gpu with 60GiB
    /// of vram. See micro_batch_size for how to split this apart on smaller.
    pub batch_size: usize,
    /// Samples actually forwarded/backwarded through the model at once. When
    /// smaller than `batch_size`, gradients are accumulated across
    /// micro-batches before each optimizer step, trading a bit of extra compute
    /// for much lower peak vram/ram use for training. Must be `<= batch_size`
    /// not really checked this is hold my beer demo code.
    pub micro_batch_size: usize,
    /// This dictates how a model adjusts parameters based on error gradients.
    /// The goal is to minimize loss as training occurs.
    pub learning_rate: f64,
    /// Per batch, this controls how many batches updates a model weight.
    pub batches_per_step: usize,
    /// This controls how a data set is validated against the model. For our
    /// purposes we'll largely ignore this but it is dependent upon the datasets
    /// being trained.
    pub validation_size: usize,
    /// This seed is used to initialize noise for training the data. We can
    /// largely ignore this for now.
    pub seed: u64,
    /// This is our model struct that describes the data we are training on.
    pub model: ModelConfig,
    /// This is our optimizer, for this test demo we'll use AdamW over tradition
    /// Adam which has issues we shant go into.
    /// Ref: https://optimization.cbe.cornell.edu/index.php?title=AdamW for more detail.
    pub optimizer: AdamWConfig,
}

impl<B: Backend> Model<B> {
    fn forward_classification(&self, batch: TokenBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.x);
        let [b, t, c] = logits.dims();
        let output = logits.reshape([b * t, c]);
        let targets = batch.y.reshape([b * t]);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());
        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = TokenBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TokenBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = TokenBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TokenBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch)
    }
}

/// Train `config.model` on `data_train`, reporting a validation loss sampled
/// from `data_val` each epoch. Checkpoints composing model + optimizer +
/// lr-scheduler state as well as an `experiment.log` are written under
/// `artifact_dir` as training continues.
///
/// Note corpus size is bounded by host RAM rather than GPU/accelerator VRAM in
/// this setup to allow for training on smaller vram equipped gpus.
pub fn train<B: AutodiffBackend>(
    config: &TrainingConfig,
    data_train: Arc<[i32]>,
    data_val: Arc<[i32]>,
    device: &B::Device,
    artifact_dir: &Path,
) -> Model<B::InnerBackend> {
    B::seed(device, config.seed);

    let micro_batch_size = config.micro_batch_size.min(config.batch_size);
    let grad_accumulation = config.batch_size.div_ceil(micro_batch_size);
    let epoch_size = config.batches_per_step * config.batch_size;

    let dataset_train = TokenWindowDataset::new(
        data_train,
        config.model.context_length,
        epoch_size,
        config.seed,
    );
    let dataset_val = TokenWindowDataset::new(
        data_val,
        config.model.context_length,
        config.validation_size,
        config.seed.wrapping_add(1),
    );

    let dataloader_train = DataLoaderBuilder::new(TokenBatcher)
        .batch_size(micro_batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(dataset_train);
    let dataloader_val = DataLoaderBuilder::new(TokenBatcher)
        .batch_size(micro_batch_size.min(config.validation_size.max(1)))
        .num_workers(1)
        .build(dataset_val);

    let model = config.model.init::<B>(device);
    let optimizer = config.optimizer.init::<B, Model<B>>();

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_val)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .grads_accumulation(grad_accumulation)
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.n_steps)
        .summary();

    training
        .launch(Learner::new(model, optimizer, config.learning_rate))
        .model
}
