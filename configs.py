import logging

from sacred import Experiment
from sacred.observers import FileStorageObserver

logger = logging.getLogger('train_logger')
logger.setLevel(logging.INFO)

ex = Experiment('waveunet')
ex.observers.append(FileStorageObserver.create('experiments_meta'))

@ex.config
def cfg():
    # Base configuration
    exp_config = dict(mode='train',
                      dataset_dir='/home/olga/Datasets/urmpv2numpy_nomulaw/',
                      num_frames=16384,
                      output_padding=[2, 2],

                      num_sources=13,
                      num_channels=1,
                      num_layers=12,
                      filter_size=15,
                      merge_filter_size=5,
                      num_initial_filters=24,
                      conditioning='none',  # none, multi, concat, sum, attention

                      batch_size=16,

                      init_lr=1e-4,
                      num_epochs=2000,
                      evaluation_steps=1000,
                      patience=15,  # epochs before decreasing learning rate
                      # decay_steps=2000,
                      # decay_rate=0.96,

                      dir_checkpoint='./checkpoints',
                      save_cp=True,

                      # testing time parameters
                      output_dir='./eval',
                      model_checkpoint='CP250.pth',

                      load_model=True,

                      expected_sr=22050,
                      mono_downmix=True,
                      output_type='direct',
                      context=False,
                      network='unet',
                      upsampling='linear',
                      augmentation=True)


@ex.named_config
def context():
    logger.info("Training with input context and valid convolutions")
    exp_config = dict(
        dataset_dir='/home/olga/Datasets/urmpv2numpy_context/',
        context=True
    )

@ex.named_config
def test_context():
    logger.info("Testing")
    exp_config = dict(
        mode='test',
        dataset_dir='/home/olga/Datasets/urmpv2numpy_context/',
        context=True,
        model_checkpoint=None,
        dir_checkpoint='./checkpoints',
        output_dir=None
    )


@ex.named_config
def conditioning():
    logger.info("Training with input context and conditioning")
    exp_config = dict(
        dataset_dir='/home/olga/Datasets/urmpv2numpy_context/',
        context=True,
        conditioning='multi'
    )

