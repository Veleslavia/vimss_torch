from __future__ import print_function, division

import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from configs import ex, logger
from utils import AttrDict

from datasets.urmp import URMP
from datasets.data_utils import _save_segments_to_songs

from models.waveunet import WaveUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelActionPipeline:

    def __init__(self, model, train_loader, val_loader, exp_config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = SummaryWriter(log_dir='./runs/{exp_id:05d}'.format(exp_id=ex.current_run._id))
        self.save_cp = exp_config.save_cp
        self.sr = exp_config.expected_sr
        self.cp_path = os.path.join(exp_config.dir_checkpoint, '{run_id:05d}'.format(run_id=ex.current_run._id))
        if self.save_cp and not os.path.exists(self.cp_path):
            os.mkdir(self.cp_path)
        self.exp_config = exp_config

        self.optimizer = optim.Adam(model.parameters(), lr=exp_config.init_lr, weight_decay=0.0005)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min',
                                                              patience=exp_config.patience,
                                                              verbose=True)
        self.criterion = nn.MSELoss(reduction='sum')
        self.num_epochs = exp_config.num_epochs

    def _write_summary(self, phase, iteration, loss, lr):
        self.writer.add_scalar('loss/{}'.format(phase), loss, iteration)
        self.writer.add_scalar('lr/{}'.format(phase), lr, iteration)

    def _get_test_segments(self):
        songs = dict()
        for sample_idx in range(len(self.val_loader.dataset)):
            inputs, gt_sources = self.val_loader.dataset[sample_idx]
            if self.exp_config.conditioning:
                mix = torch.Tensor(np.expand_dims(inputs[0], axis=0)).to(device, dtype=torch.float)
                labels = inputs[1].to(device, dtype=torch.float)
                output = np.array(self.model(mix, labels).detach().data.cpu()[0])
            else:
                mix = torch.Tensor(np.expand_dims(inputs, axis=0)).to(device, dtype=torch.float)
                output = np.array(self.model(mix).detach().data.cpu()[0])

            sample_datafile = self.val_loader.dataset.index_map[sample_idx].datafile
            sample_offset = self.val_loader.dataset.index_map[sample_idx].offset
            filenames = self.val_loader.dataset.metadata[sample_datafile][:, [sample_offset]][0][0]
            local_segment_idx = self.val_loader.dataset.metadata[sample_datafile][:, [sample_offset]][1][0]
            piece_name = filenames[0].split('/')[-2]
            if piece_name not in songs.keys():
                songs[piece_name] = [(source_filename.split('/')[-1], list()) for source_filename in filenames[1:]]
            for source_idx in range(len(output)):
                songs[piece_name][source_idx][1].append((local_segment_idx, output[source_idx][0]))
        return songs

    def train_one_epoch(self, loader, phase, epoch):
        running_loss = 0.0
        for num_it, (inputs, gt_sources) in enumerate(loader):
            if self.exp_config.conditioning:
                mix = inputs[0].to(device, dtype=torch.float)
                labels = inputs[1].to(device, dtype=torch.float)
            else:
                mix = inputs.to(device, dtype=torch.float)
                labels = None
            gt_sources = gt_sources.to(device, dtype=torch.float)
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(mix, labels)
                if self.model.context:
                    outputs = outputs[:, :, :, self.exp_config.output_padding[0]: -self.exp_config.output_padding[1]]
                loss = self.criterion(outputs, gt_sources)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            # statistics
            running_loss += loss.item() * mix.size(0)
            iteration = num_it + len(loader.dataset)*epoch
            self._write_summary(phase, iteration, loss.item(), self.optimizer.param_groups[-1]['lr'])

        epoch_loss = running_loss / len(loader.dataset)
        logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

        return epoch_loss

    def train_model(self):

        self.model.to(device)
        for epoch in range(self.num_epochs):
            logger.info('Starting epoch {}/{}.'.format(epoch, self.num_epochs))

            for phase in ['train', 'validation']:
                if phase == 'train':
                    loader = self.train_loader
                    self.model.train()  # Set model to training mode
                else:
                    loader = self.val_loader
                    self.model.eval()   # Set model to evaluate mode

                try:
                    epoch_loss = self.train_one_epoch(loader, phase, epoch)

                except KeyboardInterrupt:
                    torch.save(self.model.state_dict(), os.path.join(self.cp_path, 'INTERRUPTED.pth'))
                    logger.info('Saved interrupt')
                    sys.exit(0)

            self.scheduler.step(epoch_loss)

            if self.save_cp:
                torch.save(self.model.state_dict(),
                           os.path.join(self.cp_path, 'CP{epoch_id:04d}.pth'.format(epoch_id=epoch)))
                logger.info('Checkpoint {} saved !'.format(epoch))

    def test_model(self, model_checkpoint, output_dir):

        self.model.load_state_dict(torch.load(model_checkpoint))
        self.model.eval()
        self.model.to(device)

        songs = self._get_test_segments()
        _save_segments_to_songs(songs, output_dir=output_dir, sr=self.sr)


@ex.automain
def main(exp_config):
    # for convenient attribute access
    exp_config = AttrDict(exp_config)
    model = WaveUNet(n_sources=exp_config.num_sources, conditioning=exp_config.conditioning)
    loaders = [torch.utils.data.DataLoader(URMP(dataset_dir=os.path.join(exp_config.dataset_dir, data_type),
                                                conditioning=exp_config.conditioning),
                                           batch_size=exp_config.batch_size,
                                           shuffle=shuffle, num_workers=8)
               for data_type, shuffle in zip(['train', 'test'], [True, False])]
    pipeline = ModelActionPipeline(model=model,
                                   train_loader=loaders[0],
                                   val_loader=loaders[1],
                                   exp_config=exp_config)

    if exp_config.mode == 'train':
        pipeline.train_model()
    elif exp_config.mode == 'test':
        checkpoint_path = os.path.join(exp_config.dir_checkpoint, exp_config.model_checkpoint)
        pipeline.test_model(model_checkpoint=checkpoint_path,
                            output_dir=exp_config.output_dir)
