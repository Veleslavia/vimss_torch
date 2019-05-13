# URMP dataset in FP32 and FP16 versions
# TODO from fastai.vision import to_fp16 (apply to the model!)
# learn = Learner(data, model, metrics=[accuracy]).to_fp16()

from collections import namedtuple
import math
import multiprocessing
from multiprocessing import Pool
import os
import random

import glob
import logging
import numpy as np
import torch
import torch.utils.data

import fire
import librosa

logger = logging.getLogger('urmp_logger')
logger.setLevel(logging.INFO)

source_map = {
    'mix': 0,
    'bn': 1,
    'cl': 2,
    'db': 3,
    'fl': 4,
    'hn': 5,
    'ob': 6,
    'sax': 7,
    'tba': 8,
    'tbn': 9,
    'tpt': 10,
    'va': 11,
    'vc': 12,
    'vn': 13,
}

URMPIndex = namedtuple('URMPIndex', ['datafile', 'offset'])


class URMP(torch.utils.data.Dataset):

    def __init__(self, dataset_dir, conditioning=False, context=True):
        metafiles = glob.glob(os.path.join(dataset_dir, '*.meta.npy'))
        if context:
            self.data = None
            mix_datafiles = glob.glob(os.path.join(dataset_dir, '*.mix.npy'))
            sources_datafiles = glob.glob(os.path.join(dataset_dir, '*.sources.npy'))
            self.mix_data = dict([(filename.split('.')[0], np.load(filename, mmap_mode='r'))
                                  for filename in mix_datafiles])
            self.sources_data = dict([(filename.split('.')[0], np.load(filename, mmap_mode='r'))
                                      for filename in sources_datafiles])
        else:
            self.mix_data = None
            self.sources_data = None
            datafiles = glob.glob(os.path.join(dataset_dir, '*[!meta].npy'))
            # load a dictionary with mmap
            self.data = dict([(filename.split('.')[0], np.load(filename, mmap_mode='r')) for filename in datafiles])

        self.conditioning = conditioning
        self.metadata = dict([(filename.split('.')[0], np.load(filename, allow_pickle=True)) for filename in metafiles])
        # create index_map for __getitem__
        self.index_map = [URMPIndex(filename, offset)
                          for filename in self.metadata.keys()
                          for offset in range(self.metadata[filename].shape[1])]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        if self.data:
            sample = np.expand_dims(self.data[self.index_map[index].datafile][self.index_map[index].offset], axis=1)
            mix, sources = sample[0], sample[1:]
        else:
            mix = np.expand_dims(self.mix_data[self.index_map[index].datafile][self.index_map[index].offset], axis=0)
            sources = np.expand_dims(self.sources_data[self.index_map[index].datafile][self.index_map[index].offset], axis=1)
        if not self.conditioning:
            return torch.Tensor(mix), torch.Tensor(sources)
        else:
            # 3 is labels axis TODO name them
            labels = self.metadata[self.index_map[index].datafile][3][self.index_map[index].offset]
            return (torch.Tensor(mix), torch.Tensor(labels)), torch.Tensor(sources)


def get_labels_from_filename(filename):
    labels = [0]*(len(source_map)-1)
    label_names = (os.path.basename(filename[0]).replace(".", "_")).split("_")[3:-1]
    for label_name in label_names:
        labels[source_map[label_name]-1] = 1
    return labels


class URMPDataProcessor(object):
    """
    Forming two set of files:
    1) Files with the metadata of the dataset data points
    2) Numpy files with data. If context is used, saves two numpy arrays with mix and sources
    """

    def __init__(self, raw_data_dir, output_dir):

        self.quantization = False
        self.with_context = True
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.training_directory = 'train'
        self.test_directory = 'test'

        self.training_shards = 6
        self.test_shards = 3

        if self.with_context:
            self.mix_with_padding = 147442
        else:
            self.mix_with_padding = 16384
        self.sample_rate = 22050  # Set a fixed sample rate
        self.num_samples = 16384  # get from parameters of the model
        self.channels = 1  # always work with mono
        self.num_sources = 13  # fix 13 channels for urmp instruments
        self.cache_size = 16  # load 16 audio files in memory, then shuffle examples and write a numpy array

    def _process_audio_files_batch(self, chunk_data):
        """Processes and saves list of audio files.
        Args:
            chunk_data: tuple of chunk_files and output_file
            chunk_files: list of strings; each string is a path to an wav file
            output_file: string, unique identifier specifying the data set
        """

        chunk_files, output_file = chunk_data[0], chunk_data[1]

        # Get training files from the directory name
        chk_filenames, chk_sample_indices, chk_data_buffers, chk_num_sources = [], [], [], []
        for track in chunk_files:
            # load all wave files into memory and create a buffer
            file_data_cache = list()
            for source in track:
                data, sr = librosa.core.load(source, sr=self.sample_rate, mono=True)
                # quantize and store as float16
                if self.quantization:
                    data = mu_law_decode(mu_law_encode(data, quantization_channels=256)).astype('float16')
                file_data_cache.append([track, len(data), data])

            for segment in self._get_segments_from_audio_cache(file_data_cache):
                chk_filenames.append(segment[0])
                chk_sample_indices.append(segment[1])
                chk_data_buffers.append(segment[2])
                chk_num_sources.append(segment[3])

        # shuffle all segments
        mapIndexPosition = list(zip(chk_filenames, chk_sample_indices, chk_data_buffers, chk_num_sources))
        random.shuffle(mapIndexPosition)
        chk_filenames, chk_sample_indices, chk_data_buffers, chk_num_sources = zip(*mapIndexPosition)
        labels = [get_labels_from_filename(chk_filenames[i]) for i in range(len(chk_filenames))]

        # save data and metadata
        np.save(output_file + '.mix', np.stack(np.array(chk_data_buffers)[:, 0]))
        np.save(output_file + '.sources',
                np.stack([np.stack(sample) for sample in np.array(chk_data_buffers)[:, 1:]]))
        np.save(output_file + '.meta', [chk_filenames, chk_sample_indices, chk_num_sources, labels])

        return output_file

    def _process_dataset(self, filenames,
                         output_directory,
                         prefix,
                         num_shards):
        """Processes and saves list of audio files as numpy arrays of chunks and metadata.
        Args:
        filenames: list of strings; each string is a path to an audio file
        channel_names: list of strings; each string is a channel name (vocals, bass, drums etc)
        labels: map of string to integer; id for all channel name
        output_directory: path where output files should be created
        prefix: string; prefix for each file
        num_shards: number of chucks to split the filenames into
        Returns:
        files: list of tf-record filepaths created from processing the dataset.
        """
        chunksize = int(math.ceil(len(filenames) / float(num_shards)))

        pool = Pool(multiprocessing.cpu_count()-1)

        def output_file(shard_idx):
            return os.path.join(output_directory, '%s-%.5d-of-%.5d' % (prefix, shard_idx, num_shards))

        # chunk data consists of chunk_filenames and output_file
        chunk_data = [(filenames[shard * chunksize: (shard + 1) * chunksize],
                      output_file(shard)) for shard in range(num_shards)]

        files = pool.map(self._process_audio_files_batch, chunk_data)

        return files

    def _get_segments_from_audio_cache(self, file_data_cache):
        """
        Args:
            file_data_cache: list of raw audio files, mix and k sources data: [filename, len(data), data]
        Returns:
             segments: k segments of raw data
                each one contains file_basename, sample_idx, k+1 raw data audio frames in a single list
        """
        segments = list()
        offset = (self.mix_with_padding - self.num_samples) // 2
        start_idx = offset
        end_idx = file_data_cache[0][1] - offset - 1
        for sample_idx in range((end_idx - start_idx) // self.num_samples):
            # sampling segments, ignore first and last incomplete segments
            # notice that we sample MIX_WITH_PADDING from the mix and central cropped NUM_SAMPLES from the sources
            segments_data = list()
            sample_offset_start = start_idx + sample_idx * self.num_samples
            sample_offset_end = start_idx + (sample_idx + 1) * self.num_samples
            # adding big datasample for mix
            segments_data.append(file_data_cache[0][2][sample_offset_start - offset:sample_offset_end + offset])
            # adding rest of the sources
            if len(segments_data[0]) != self.mix_with_padding:
                logger.error("We have a problem with %s, sample id %s, len %s" % (
                    file_data_cache[0][0][0], sample_idx, len(segments_data[0])))
            # assert len(segments_data[0]) == MIX_WITH_PADDING
            for source in file_data_cache[1:]:
                segments_data.append(source[2][sample_offset_start:sample_offset_end])
            segments.append([file_data_cache[0][0], sample_idx, segments_data, len(file_data_cache) - 1])
        return segments

    def get_wav(self, database_path):
        """ Iterate through .wav files from URMP dataset
            returns data_list: List[List[path_to_wavefiles]] """

        silence_path = '../silence.wav'
        track_list = []

        # Iterate through each tracks
        for folder in os.listdir(database_path):
            track_sources = [0 for i in range(14)]  # 1st index must be mix source + 13 individual sources

            # Create Sample object for each instrument source files present
            for filename in os.listdir(os.path.join(database_path, folder)):
                if filename.endswith(".wav"):
                    if filename.startswith("AuMix"):
                        # Place mix source to the first index
                        mix_path = os.path.join(database_path, folder, filename)
                        track_sources[0] = mix_path
                    else:
                        # Place Sample object mapping to its instrument index
                        source_name = filename.split('_')[2]
                        source_idx = source_map[source_name]
                        source_path = os.path.join(database_path, folder, filename)
                        track_sources[source_idx] = source_path

            # Create and insert silence Sample object for instruments not present in the track
            for i, track in enumerate(track_sources):
                if track == 0:
                    track_sources[i] = silence_path

            track_list.append(track_sources)

        return track_list

    def parallel_data_processor(self):
        training_files = self.get_wav(os.path.join(self.raw_data_dir, self.training_directory))
        test_files = self.get_wav(os.path.join(self.raw_data_dir, self.test_directory))

        # Create training data
        training_records = self._process_dataset(training_files,
                                                 os.path.join(self.output_dir, self.training_directory),
                                                 self.training_directory, self.training_shards)
        logger.info(training_records)

        # Create validation data
        test_records = self._process_dataset(test_files,
                                             os.path.join(self.output_dir, self.test_directory),
                                             self.test_directory, self.test_shards)
        logger.info(test_records)


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from data_utils import mu_law_encode, mu_law_decode
    fire.Fire(URMPDataProcessor)
