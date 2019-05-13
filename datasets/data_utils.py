import os

import numpy as np
import librosa


def _save_segments_to_songs(songs, output_dir, sr):
    for piece_name in songs.keys():
        piece_dir = os.path.join(output_dir, piece_name)
        if not os.path.exists(piece_dir):
            os.mkdir(piece_dir)

        for source_idx in range(len(songs[piece_name])):
            source_name = songs[piece_name][source_idx][0]
            unsorted_data = songs[piece_name][source_idx][1]
            source_wav = np.concatenate([data for idx, data in (sorted(unsorted_data, key=lambda x: x[0]))])
            librosa.output.write_wav(os.path.join(piece_dir, '{:02d}_'.format(source_idx) + source_name),
                                     source_wav, sr)

# mu-law quantization is copied from https://github.com/ibab/tensorflow-wavenet
def mu_law_encode(audio, quantization_channels=256):
    """Quantizes waveform amplitudes."""

    mu = (quantization_channels - 1)*1.0
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(int)


def mu_law_decode(output, quantization_channels=256):
    """Recovers waveform from quantized values."""
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * ((output*1.0) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return np.sign(signal) * magnitude
