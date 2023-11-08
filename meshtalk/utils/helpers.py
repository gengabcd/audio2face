import numpy as np
import torch
import torchaudio as ta

def load_audio(wave_file):
    audio, sr = ta.load(wave_file)
    if not sr == 16000:
        audio = ta.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # normalize such that energy matches average energy of audio used in training
    audio = 0.01 * audio / torch.mean(torch.abs(audio))
    return audio

def audio_chunking(audio: torch.Tensor, frame_rate: int = 30, chunk_size: int = 16000):
    """
    :param audio: 1 x T tensor containing a 16kHz audio signal
    :param frame_rate: frame rate for video (we need one audio chunk per video frame)
    :param chunk_size: number of audio samples per chunk
    :return: num_chunks x chunk_size tensor containing sliced audio
    """
    samples_per_frame = 16000 // frame_rate
    padding = (chunk_size - samples_per_frame) // 2
    audio = torch.nn.functional.pad(audio.unsqueeze(0), pad=[padding, padding]).squeeze(0)
    anchor_points = list(range(chunk_size//2, audio.shape[-1]-chunk_size//2, samples_per_frame))
    audio = torch.cat([audio[:, i-chunk_size//2:i+chunk_size//2] for i in anchor_points], dim=0)
    return audio