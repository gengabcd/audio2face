import librosa
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import numpy as np
def read_wav(wav_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
    input_values = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
    print(input_values.shape)
    pass

if __name__ == "__main__":
    wav_path = "01-01-01-01-01-01-01.wav"
    read_wav(wav_path)