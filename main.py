from meshtalk.models.encoders import AudioEncoder
from meshtalk.utils.helpers import load_audio, audio_chunking
if __name__ == "__main__":
    encoder = AudioEncoder(latent_dim=128)
    audio = load_audio("data/RD_Radio1_000.wav")
    audio = audio_chunking(audio, frame_rate=30, chunk_size=16000)
    x = encoder(audio.unsqueeze(0))
    print(x.shape)

