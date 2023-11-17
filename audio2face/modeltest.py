import torch
from audio2face.models.audio2face import NvidiaModel
from audio2face.utils.LPC import audioProcess
import numpy as np
import pandas as pd
import sys

# import

if __name__ == "__main__":
    model = NvidiaModel()
    # load model weights
    # model weights are located in the dir: "../checkpoint/"
    model.load_state_dict(torch.load("checkpoint/model_epochs_14.pth", map_location='cpu'))
    model.eval()
    # load wav audio from wav_path
    wav_path = "../data/HDTF/audio/RD_Radio1_000.wav"
    audio = audioProcess(wav_path=wav_path)
    audio = torch.tensor([audio], dtype=torch.float)
    res = model(audio)
    res = res.detach().numpy()
    # save blendshape_output in output_path
    output_path = "npy/output.csv"
    np_to_csv = pd.DataFrame(data=res)
    np_to_csv.to_csv(output_path)