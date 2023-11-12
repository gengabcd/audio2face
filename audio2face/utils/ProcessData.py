import numpy as np

from LPC import audioProcess
import os

def process_wavlist(data_path, saved_path):
    for root, dirs, files in os.walk(data_path):
        for file in files:
            wav_path = os.path.join(root, file)
            saved_file_name = file[:-3] + 'npy'
            saved_path_ = os.path.join(saved_path, saved_file_name)
            if not os.path.exists(saved_path_):
                inputData_array = audioProcess(wav_path)
                np.save(saved_path_,inputData_array)
                print(saved_path_)
if __name__ == '__main__':
    data_path = "../../data/HDTF/audio"
    saved_path = "../dataset/audio"
    process_wavlist(data_path,saved_path)