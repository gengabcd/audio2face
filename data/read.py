import numpy as np

if __name__ == '__main__':
    file_path = './HDTF/blendshape/RD_Radio1_000.npy'
    data = np.load(file_path)
    print(data.shape)