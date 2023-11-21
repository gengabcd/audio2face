import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
from tqdm import tqdm
import json
import random
import librosa
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
from faceformer import Faceformer, loss

class Args:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 1
        self.lr = 0.0001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.audio_path = "../../data/HDTF/audio"
        self.blendshape_path = "../../data/HDTF/blendshape"
        self.checkpoint_saved_path = "../checkpoint"
        self.res_path = "../res/res.json"
        self.feature_dim = 64
        self.blendshape = 52
        self.period = 30
        self.dataset = "HDTF"
args = Args()
class Dataset_HDTF(Dataset):
    def __init__(self, audio_path, blendshape_path,flag='train'):
        assert flag in ['train', 'test']
        self.flag = flag
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data

        self.data = self.__load_data__(audio_path, blendshape_path)

    def __getitem__(self, index):
        audio, label, one_hot = self.data[index]
        audio = torch.tensor(audio,dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        one_hot = torch.tensor(one_hot, dtype=torch.float)
        return audio,label,one_hot

    def __len__(self):
        return len(self.data)

    def __load_data__(self, audio_path, blendshape_path):
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        cnt = 6
        data = []
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                cnt -= 1
                if cnt == 0:
                    break
                wav_path = os.path.join(root, file)
                bf = file
                bf = bf.replace("wav","npy")
                blendshape_file = os.path.join(blendshape_path,bf)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                wav_data = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
                blendshape_data = np.load(blendshape_file)
                one_hot = [0,1]
                for i in range (blendshape_data.shape[0]//300):
                    w = wav_data[i*16000:(i+1)*16000]
                    b = blendshape_data[i*300:(i+1)*300]
                    data.append((w,b,one_hot))
        random.shuffle(data)

        if self.flag == "train":
            return data[:int(len(data)*0.8)]
        else:
            return data[int(len(data)*0.8):]

    def preprocess(self, data):
        # 将data 做一些预处理
        pass

def train(epochs,
          batch_size,
          model,
          audio_path,
          blendshape_path,
          optimizer,
          res_path,
          checkpoint_save_path,
          ):
    train_dataset = Dataset_HDTF(flag="train", audio_path=audio_path, blendshape_path=blendshape_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Dataset_HDTF(flag="test", audio_path=audio_path, blendshape_path=blendshape_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    train_epochs_loss = []
    test_epochs_loss = []

    for epoch in range(epochs):
        print("epoch: " + str(epoch))
        model.train()
        train_epoch_loss = []
        # acc, nums = 0,0
        for idx, (audio, label, one_hot) in enumerate(tqdm(train_dataloader)):

            audio = audio.to(args.device)
            label = label.to(args.device)
            one_hot = one_hot.to(args.device)
            outputs = model(audio,one_hot,label)
            # print("output: " + str(outputs.shape))
            # print("label: " + str(label.shape))
            optimizer.zero_grad()
            mloss = loss(outputs, label)
            # print(mloss)
            mloss.backward()
            optimizer.step()
            train_epoch_loss.append(mloss.item())
        train_epochs_loss.append(np.average(train_epoch_loss))
        print("train_loss = {}".format(np.average(train_epoch_loss)))
        if epoch % 10 == 0:
            it = epoch // 10
            dict_path = f"{checkpoint_save_path}/model_epochs_{it}.pth"
            # dict_path = checkpoint_save_path + "/model_epochs_" + str(it) + ".pt"
            torch.save(model.state_dict(), dict_path)
            print(dict_path + " saved")

        with torch.no_grad():
            model.eval()
            test_epoch_loss = []
            # acc, nums = 0, 0
            for idx, (audio, label, one_hot) in enumerate(tqdm(test_dataloader)):
                audio = audio.to(args.device)
                label = label.to(args.device)
                one_hot = one_hot.to(args.device)
                outputs = model(audio, one_hot, label)
                mloss = loss(outputs, label)
                test_epoch_loss.append(mloss.item())
            test_epochs_loss.append(np.average(test_epoch_loss))

            print("test_loss = {}".format(np.average(test_epoch_loss)))

        res = {
            "train_epochs_loss":train_epochs_loss,
            "test_epochs_loss":test_epochs_loss,
        }
        with open(res_path, 'w') as f:
            json.dump(res,f)
    pass

if __name__ == "__main__":
    model = Faceformer(args=args)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # def train(epochs,
    #           batch_size,
    #           model,
    #           audio_path,
    #           blendshape_path,
    #           optimizer,
    #           res_path,
    #           checkpoint_save_path,
    #           ):
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        model=model,
        audio_path=args.audio_path,
        blendshape_path=args.blendshape_path,
        optimizer=optimizer,
        res_path=args.res_path,
        checkpoint_save_path=args.checkpoint_saved_path
    )
