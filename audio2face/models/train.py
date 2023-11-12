import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from audio2face import nvidia_loss
import os
from tqdm import tqdm
class Args:
    def __init__(self):
        self.epochs = 200
        self.batch_size = 8
        self.lr = 0.001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.audio_path = ""
        self.blendshape_path = ""
        self.check_point_saved_path = ""
args = Args()
class Dataset_HDTF(Dataset):
    def __init__(self, audio_path, blendshape_path,flag='train'):
        assert flag in ['train', 'test']
        self.flag = flag
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data

        self.data = self.__load_data__(audio_path, blendshape_path)

    def __getitem__(self, index):
        # 根据索引返回数据
        # data = self.preprocess(self.data[index]) # 如果需要预处理数据的话
        audio, label = self.data[index]

        return torch.tensor(label,dtype=torch.float), torch.tensor(label, dtype=torch.float)

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    # 如果不是直接传入数据data，这里定义一个加载数据的方法
    def __load_data__(self, audio_path, blendshape_path):
        # 假如从 csv_paths 中加载数据，可能要遍历文件夹读取文件等，这里忽略
        # 可以拆分训练和验证集并返回train_X, train_Y, valid_X, valid_Y
        data = []
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                wav_file = os.path.join(root, file)
                blendshape_file = os.path.join(blendshape_path,file)
                wav_data = np.load(wav_file)
                blendshape_data = np.load(blendshape_file)
                data.append(wav_data,blendshape_data)
        if self.flag == "train":
            return data[:int(len(data)*0.8)]
        else:
            return data[int(len(data)*0.8):]

    def preprocess(self, data):
        # 将data 做一些预处理
        pass



def train(epochs,
          batch_size,
          lr,
          model,
          checkpoint_save_path,
          loss,
          optimizer,
          audio_path,
          blendshape_path
          ):

    train_dataset = Dataset_HDTF(flag="train",audio_path=audio_path, blendshape_path=blendshape_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)

    test_dataset = Dataset_HDTF(flag="test", audio_path=audio_path, blendshape_path=blendshape_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    train_epochs_loss = []
    test_epochs_loss = []
    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []
        acc, nums = 0,0
        for idx, (audio, label) in enumerate(tqdm(train_dataloader)):
            audio = audio.to(args.device)
            label = label.to(args.device)
            outputs = model(audio)
            optimizer.zero_grad()
            loss = nvidia_loss(outputs,label)
            optimizer.step()
            train_epoch_loss.append(loss.item())
            acc += torch.sum(torch.equal(outputs,label))
            nums += label.size()[0]
        train_epochs_loss.append(np.average(train_epoch_loss))
        train_acc.append(100.0*acc/nums)
        print("train acc = {:.3f}%, loss = {}".format(100 * acc / nums, np.average(train_epoch_loss)))
        with torch.no_grad():
            model.eval()
            test_epoch_loss = []
            acc, nums = 0, 0
            for idx, (audio, label) in enumerate(tqdm(test_dataloader)):
                audio = audio.to(args.device)
                label = label.to(args.device)
                outputs = model(audio)
                loss = nvidia_loss(outputs, label)
                test_epoch_loss.append(loss.item())
                acc += torch.sum(torch.equal(outputs,label))
                nums += label.size()[0]
            test_epochs_loss.append(np.average(test_epoch_loss))
            test_acc.append(100.0*acc/nums)

            print("epoch = {}, test acc = {:.2f}%, loss = {}".format(epoch, 100 * acc / nums,
                                                                      np.average(test_epoch_loss)))




