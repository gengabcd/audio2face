import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from audio2face import NvidiaModel, loss
import os
from tqdm import tqdm
import json
import random

class Args:
    def __init__(self):
        self.epochs = 201
        self.batch_size = 1
        self.lr = 0.001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.audio_path = "../../../dataset_HPC/audio"
        self.blendshape_path = "../../../HDTF/blendshape"
        self.checkpoint_saved_path = "../checkpoint"
        self.res_path = "../res/res.json"
        self.temporal_pairs = 50

args = Args()
# class Args:
#     def __init__(self):
#         self.epochs = 11
#         self.batch_size = 1
#         self.lr = 0.001
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.audio_path = "../dataset/audio"
#         self.blendshape_path = "../../data/HDTF/blendshape"
#         self.checkpoint_saved_path = "../checkpoint"
#         self.res_path = "../res/res.json"
# args = Args()

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
        # audio = audio[audio.shape[0]-label.shape[0]:]
        # print("getitem audio_size: "+ str(audio.shape))
        # print("getitem audio_size: "+ str(label.shape))
        audio = torch.tensor(audio,dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)
        # print("getitem taudio_size: " + str(audio.shape))
        # print("getitem taudio_size: " + str(label.shape))

        return audio,label

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    # 如果不是直接传入数据data，这里定义一个加载数据的方法
    def __load_data__(self, audio_path, blendshape_path):
        cnt = 6
        data = []
        temporal_pairs = args.temporal_pairs
        for root, dirs, files in os.walk(audio_path):
            for file in files:
                cnt -= 1
                if cnt == 0:
                    break
                wav_file = os.path.join(root, file)
                blendshape_file = os.path.join(blendshape_path,file)
                wav_data = np.load(wav_file)
                blendshape_data = np.load(blendshape_file)
                for i in range(blendshape_data.shape[0]//temporal_pairs):
                    w = wav_data[i*temporal_pairs:(i+1)*temporal_pairs]
                    b = blendshape_data[i*temporal_pairs:(i+1)*temporal_pairs]
                    data.append((w,b))
                # data.append((wav_data,blendshape_data))
                # print("load " + file)
                # print("wavsize: " + str(wav_data.shape))
                # print("blendshapesize: " + str(blendshape_data.shape))
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
          checkpoint_save_path,
          optimizer,
          audio_path,
          blendshape_path,
          res_path,
          ):

    train_dataset = Dataset_HDTF(flag="train",audio_path=audio_path, blendshape_path=blendshape_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)

    test_dataset = Dataset_HDTF(flag="test", audio_path=audio_path, blendshape_path=blendshape_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    train_epochs_loss = []
    test_epochs_loss = []
    # train_acc = []
    # test_acc = []

    # sheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)
    for epoch in range(epochs):
        print("epoch: " + str(epoch))
        model.train()
        train_epoch_loss = []
        # acc, nums = 0,0
        for idx, (audio, label) in enumerate(tqdm(train_dataloader)):
            # print("train audio:" + str(audio.shape))
            # print("train label:" + str(label.shape))
            audio = audio.to(args.device)
            label = label.to(args.device)
            label = torch.squeeze(label,0)
            outputs = model(audio)
            optimizer.zero_grad()
            mloss = loss(outputs,label)
            # print("loss: " + str(loss))
            mloss.backward()
            optimizer.step()
            # sheduler.step()
            train_epoch_loss.append(mloss.item())
            # sp = (outputs == label).sum()
            # acc += torch.sum(torch.eq(outputs,label))
            # nums += label.size()[0]*label.size()[1]
        train_epochs_loss.append(np.average(train_epoch_loss))
        # train_acc.append(100.0*acc/nums)
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
            for idx, (audio, label) in enumerate(tqdm(test_dataloader)):
                audio = audio.to(args.device)
                label = label.to(args.device)
                label = label.view(label.shape[1], -1)
                outputs = model(audio)
                mloss = loss(outputs, label)
                test_epoch_loss.append(mloss.item())
                # sp = torch.equal(outputs,label)
                # acc += torch.sum(torch.equal(outputs,label))
                # nums += label.size()[0]
            test_epochs_loss.append(np.average(test_epoch_loss))
            # test_acc.append(100.0*acc/nums)

            print("test_loss = {}".format(np.average(test_epoch_loss)))

        res = {
            "train_epochs_loss":train_epochs_loss,
            "test_epochs_loss":test_epochs_loss,
            # "train_acc":train_acc,
            # "test_acc":test_acc
        }
        with open(res_path, 'w') as f:
            json.dump(res,f)



if __name__ == "__main__":
    model = NvidiaModel()
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    # def train(epochs,
    #           batch_size,
    #           model,
    #           checkpoint_save_path,
    #           optimizer,
    #           audio_path,
    #           blendshape_path,
    #           ):
    train(epochs=args.epochs,
          batch_size=args.batch_size,
          model=model,
          checkpoint_save_path=args.checkpoint_saved_path,
          optimizer=optimizer,
          audio_path=args.audio_path,
          blendshape_path=args.blendshape_path,
          res_path=args.res_path
          )




