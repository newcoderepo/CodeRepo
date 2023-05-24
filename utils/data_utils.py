
# We adaot the partition method from FedLab-NLP/fedlab/utils/dataset/functional.py
# For details, please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

from tqdm import trange
import numpy as np
import random
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

import pandas as pd
from pathlib import Path
import math, random
import torch
import torchaudio

from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

from tqdm import tqdm
from utils.functional import hetero_dir_partition, partition_report,setup_seed,label_skew_quantity_based_partition


# others
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, target in dataloader:
        #print(inputs.shape) torch.Size([1, 3, 32, 32])
        #inputs = inputs[:,0,:,:,:]
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print(mean, std)
    return mean, std

def split_data(dataset, num_classes, num_clients, split_method, split_para):

    if dataset == 'Cifar10':
        mean, std = [0.49139968, 0.48215827, 0.44653124],[0.24703233, 0.24348505, 0.26158768]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data
        
        data = trainset.data
        targets = trainset.targets    
        test_targets = testset.targets.tolist()
        test_datas = [testset.data[i].tolist() for i in range(len(test_targets))]
    
    
    ## download from https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
    ## preprocessing from https://www.kaggle.com/code/longx99/sound-classification
    if dataset == 'UrbanSound':
        metadata_file = "./data/UrbanSound8K/metadata/UrbanSound8K.csv"
        data_path = './data/UrbanSound8K/audio/'
        df = pd.read_csv(metadata_file)
        df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
        df = df[['relative_path', 'classID']]
        print(df.head())
        
        myds = SoundDS(df, data_path)

        # Random split of 80:20 between training and validation
        num_items = len(myds)
        num_train = round(num_items * 0.8)
        num_val = num_items - num_train
        train_ds, val_ds = random_split(myds, [num_train, num_val])
        print(len(train_ds), len(val_ds))

        
        data = []
        targets = []
        test_datas = []
        test_targets = []
        for dat, target in tqdm(train_ds):
            data.append(dat) 
            targets.append(target)
        for dat, target in tqdm(val_ds):
            test_targets.append(target) 
            test_datas.append(dat)
            
        # for i, temp in enumerate(train_ds):
            # dat, target = temp
            # data.append(dat) 
            # targets.append(target)
            # if i == 100:
                # break
        # for i, temp in enumerate(val_ds):
            # dat, target = temp
            # test_targets.append(target) 
            # test_datas.append(dat)
            # if i == 50:
                # break


    ### 
    setup_seed(1)
    NUM_USERS = num_clients    # total number of clients

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': [] ,'local_ood': {},  'global_test':{}}
    #                          local test                          local ood              


    # CREATE USER DATA SPLIT
    if split_method == 'quantity':
        client_dict = label_skew_quantity_based_partition(targets, NUM_USERS, num_classes, major_classes_num = int(split_para))
        major_classes_num = int(split_para)
    elif split_method == 'distribution':
        print('Here')
        client_dict = hetero_dir_partition(targets, NUM_USERS, num_classes, dir_alpha=split_para)
        print('Done')
        major_classes_num = int(num_classes)
        
    partition_report(targets,client_dict,class_num=num_classes,file='data_split_report.txt')


    number = np.zeros([NUM_USERS, num_classes])


    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        idx = client_dict[i]

        if dataset == 'Cifar10':
            X = data[idx].tolist()
            y = targets[idx].tolist()
       
        if dataset == 'UrbanSound':
            X = [data[k] for k in idx]
            y = [targets[k] for k in idx]
        
        
        idx = list(range(len(y)))
        random.seed(1)
        random.shuffle(idx)
        idx = idx[:] #use all
        
        for j in idx:
            number[i][y[j]] +=1

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': [X[j] for j in idx], 'y': [y[j] for j in idx]}
        train_data['num_samples'].append(len(idx))


        ###### local testing set, may not be used
        numbers = Counter(y)
        
        present_majority = [l[0] for l in numbers.most_common(major_classes_num)]  #for quanlity distribution it is present class
        present_minority = list(set(range(num_classes))-set(present_majority))   #miniroty class, unpresent
       

        present_idx = []              #local iid
        unpresent_idx = []            #local ood, for dir, may be zero
        for j in range(len(test_targets)):
            if test_targets[j] in present_majority:
                present_idx.append(j) 
            if test_targets[j] in present_minority:
                unpresent_idx.append(j)    
                
        present_X = [test_datas[j] for j in present_idx]
        present_y = [test_targets[j] for j in present_idx]
        test_len = len(present_y)    # local iid testing length
        
         
        random.seed(1)
        random.shuffle(unpresent_idx)
        unpresent_idx = unpresent_idx[:test_len]
        unpresent_X = [test_datas[j] for j in unpresent_idx]
        unpresent_y = [test_targets[j] for j in unpresent_idx]

        if split_method == 'quantity':
            print(i,np.unique(y), np.unique(present_y), np.unique(unpresent_y))
            print(i, 'training:', Counter(y), 'testing iid:', Counter(present_y))#, 'testing ood:', Counter(unpresent_y))
         
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x':  present_X, 'y': present_y}
        test_data['num_samples'].append(test_len)
        #test_data['local_ood'].append('x':  unpresent_X, 'y': unpresent_y)

        
    test_data['global_test']['x'] = test_datas
    test_data['global_test']['y'] = test_targets ##list variable
 
     
    print("Num_samples of Training set per client:", train_data['num_samples'])
    print("Total_training_samples:", sum(train_data['num_samples']))
    print("Global test set:", len(test_targets))
    print("Finish Generating Samples, distribution saved")

    #np.savetxt('./data/cifar10_' + str(num_clients) + '_'  +str(split_method) + '_' +str(split_para) + '.txt', number,fmt='%d')
    
    return train_data, test_data









class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
          # Nothing to do
          return aud

        if (new_channel == 1):
          # Convert from stereo to mono by selecting only the first channel
          resig = sig[:1, :]
        else:
          # Convert from mono to stereo by duplicating the first channel
          resig = torch.cat([sig, sig])

        return ((resig, sr))
    
    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
    
    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)
    
    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
    
    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
    
    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
        

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id
        