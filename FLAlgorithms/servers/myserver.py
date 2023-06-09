import torch
from tqdm import tqdm
import os

from FLAlgorithms.users.myuser import UserMain
from FLAlgorithms.servers.serverbase import Server 
 
from utils.model_utils import read_user_data, read_data
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable



# Implementation for FedAvg Server
class ServerMain(Server):
    def __init__(self, dataset, data, algorithm, model, batch_size, learning_rate, num_glob_iters, layer, percent, fea_dim, is_interpolated, 
                 local_epochs, optimizer, num_users, times, device, #The last line of user parameter
                 personal_learning_rate=0.001,output_dim=10, post_fix_str='', loss='NLL', beta=0.01, seed=0):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, num_glob_iters, layer, percent,
                         local_epochs, optimizer, num_users, times, device)

        # Initialize data for all  users
        self.algorithm = algorithm
        self.personal_learning_rate = personal_learning_rate
        self.post_fix_str = post_fix_str
        self.data = data #already split 
        self.dataset = dataset
        self.device = device
        self.loss = loss
        self.total_users = len(data[0]) #data[0] is the list of clients
        self.output_dim = output_dim
        self.fea_dim = fea_dim
        self.is_interpolated = is_interpolated
        self.seed = seed  # seed for client sampling 
        self.modelname = model[1]
        print('clients initializting...')
        print('output size:', output_dim)
        
        # global testing set
        id, train, test_data, test_ood, test_gb = read_user_data(0, self.data, dataset, device)
        self.test_batch_size = 128
        self.testallloader = DataLoader(test_gb, self.test_batch_size)## testing in batch
        
         
            
        for i in tqdm(range(self.total_users), total=self.total_users):
            id, train, test, test_ood, test_gb_ood = read_user_data(i, self.data, dataset, device)
            user = UserMain(id, train, test, test_ood, test_gb_ood, model, layer, percent, fea_dim, is_interpolated, 
                                 batch_size, learning_rate,local_epochs, optimizer, 
                                 personal_learning_rate, device, output_dim,loss,beta)
                                 
            self.users.append(user)
            self.total_train_samples += user.train_samples

            
        print("Number of users per round / total users:", num_users, " / " ,len(self.data[0]))
        print("Finished creating FL server.")

    def train(self):
        print('=== Training starts: algorithm', self.algorithm, '===')

        if self.algorithm == 'FedAvg' or self.algorithm == 'FedNTD' or self.algorithm == 'FedProx' or self.algorithm == 'FedLC' or self.algorithm == 'FedRS':
            ACC = 0
            for glob_iter in range(self.num_glob_iters):
                pd_accs = []
                gd_accs = []
                
                print("-------------Round number: ",glob_iter, " -------------")
                self.selected_users = self.select_users(glob_iter, self.num_users)## all participate
                self.send_parameters(self.loss)
 
                for  i, user in enumerate(self.selected_users):
                    user.train(self.local_epochs, glob_iter) #personalised model evaluation

                self.aggregate_parameters()
                if glob_iter % 1== 0:
                    global_acc = self.evaluate(glob_iter)  #global model evaluation
                    if global_acc > ACC:
                        self.save_model(self.post_fix_str)
                        ACC = global_acc
                        print('save a model')
                    
        if self.algorithm == 'CCVR':
        
            model_path = os.path.join("/auto/homes/tx229/federated/FL_v4/models", 'Dir_0.5') #based on FedAVG
            checkpoint_path = os.path.join(model_path, "server_FedAvg_MOBNET_Cifar10_loss_CE_epoch_10_100.pt")
            checkpoint= torch.load(checkpoint_path)
            print('Load model checkpoint from name succuessfully!')   
            self.model = checkpoint    
                
            self.evaluate(0)  #global model evaluation

            #Training with feature sharing
            self.selected_users = self.select_users_fea(100)## 10% of the feature, 
            self.send_parameters(self.loss)
            print('CCVR users selcted:', [user.id for user in self.selected_users])
            
            #collecting feature
            self.send_features()   #use raw but frozen feature extractor 
            
            #one round calibration
            for  i, user in enumerate(self.selected_users):
                if i==0: ## only use client 0 with local testing set
                    user.post_train(self.local_epochs) #local adaptation with top layers frozen 
                    user.test('r' + str(0) + '_u' + str(i)) #personalised model evaluation     

        if self.algorithm == 'FLea' or self.algorithm == 'Data': 
            ACC = 0
            
            # Training with feature sharing
            for glob_iter in range(self.num_glob_iters):
                pd_accs = []
                gd_accs = []
                
                print("-------------Round number: ",glob_iter, " -------------")
                self.selected_users = self.select_users(glob_iter, self.num_users, self.seed)## selected participate
                print('selected users:', [u.id for u in self.selected_users])
                self.send_parameters(self.loss) # Synchronize global model
                self.send_features()# aggragting and synchronize features
                
                for  i, user in enumerate(self.selected_users):
                    user.train_feature(self.local_epochs, glob_iter) #local adaptation with top layers frozen 
                    #user.test('r' + str(glob_iter) + '_u' + str(i)) #local model evaluation  

                self.aggregate_parameters()
                global_acc = self.evaluate(glob_iter)  #global model evaluation
                if global_acc > ACC:
                    self.save_model(self.post_fix_str)
                    ACC = global_acc
                    print('save a model')
                    

    def evaluate(self,glob_iter):
        self.model.eval() #global model
        test_batch = self.test_batch_size
 
        # PM_accs = []
        # for i in range(self.total_users):
            # id, train, test_data, test_ood, test_gb_ood = read_user_data(i, self.data, self.dataset, self.device)
            # self.testloader = DataLoader(test_data, test_batch)
            # test_acc = 0
            # num = 0
            # for data,label in self.testloader: 
                # output = self.model(data) 
                # test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                # num += len(data)
            # PM_accs.append(test_acc*1.0/num)
        # print('Avaraged GM acc on local data:', np.mean(PM_accs), 'length of data:', num)
        # #print(PM_accs)
        
        test_acc = 0
        num = 0
        for data,label in self.testallloader:
            output = self.model(data) 
            test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
            #test_acc += torch.sum(torch.argmax(output, dim=1) == torch.argmax(label, dim=1)).item() 
            num += len(data)
            del data, label, output
        print('Global Model Acc on global data:', test_acc*1.0/num, 'length of data:', num)
        
        return test_acc*1.0/num    


    def save_model(self, post_fix_str):
        model_path = os.path.join("models", 'saved')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        file_name = os.path.join(model_path, "server_" + post_fix_str + ".pt")
        torch.save(self.model, file_name)
        return file_name