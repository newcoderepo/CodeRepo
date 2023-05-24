import copy
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from FLAlgorithms.trainmodel.models import *
from FLAlgorithms.users.userbase import User
from torch import nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader

class UserMain(User):   #model = model, model_name
    def __init__(self, numeric_id, train, test, test_ood, test_gb_ood, model, layer, percent, fea_dim, is_interpolated, 
                        batch_size, learning_rate, local_epochs, optimizer, 
                        personal_learning_rate, device, output_dim=10, loss='NLL', beta=1):
                                 
        super().__init__(numeric_id, train, test, test_ood, test_gb_ood, model, layer, percent, batch_size, learning_rate,
                         local_epochs, device, output_dim=output_dim,loss=loss,beta=beta)

        self.batch_size = batch_size
        self.N_Batch = len(train) // batch_size + 1
        self.lr = learning_rate
        self.plr = personal_learning_rate
        self.beta = beta
        self.loss = loss
        self.device = device 
        self.fea_dim = fea_dim
        self.is_interpolated = is_interpolated


    def train(self, epochs, glob_iter):
        LOSS = 0
        N_Samples = 1
        Round = 1

        lr_decay = 1-glob_iter*0.018 if glob_iter < 50 else 0.1
        print('loss:', self.loss, 'learning rate:', self.plr*lr_decay)

        self.personal_model.train()
        self.global_model.eval()

        losses = torch.zeros(self.local_epochs) 
        losses2 = torch.zeros(self.local_epochs) 
       
        if self.loss == 'CE':  ##for baseline
            Loss = nn.CrossEntropyLoss(reduction='mean')
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y)
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
            print('training loss:', losses.mean()/self.N_Batch)

        if self.loss == 'CE_Prox':
            '''
            from https://github.com/ki-ljl/FedProx-PyTorch/blob/main/client.py
            '''
            Loss = nn.CrossEntropyLoss(reduction='mean')
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y)
                    
                    mu = self.beta
                    proximal_term = 0.0
                    for w, w_t in zip(self.personal_model.parameters(), self.global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss += (mu / 2) * proximal_term
                    #print(loss.item(), (mu / 2) * proximal_term.item())
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
                    
            print('training loss:', losses.mean()/self.N_Batch)

        if self.loss == 'CE_LC': 
            Loss = nn.CrossEntropyLoss(reduction='mean')
            def refine_loss(logits, targets):
                tau = self.beta
                num_classes = logits.shape[1]
                cla_fre = self.class_fre + 1e-6
                cal = cla_fre.repeat(logits.size(0), 1).to(logits.device)
                logits -= tau * cal**(-0.25)
                return logits

                # nt_positions = torch.arange(0, num_classes).to(logits.device)
                # nt_positions = nt_positions.repeat(logits.size(0), 1)
                # nt_positions = nt_positions[nt_positions[:, :] == targets.view(-1, 1)]
                # nt_positions = nt_positions.view(-1, 1)
                # t_logits = torch.gather(logits, 1, nt_positions)

                # t_logits = torch.exp(t_logits)
                # nt_logits = torch.exp(logits)
                # nt_logits = torch.sum(nt_logits, dim=1).view(-1,1) #- t_logits + 1e-6  #aviod inf
                
                # t_logits = torch.log(t_logits)
                # nt_logits = torch.log(nt_logits)
                
                # # print('t_logits', t_logits)
                # # print('nt_sum', nt_logits)
                # # print(t_logits - nt_logits)
                
                # #print(t_logits.shape, torch.sum(nt_logits, dim=1).shape)
                
                # loss = - t_logits + nt_logits
                # #print('loss:', loss)
                # return loss.mean()
 
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    logit = refine_loss(logit, batch_Y)
                    loss = Loss(logit,batch_Y)
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
            print('training loss:', losses.mean()/self.N_Batch)
            
        if self.loss == 'CE_RS': ##KDD FedRS baseline
            for epoch in range(self.local_epochs):
                batch_X,batch_Y = self.get_next_train_batch()
                output = self.personal_model(batch_X) #output logits
                rs_mask = torch.ones(output.shape).type(torch.float32).to(self.device)
                labels = torch.unique(batch_Y)
                for l in range(output.shape[1]):
                    if l not in labels:
                        rs_mask[:,l] = 0.5
                output = torch.mul(output,rs_mask)
                
                Loss = nn.CrossEntropyLoss()
                loss = Loss(output, batch_Y)
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.personal_model.parameters()),lr=self.plr*lr_decay) 
                self.optimizer.zero_grad()        
                loss.backward()
                nn.utils.clip_grad_norm_(self.personal_model.parameters(), 1) #gradient clip
                self.optimizer.step()
                losses[epoch] += loss.item()
            print('training loss:', losses.mean()/self.N_Batch)
            
        if self.loss == 'NT_CE': ## FedNTD baseline
            ###### 'code from Neurips paper'
            def refine_as_not_true(logits, targets):
                num_classes = logits.shape[1]
                nt_positions = torch.arange(0, num_classes).to(logits.device)
                nt_positions = nt_positions.repeat(logits.size(0), 1)
                nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
                nt_positions = nt_positions.view(-1, num_classes - 1)
                logits = torch.gather(logits, 1, nt_positions)
                return logits
            
        
            Loss = nn.CrossEntropyLoss(reduction='mean')
            KLDiv = nn.KLDivLoss(reduction="batchmean")
            
            tau = 1
         
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y)
                    #print('CE loss:', loss.item())
                    
                    # Get smoothed local model prediction
                    logits = refine_as_not_true(logit, batch_Y)
                    pred_probs = F.log_softmax(logits / tau, dim=1)

                    # Get smoothed global model prediction
                    dg_logits = self.global_model(batch_X)  #remove no gred
                    dg_logits = refine_as_not_true(dg_logits, batch_Y)
                    dg_probs = F.softmax(dg_logits / tau, dim=1) ##note here torch.softmax is used

                    kl_loss = self.beta * (tau ** 2) * KLDiv(pred_probs, dg_probs)  ## CE loss with KL for not-true class
                    #print('KL loss:', kl_loss.item())
                    loss += kl_loss
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()
                    losses2[epoch] += kl_loss.item()
                    
            print('training loss:', losses.mean()/self.N_Batch, 'KL loss:', losses2.mean()/self.N_Batch)
    
        # for key in self.model.state_dict().keys(): 
            # self.model.state_dict()[key].data.copy_(self.personal_model.state_dict()[key])
            
        return LOSS

    def personal_train(self, epochs, glob_iter): 
        LOSS = 0
        N_Samples = 1
        Round = 1
        
        lr_decay = 1-glob_iter*0.005 if glob_iter < 100 else 0.5

        self.personal_model.train()
        self.global_model.eval()


        losses = torch.zeros(self.local_epochs)#default 20
        
        if self.loss == 'CE':  ##for baseline
            Loss = nn.CrossEntropyLoss(reduction='mean')
            for epoch in range(self.local_epochs):
               
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y)
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss.backward()
                    self.optimizer.step()
                    losses[epoch] += loss.item()

        print('training loss:', losses.mean()/self.N_Batch)
                
    def post_train(self, epochs):   #used for baselines
  
        print('post-hoc retraining learning rate:', self.plr)
        print(self.modelname)
        
        self.personal_model.train()
        self.global_model.eval()
        
        # if self.modelname == 'MOBNET': ##Only update the final layer
            # for param in self.personal_model.conv1.parameters():
                # param.requires_grad = False
            # for param in self.personal_model.bn1.parameters():
                # param.requires_grad = False
            # for i in range(17):
                # for param in self.personal_model.layers[i].parameters():
                    # param.requires_grad = False    
            # for param in self.personal_model.conv2.parameters():
                # param.requires_grad = False
            # for param in self.personal_model.bn2.parameters():
                # param.requires_grad = False   


        if self.modelname == 'AudioNet':
            for param in self.personal_model.layers.parameters():
                param.requires_grad = False    
                
        losses = torch.zeros(self.local_epochs)
        if self.loss == 'CE':  ##for baseline
            Loss = nn.CrossEntropyLoss(reduction='mean')
            for epoch in range(self.local_epochs):
                for batch in range(self.feature_batch):
                    batch_X, batch_Y = self.get_next_feature_batch()
                    #print('batch feature:', batch_X.shape, batch_Y.shape)
                    
                    logit = self.personal_model(batch_X)
                    
                    loss = Loss(logit,batch_Y)
                    
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr)
                    self.optimizer.zero_grad()        
                    loss.backward() #retain_graph=True
                    self.optimizer.step()
                    losses[epoch] += loss.item()
                    
                print('training loss:', losses[epoch].mean()/self.N_Batch)
                self.test('0')
        
    def train_feature(self, epochs, glob_iter):
  
        lr_decay = 1-glob_iter*0.018 if glob_iter < 50 else 0.1
        print('Local adapation with distilling:', self.plr*lr_decay)
         
        #self.model.train()
        self.personal_model.train()
        self.global_model.eval()
        
            
        losses1 = torch.zeros(self.local_epochs)
        losses2 = torch.zeros(self.local_epochs)
        
        
        if self.loss == 'CE_CE':  ##for baseline 
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()

                    logit = self.personal_model(batch_X)
                    loss = Loss(logit,batch_Y) 

                    losses1[epoch] += loss.item()
                   
                    if self.layer >=0:
                        logit = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        loss2 = Loss(logit,batch_Y_fea)*self.beta
                    else:
                        logit = self.personal_model(batch_X_fea) #sharing raw data
                        loss2 = Loss(logit,batch_Y_fea)*self.beta
 
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()

                    losses2[epoch] += loss2.item()
         
        if self.loss == 'CE_CE_KL':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    loss2 = Loss(logit_lc,batch_Y_fea)
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea

        if self.loss == 'CE_KL':  ##CE and KL for global feature
            self.tau = 10
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    #loss2 = Loss(logit_lc,batch_Y_fea)
                    loss = loss1 #+loss2
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea
                    
        if self.loss == 'CE_CE_NT':   ## with FedNTD
            ###### 'code from Neurips paper'
            def refine_as_not_true(logits, targets):
                num_classes = logits.shape[1]
                nt_positions = torch.arange(0, num_classes).to(logits.device)
                nt_positions = nt_positions.repeat(logits.size(0), 1)
                nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
                nt_positions = nt_positions.view(-1, num_classes - 1)
                logits = torch.gather(logits, 1, nt_positions)
                return logits
        
            Loss = nn.CrossEntropyLoss(reduction='mean')
            KLDiv = nn.KLDivLoss(reduction="batchmean")
            
            tau = 1
         
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch):
                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y)
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_fe = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_fe_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_fe = self.personal_model(batch_X_fea) #sharing raw data
                        logit_fe_gb = self.global_model(batch_X_fea)
                         
                    loss2 = Loss(logit_fe,batch_Y_fea)
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
                    
                  
                    ## Global feature KL
                    logits_fe = refine_as_not_true(logit_fe, batch_Y_fea)
                    lc_probs_fe = F.log_softmax(logits_fe / tau, dim=1)
                    dg_logits_fe = refine_as_not_true(logit_fe_gb, batch_Y_fea)
                    dg_probs_fe = F.softmax(dg_logits_fe / tau, dim=1) ##note here torch.softmax is used
                    loss2 = self.beta * (tau ** 2) * KLDiv(lc_probs_fe, dg_probs_fe)
                    losses2[epoch] += loss2.item() 
                    
                    loss_all = loss + loss2
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                
                    del loss1, loss2, loss, loss_all, logit, logit_fe_gb, dg_logits_fe, dg_probs_fe, batch_X, batch_Y, batch_X_fea, batch_Y_fea
                  
        if self.loss == 'CE_CE_KL_Prox':  ##CE and KL for global feature
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    loss1 = Loss(logit,batch_Y) 
                    
                    ## prox loss
                    mu = 0.01
                    proximal_term = 0.0
                    for w, w_t in zip(self.personal_model.parameters(), self.global_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    proximal_term = (mu / 2) * proximal_term
                    
                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    loss2 = Loss(logit_lc,batch_Y_fea)
                    loss = loss1 + loss2 + proximal_term
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2 
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea, batch_Y_fea, proximal_term
        
        if self.loss == 'CE_CE_KL_LC':  ##CE and KL for global feature
            def refine_loss(logits, targets):
                #tau = self.beta
                num_classes = logits.shape[1]
                cla_fre = self.class_fre + 1e-6
                cal = cla_fre.repeat(logits.size(0), 1).to(logits.device)
                logits -= 1.0 * cal**(-0.25)
                return logits
                
            self.tau = 1
            KLLoss = nn.KLDivLoss(reduction='batchmean')
            Loss = nn.CrossEntropyLoss(reduction='mean')
           
            for epoch in range(self.local_epochs):
                for batch in range(self.N_Batch): #feature data

                    batch_X, batch_Y = self.get_next_train_batch()
                    batch_X_fea, batch_Y_fea = self.get_next_feature_batch()
                    
                    ## local data CE
                    logit = self.personal_model(batch_X)
                    logit = refine_loss(logit, batch_Y)
                    loss1 = Loss(logit,batch_Y) 

                    ## global data CE
                    if self.layer >= 0:
                        logit_lc = self.personal_model.forward_feature(batch_X_fea, idx=self.layer)
                        logit_gb = self.global_model.forward_feature(batch_X_fea, idx=self.layer)
                    else:
                        logit_gb = self.global_model(batch_X_fea)
                        logit_lc = self.personal_model(batch_X_fea)
                    
                    loss2 = Loss(logit_lc,batch_Y_fea)
                    loss = loss1+loss2
                    losses1[epoch] += loss.item()
   
                    ## global data distilling
                    pro_gb = F.softmax(logit_gb / self.tau, dim=1)     ## y
                    pro_lc = F.log_softmax(logit_lc / self.tau, dim=1) ## x
                    # The smallest KL is 0, always positive 
                    loss2 = self.beta * (self.tau ** 2) * KLLoss(pro_lc,pro_gb)
                    losses2[epoch] += loss2.item()
                   
                    loss_all = loss + loss2 
                    self.optimizer = torch.optim.Adam(self.personal_model.parameters(), lr=self.plr*lr_decay)
                    self.optimizer.zero_grad()        
                    loss_all.backward()  #retain_graph=True
                    self.optimizer.step()
                    
                    del loss1, loss2, loss, loss_all, logit, logit_gb, logit_lc, pro_gb, pro_lc, batch_X, batch_Y, batch_X_fea
                    
        print('training loss:', losses1.mean()/self.N_Batch, 'feature loss:', losses2.mean()/self.N_Batch)           
 