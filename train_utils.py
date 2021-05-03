import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from noise import NoisyDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader


class Train:
    
    def __init__(self, architecture, train_dir, valid_dir, params):
        
        self.cuda = params['cuda']
        if self.cuda:
            self.architecture = architecture.cuda()
        else:
            self.architecture = architecture

        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.noise_model = params['noise_model']
        self.crop_size = params['crop_size']
        self.clean_targs = params['clean_targs']
        self.lr = params['lr']
        self.epochs = params['epochs']
        self.batch_size = params['bs']
        

        self.loaded_train, self.loaded_valid = self.__getdataset__()
        self.optimizer = self.__getoptimizer__()
        self.scheduler = self.__getscheduler__()
        self.loss_fn = nn.MSELoss()

    
    def train(self):
        
        for _ in range(self.epochs):
            tr_loss = 0
            self.architecture.train()
            for _list in tqdm(self.loaded_train):
                if self.cuda:
                    source = _list[0].cuda()
                    target = _list[-1].cuda()
                else:
                    source = _list[0]
                    target = _list[-1]
                _pred = self.architecture(Variable(source))
                if len(_list) == 4:
                    if self.cuda:
                        mask = Variable(_list[1].cuda())
                    else:
                        mask = Variable(_list[1])
                    _loss = self.loss_fn(mask * _pred, mask * Variable(target))
                else:
                    _loss = self.loss_fn(_pred, Variable(target))
                tr_loss += _loss.data

                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
            
            val_loss = self.evaluate()
            #self.scheduler.step(val_loss)
            print(f'Training loss = {tr_loss}, Validation loss = {val_loss}')


    def evaluate(self):
        
        val_loss = 0
        self.architecture.eval()

        for _, _list in enumerate(self.loaded_valid):
            if self.cuda:
                source = _list[0].cuda()
                target = _list[-1].cuda()
            else:
                source = _list[0]
                target = _list[-1]
            _pred = self.architecture(Variable(source))
            if len(_list) == 4:
                if self.cuda:
                    mask = Variable(_list[1].cuda())
                else:
                    mask = Variable(_list[1])
                _loss = self.loss_fn(mask * _pred, mask * Variable(target))
            else:
                _loss = self.loss_fn(_pred, Variable(target))
            val_loss += _loss.data
        
        return val_loss

    
    def __getdataset__(self):
        
        train_data = NoisyDataset(self.train_dir, crop_size=self.crop_size, train_noise_model=self.noise_model,
                                        clean_targ=self.clean_targs)
        loaded_train = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        valid_data = NoisyDataset(self.valid_dir, crop_size=self.crop_size, train_noise_model=self.noise_model,
                                        clean_targ=True)
        loaded_valid = DataLoader(valid_data, batch_size=self.batch_size)

        return loaded_train, loaded_valid

    def __getoptimizer__(self):
        
        return optim.Adam(self.architecture.parameters(), self.lr)

    def __getscheduler__(self):
        
        return lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.epochs/4, factor=0.5, verbose=True)
    