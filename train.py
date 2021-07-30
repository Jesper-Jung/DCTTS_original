from typing_extensions import IntVar
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import yaml
from tqdm import tqdm
import argparse

from Network.module import *
from Network.model import *

from Dataset.dataset import *
from torch.utils.data import DataLoader
from loss_function import Binary_Divergence_Loss as BDLoss


class Train():
    def __init__(self, config, args):
        """ Path """
        self.log_path = config['Train']['log_path']
        self.model_log_path = config['Train']['model_log_path']


        """ Device """ 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Available devices: {}'.format(torch.cuda.device_count()))
        print('Using devices: {}'.format(args.device_ids))


        """ Learning HyperParameters """ 
        learning_rate = config["Train"]["learning_rate"]
        beta = config["Train"]["ADAM"]["beta"]
        epsilon = config["Train"]["ADAM"]["epsilon"]
    
        batch_size = config["Dataset"]["batch_size"]
        

        """ Restore step (Args) """
        self.restore_step = args.restore_step
        self.save_step = config["Train"]["save_step"]
        self.model_save_step = config["Train"]["model_save_step"]

        assert isinstance(self.restore_step, int), "Make sure to input restore_step arguments!"


        """ Dataloader """
        dataset = Individual_Dataset(config)
        self.train_loader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True,
            collate_fn = dataset.collate_fn,
            num_workers = 8
        )


        """ Model """
        assert isinstance(args.device_ids, int), "Make sure to input device_ids arguments!"
        self.TTS_Model = DCTTS(config)
        if args.device_ids == 1:
            self.TTS_Model = self.TTS_Model.to(self.device)
        else:
            self.TTS_Model = nn.DataParallel(self.TTS_Model, device_ids=list(range(args.device_ids))).to(self.device)


        """ Optimizer """
        self.optimizer = torch.optim.Adam(
            self.TTS_Model.parameters(),
            lr = learning_rate,
            betas = beta
        )

        """ Criterion """
        self.BD_loss = BDLoss()
        self.L1_loss = nn.L1Loss()


    def fit(self, tot_epoch):
        self.total_step = tot_epoch * len(self.train_loader)

        self.outer_bar = tqdm(total=self.total_step, desc="Training", position=0)
        self.outer_bar.n = self.restore_step
        
        for epo in range(tot_epoch):    
            self.inner_bar = tqdm(total=len(self.train_loader), desc="Epoch {}".format(epo+1), position=1)
            self.training()
            self.inner_bar.close()
            
    def training(self):
        for _, data in enumerate(self.train_loader):
            phoneme, mel_input, mel_target, scr_len, mel_len = self._unpack_numpy_to_torch(data)

            """ Training """
            # 1) zero grad
            self.optimizer.zero_grad()

            # 2) Forward
            mel_pred = self.TTS_Model(mel_input, phoneme)

            # 3) Calculate Loss
            bd_loss = self.BD_loss(mel_pred, mel_target)
            l1_loss = self.L1_loss(mel_pred, mel_target)
            loss = bd_loss + l1_loss

            # 4) Backward
            loss.backward()

            # 5) step
            self.optimizer.step()

            """ restore step update """
            self.restore_step += 1

            """ save and print """
            # log save
            if self.restore_step % self.save_step == 0:
                tot_loss = loss.item()
                bd_loss_print = bd_loss.item()
                l1_loss_print = l1_loss.item()

                message1 = "Step {}/{} || ".format(self.restore_step, self.total_step)
                message2 = "Total Loss: {} || Binary Diverge Loss: {} || L1 Loss: {}".format(
                    tot_loss, bd_loss_print, l1_loss_print
                )

                with open(os.path.join(self.log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")
                    f.close()

                self.outer_bar.write(message1 + message2)

            # model save
            if self.restore_step % self.model_save_step == 0:
                self._model_save(self.model_log_path+"/model_"+str(self.restore_step)+".pth.tar")

            self.outer_bar.update()
            self.inner_bar.update()

    def _unpack_numpy_to_torch(self, data):
        # ? Input Data
        # phoneme_list: (batch, scr_len)
        # mel_list: (batch, mel_len)
        # mel_target_list: (batch, mel_len)
        # scr_len_list: (batch), *numpy*, int
        # mel_len_list: (batch), *numpy*, int

        phoneme_list = torch.LongTensor(data[0]).to(self.device)
        mel_list, mel_target_list = map(lambda v: torch.FloatTensor(v).to(self.device), data[1:3])
        scr_len = data[3]
        mel_len = data[4]

        return phoneme_list, mel_list, mel_target_list, scr_len, mel_len

    def _model_save(self, save_name_pth):
        ### Save model to designated dir.
        if isinstance(self.TTS_Model, nn.DataParallel):
            torch.save({
                'model': self.TTS_Model.module.state_dict(),
                'optim': self.optimizer.state_dict(),
                'step': self.restore_step,
            }, save_name_pth)
        else:
            torch.save({
                'model': self.TTS_Model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'step': self.restore_step,
            }, save_name_pth)



def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int)
    parser.add_argument('--device_ids', type=int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    config = yaml.load(
        open("./config.yaml", 'r'), Loader=yaml.FullLoader
    )

    args = argument_parse()
    Train_Command = Train(config, args)

    tot_epoch = config["Train"]["epoch"]
    Train_Command.fit(tot_epoch)

    


