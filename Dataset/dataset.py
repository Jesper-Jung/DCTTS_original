### You run this script in the directory "./DCTTS" ###
import sys
sys.path.append("./data")
sys.path.append(".")

import numpy as np
import unicodedata
import os
import yaml
from tqdm import tqdm
from utils import pad1D, pad2D

from torch.utils.data import Dataset
from Dataset.phonemes_Table import *

from KoG2P.g2p import runKoG2P


class Individual_Dataset(Dataset):
    def __init__(
        self, config, test_mode=False
    ):
        """ Configuration """
        self.script_path = config["Dataset"]["script_path"] # default: './data/script.txt'
        self.mels_path = config["Dataset"]["mels_path"]
        self.n_mel = config["Preprocess"]["n_mel"]

        self.batch_size = config["Dataset"]["batch_size"]
        self.sort = config["Dataset"]["mode_sort"]
        self.drop_last = False
        
        # Load corresponded .wav files, and graphemes.
        f = tqdm(open(self.script_path).readlines(), desc='Reading to prepare dataset...')

        self.script_list = []
        self.mel_path_list = []
        for item in f:
            wav_path = item.split('|')[0]
            graphemes = item.split('|')[3]
            script = unicodedata.normalize('NFC', graphemes)

            self.script_list.append(script)
            self.mel_path_list.append(wav_path.split('/')[-1].replace('.wav', '.npy'))
        f.close()

        assert len(self.script_list) == len(self.mel_path_list), "Mel data and script data have to have same numbers"

        # Test to Load Dataset
        self.test_mode = test_mode
        if test_mode:
            self.pbar = tqdm(total=len(self.script_list), desc='Dataset Loading...')

    def __len__(self):
        return len(self.mel_path_list)

    def __getitem__(self, idx):
        # phoneme to index
        phoneme_seq = self.g2p(self.script_list[idx])
        phoneme_index = self.phone_to_idx(phoneme_seq, self.script_list[idx])

        # mels
        mel_target = np.load(os.path.join(self.mels_path, self.mel_path_list[idx]))
        mel_input = np.concatenate([np.zeros((self.n_mel, 1)), mel_target[:, :-1]], axis=-1)

        return {
            'phoneme_index': phoneme_index,
            'mel_input': mel_input,
            'mel_target': mel_target
        }

    def g2p(self, script):
        # ! Parameter
        # ? Input
        # script: script normalized by NFC.

        phoneme_seq = [' '] # initial sil.
        for graphemes in script.split(' '):
            phonemes = runKoG2P(graphemes, './data/KoG2P/rulebook.txt')
            for phone in phonemes.split(' '):
                phoneme_seq.append(phone)
            phoneme_seq.append(' ')
        
        return phoneme_seq

    def phone_to_idx(self, phonemes, script):
        try:
            idx_list = np.asarray([np.where(phonemes_table == phone)[0][0] for phone in phonemes])

            if self.test_mode:
                self.pbar.update()
                if self.pbar.n == len(self.mel_path_list):
                    self.pbar.close()

            return idx_list

        except IndexError as e:
            print(e)
            print("Phonemes and script which occurs the error: {}, {}".format(phonemes, script))


    def process_meta(self, batch):
        # preparing empty lists of batch data
        phoneme_list = []
        mel_list = []
        mel_target_list = []
        scr_len_list = []
        mel_len_list = []

        # process
        for d in batch:
            phoneme_list.append(d['phoneme_index'])
            scr_len_list.append(d['phoneme_index'].shape[-1])
            mel_list.append(d['mel_input'])
            mel_len_list.append(d['mel_input'].shape[-1])
            mel_target_list.append(d['mel_target'])

        # padding to sample data for same length to use a multiple GPUs, and change to numpy array.
        phoneme_list = pad1D(phoneme_list)
        mel_list = pad2D(mel_list, self.n_mel)
        mel_target_list = pad2D(mel_target_list, self.n_mel)
        scr_len_list = np.asarray(scr_len_list)
        mel_len_list = np.asarray(mel_len_list)

        return phoneme_list, mel_list, mel_target_list, scr_len_list, mel_len_list

    """ Collate function """    
    def collate_fn(self, batch):
        # ? Output
        # phoneme_list, mel_list, mel_target_list, scr_len_list, mel_len_list
        #=================================================================================

        # devide index for each batches.
        len_batch = len(batch)

        if self.sort:
            scr_len_list = np.asarray([len(d["phoneme_index"]) for d in batch]) # ex. (63, 128, 99)
            ind_arr = np.argsort(-scr_len_list) # ex. (2, 0, 1)
        else:
            ind_arr = np.arange(len_batch)
        
        batch = [batch[i] for i in ind_arr]
        
        batch_list = self.process_meta(batch)
        return batch_list



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    config = yaml.load(
        open("./config.yaml", 'r'), Loader=yaml.FullLoader
    )
    dataset = Individual_Dataset(config)


    batch_size = config["Dataset"]["batch_size"]
    test_dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = dataset.collate_fn
    )

    for i, data in enumerate(test_dataloader):
        print(data)
        break


