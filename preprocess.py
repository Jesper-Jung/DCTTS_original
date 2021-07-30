import numpy as np
import yaml
import os

import glob
from tqdm import tqdm
from utils import get_mel_spectrogram

config = yaml.load(
    open("./config.yaml", 'r'), Loader=yaml.FullLoader
)

""" Configuration """
wave_path = config["Dataset"]["wave_path"]
mels_path = config['Dataset']["mels_path"]

if not os.path.exists(mels_path):
    os.makedirs(mels_path)

# Save Mel spectrograms to array 
wav_path_list = glob.glob(wave_path + "/**", recursive=True)
wav_path_list = [item for item in wav_path_list if '.wav' in item]
print("{} wave files are loaded.".format(len(wav_path_list)))

# Save all
def _path_concat(ori_path):
    mel_file = ori_path.split('/')[-1].replace('.wav', '.npy')
    return os.path.join(mels_path, mel_file)

for path in tqdm(wav_path_list):
    mel = get_mel_spectrogram(path)
    mel_path = _path_concat(path)
    np.save(mel_path, mel)

print("Mel Spectrograms are Saved!!, {}".format(len(wav_path_list)))
    

