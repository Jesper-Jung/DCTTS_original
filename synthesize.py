import torch
import numpy as np

import argparse
import yaml
import librosa
from tqdm import tqdm

from Network.model import *
from data.KoG2P.g2p import runKoG2P
from Dataset.phonemes_Table import *

class Synthesis():
    def __init__(self, config, args):
        """ Config """
        self.config = config

        """ Device """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        """ Path of trained model """
        self.model_log_path = args.model_log_path

        """ Model """
        self.model = self._load_model(args.load_step)

        """ Inference Hyperparameters """
        self.inf_max_frames = config['Test']['inference_max_mel_frames']
        self.n_mel = config['Preprocess']['n_mel']
        self.sr = config['Preprocess']['sr']
        self.n_hop = config['Preprocess']['n_hop']
        self.attn_save_step = config['Test']['attn_save_frame_step']
        self.norm_coeff = config['Preprocess']['norm_coeff']
        self.invert_coeff = config['Preprocess']['invert_coeff']
        
    def inference(self, syn_text):
        """
            Inference mel spectrogram corresponded from given text.
        """
        # ! Parameters
        # ? Output
        # mel: np.asarray

        result_mel_spec = torch.zeros((1, self.n_mel, 1)).float().to(self.device)

        input_phoneme = self._text2phoneIdx(syn_text)

        pbar = tqdm(range(self.inf_max_frames))
        with torch.no_grad():
            for step, bar in enumerate(pbar):
                # Inference, Greedy Decoding step by step
                output = self.model(result_mel_spec, torch.tensor(input_phoneme).unsqueeze(0).long().to(self.device))
                result_mel_spec = torch.cat(
                    [result_mel_spec, output[:, :, -1].unsqueeze(-1)], dim=-1
                )

                # Save Attention Maps
                if (step + 1) % self.attn_save_step == 0:
                    self.plot_attention_map(
                        np.asarray(self.model.attention_map.squeeze(0).tolist()), step + 1
                    )


        return self._denormalize(np.asarray(result_mel_spec.squeeze(0).tolist()))

    def synthesize(self, syn_mel):
        import soundfile as sf
        
        """
            Synthesis waveform from a denormalized mel spectrogram (used pretrained MelGAN)
        """

        # Load the pretrained vocoder, MelGAN
        # ! 참고
        # vocoder가 아닌 algorithm으로 Griffin-Lim이 있다.
        # 수학 이론으로 pseudo inverse가 있다.
        vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # Scaling the predicted mel, and synthesize!!!
        syn_mel = torch.log(torch.clamp(torch.tensor(syn_mel).float(), min=1e-5)).unsqueeze(0)
        syn_wave = vocoder.inverse(syn_mel)

        sf.write('./Inference_data/synthesized_wave.wav', np.asarray(syn_wave.squeeze(0).tolist()), self.sr)

        # ! 참고
        # World vocoder로 pitch 정보, aperiodic 정보 등을 조절할 수 있다.
        # Vocoder => voder (작은 정보들을 뽑아내는 일) + coder (정보들로 다시 합성하는 일)
        # f0 -> 모음, voice을 위주
        # smooth spec -> 전반적인 발음
        # aperiodic -> 자음을 위주


    def plot_melspectrogram(self, result):
        import matplotlib.pyplot as plt
        from librosa.display import specshow

        plt.figure(figsize=(10, 4))
        specshow(librosa.power_to_db(result, ref=np.max), y_axis='mel', sr=self.sr, hop_length=self.n_hop, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.tight_layout()
        plt.savefig('./Inference_data/Mel-Spectrogram example.png')
        plt.close()

    def plot_attention_map(self, attn_map, step):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 4))
        plt.matshow(attn_map)
        plt.title('Attention Map')
        plt.savefig('./Inference_data/Attention Map {}.png'.format(step))
        plt.close()

    def save_to_npy_melspectrogram(self, result):
        np.save("./Inference_data/inference_data.npy", result)

    def _load_model(self, load_step):
        model = DCTTS(self.config).to(self.device)

        if load_step:
            ckpt_path = os.path.join(
                self.model_log_path,
                "model_{}.pth.tar".format(load_step),
            )
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
            model.load_state_dict(ckpt["model"])

        model.eval()
        model.requires_grad_ = False
        return model

    def _text2phoneIdx(self, syn_text):
        """
            Convert given text to correspond phonemes
        """

        phoneme_seq = [' '] # initial sil.
        for graphemes in syn_text.split(' '):
            phonemes = runKoG2P(graphemes, './data/KoG2P/rulebook.txt')
            for phone in phonemes.split(' '):
                phoneme_seq.append(phone)
            phoneme_seq.append(' ')
        
        idx_list = np.asarray([np.where(phonemes_table == phone)[0][0] for phone in phoneme_seq])
        return idx_list

    def _denormalize(self, mel):
        return np.power(mel, self.invert_coeff/self.norm_coeff)

    



def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_step', type=int)
    parser.add_argument('--syn_text', type=str, help="Input text which you wanna synthesis.")
    parser.add_argument('--model_log_path', type=str, help="Input directory that you saved your trained model.")
    parser.add_argument('--inf_max_frame', type=int, default=300)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    print("Synthesis Start!!")
    
    config = yaml.load(
        open("./config.yaml", 'r'), Loader=yaml.FullLoader
    )
    args = argument_parse()

    synthesis_command = Synthesis(config, args)
    result_melspec = synthesis_command.inference(args.syn_text)
    synthesis_command.plot_melspectrogram(result_melspec)
    synthesis_command.save_to_npy_melspectrogram(result_melspec)
    synthesis_command.synthesize(result_melspec)
    
