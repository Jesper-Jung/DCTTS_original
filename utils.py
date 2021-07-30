import numpy as np
import librosa
import yaml

config = yaml.load(
    open("./config.yaml", "r"), Loader=yaml.FullLoader
)

def get_mel_spectrogram(wav_path, trim_on=False):
    global config

    """ Configure """
    n_fft = config["Preprocess"]["n_fft"]
    n_hop = config["Preprocess"]["n_hop"]
    n_mel = config["Preprocess"]["n_mel"]
    mode_norm = config["Preprocess"]["mode_norm"]
    norm_coeff = config["Preprocess"]["norm_coeff"]
    sr = config["Preprocess"]["sr"]

    # load .wav file
    audio, _ = librosa.load(wav_path, sr=sr)

    # audio trim
    if trim_on:
        audio, _ = librosa.effects.trim(audio)
        audio = np.pad(audio, (n_hop * 6, n_hop * 4), 'constant', constant_values=0)

    # transform to mels
    linear = np.abs(librosa.stft(y = audio, n_fft = n_fft, hop_length = n_hop, center = False))
    
    # ! 참고
    # librosa.filters.mel 할 때 fmin=50, fmax=7600 정도로 하는 게 국룰이다.
    filter_banks = librosa.filters.mel(sr, n_fft, n_mel)
    mel = np.dot(filter_banks, linear)

    # normalization
    # ! 참고 1
    # 음성 합성 쪽에서 쓰는 normalization technique가 따로 있다.
    # 아마도 ref dB 같음.

    # ! 참고 2
    # Paper 에서는 mel frame을 4로 줄인다.
    # Text2Mel이 해야할 일을 SSRN 에게 Super-resolution 이라는 task
    # 로써 넘겨준 것.
    if mode_norm:
        mel = mel / np.max(mel)
        mel = np.power(mel, norm_coeff)

    

    return mel
        
def pad1D(data, max_len_mode = False):
    # ! Parameter
    # ? Input
    # data: list of data            || len(data) = batch
    # ? Output
    # output: array of data         || (input_list_len, max_len)
    # max_len: max length of list   || int

    max_len = max([_data.shape[0] for _data in data])
    
    def _pad1d(len_data):
        nonlocal max_len
        len_data = np.concatenate(
            [len_data, np.zeros(max_len - len(len_data))], axis=0
        )

        return len_data

    output = []
    for _data in data:
        output.append(_pad1d(_data))
    
    if max_len_mode:
        return np.asarray(output), max_len
    else:
        return np.asarray(output)


def pad2D(data, n_mel, max_len_mode = False):
    # ! Parameter
    # ? Input
    # data: list of data            || len(data) = batch
    # ? Output
    # output: array of data         || (input_list_len, n_mels, max_len)
    # max_len: max length of list   || int

    max_len = max([_data.shape[-1] for _data in data])
    
    def _pad2d(mel_data):
        nonlocal max_len
        mel_data = np.concatenate(
            [mel_data, np.zeros((n_mel, max_len - mel_data.shape[-1]))], axis=-1
        )

        return mel_data

    output = []
    for _data in data:
        output.append(_pad2d(_data))
    
    if max_len_mode:
        return np.asarray(output), max_len
    else:
        return np.asarray(output)













