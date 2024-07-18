import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2_StyleEncoder_Multilingual_Diffusion_Style

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style(args, configs, device, train=False):
  (preprocess_config, model_config, train_config) = configs

  model = FastSpeech2_StyleEncoder_Multilingual_Diffusion_Style(args, preprocess_config, model_config, train_config).to(
    device)
  if args.ckpt_path:
      ckpt = torch.load(args.ckpt_path)
      model.load_state_dict(ckpt["model"], strict=True)
      print("Load model: ", args.ckpt_path)

  model.eval()
  model.requires_grad_ = False
  return model

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath)
    print("Complete.")
    return checkpoint_dict

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(configs, device):
    preprocess_config, model_config, train_config = configs
    config = model_config
    path_script = train_config["checkpoint"]["path_hifigan"]
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open(os.path.join(path_script, "hifigan/config.json"), "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load(os.path.join(path_script, "hifigan/generator_LJSpeech.pth.tar"))
        elif speaker == "universal":
            ckpt = torch.load(os.path.join(path_script, "hifigan/generator_universal.pth.tar"))
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)
    max_value = 32766
    wavs = (
        wavs.cpu().numpy()
        * max_value
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
