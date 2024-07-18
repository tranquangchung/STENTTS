import re
import argparse
import os
import json
import torch
import yaml
import numpy as np

from utils.model import get_vocoder
from utils.model import get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style
from utils.tool_lexicon import processes_all_languages
from utils.model import vocoder_infer
from utils.tool_audio import preprocess_audio
import audio as Audio
import librosa
from scipy.io.wavfile import write
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_style(model, control_values, ref_audio):
    pitch_control, energy_control, duration_control = control_values
    _sftf = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        preprocess_config["preprocessing"]["stft"]["hop_length"],
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )

    wav, _ = librosa.load(ref_audio, sr=22050)
    wav = wav / max(abs(wav)) # normalize to 0-1
    path_reference = "./wav_output/{0}_X_{1}.wav".format(args.name, "original")
    if not os.path.exists(path_reference):
        write(path_reference, 22050, wav)
    ref_mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _sftf)
    ref_mel = torch.from_numpy(ref_mel_spectrogram).transpose(0,1).unsqueeze(0).to(device=device)
    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)
    return style_vector

def synthesize(model, configs, vocoder, batch, control_values, args, style_vector):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    # Forward
    src = torch.from_numpy(batch[4]).to(device=device)
    src_len = torch.from_numpy(batch[5]).to(device=device)
    language = torch.tensor([batch[2]]).to(device=device)
    max_src_len = batch[6]
    with torch.no_grad():
        output = model.inference(style_vector, src, language, src_len=src_len,
                                                 max_src_len=max_src_len, p_control=pitch_control,
                                                 e_control=energy_control, d_control=duration_control)
    output = output.transpose(1, 2)
    wav = vocoder_infer(output, vocoder, model_config, preprocess_config)
    write("./wav_output/{0}_{1}.wav".format(args.name, args.language), 22050, wav[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="name to save",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="language to synthesis",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")
    parser.add_argument(
        "--model",
        type=str,
        choices=["naive", "aux", "shallow"],
        required=True,
        help="training model type",
    )

    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model_fastSpeech2_StyleEncoder_MultiLanguage_Difffusion_Style(args, configs, device, train=False)
    vocoder = get_vocoder(configs, device)

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "languages.json")) as f:
        language_map = json.load(f)
    # Preprocess texts

    # language = ["chinese", "english", "indonesian", "japanese", "vietnamese"]
    language = ["english"]
    texts = [
        # "這是日本先進科學技術研究所研究團隊開發的系統",
        # "this is the system developed by the research team of the japan advanced institute of science and technology",
        # "ini adalah sistem yang dikembangkan oleh tim peneliti institut sains dan teknologi maju jepang",
        # "これは北陸先端科学技術大学院大学の研究チームが開発したシステムです",
        # "đây là hệ thống phát triển bởi đội nghiên cứu của viện khoa học và công nghệ tiên tiến nhật bản",
        "This is a picture of japan advanced institute of science and technology"
    ]
    os.makedirs("wav_output", exist_ok=True)
    control_values = args.pitch_control, args.energy_control, args.duration_control
    with torch.no_grad():
        style_vector = get_style(model, control_values, args.ref_audio)
    for lang, text in zip(language, texts):
        args.language = lang
        args.text = text

        ids = raw_texts = [args.text[:100]]
        print(lang)
        print(args.text)
        texts = np.array([processes_all_languages(args.text, args.language, preprocess_config)])
        speakers = np.array([args.speaker_id])
        text_lens = np.array([len(texts[0])])
        lang_id = language_map[args.language]
        batch = [ids, raw_texts, lang_id, speakers, texts, text_lens, max(text_lens)]
        synthesize(model, configs, vocoder, batch, control_values, args, style_vector=style_vector)
    print("DONE, please check directory wav_out")