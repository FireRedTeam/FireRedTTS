import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers import AutoTokenizer

from fireredtts.modules.gpt.gpt import Speech_LLM_GPT2
from fireredtts.modules import Token2Wav_v2, MelSpectrogramExtractorV2

# from fireredtts.modules.codec import SemanticTokenizer
from fireredtts.modules.text_normalizer.normalize_redtn import TextNormalizer
from fireredtts.utils.utils import load_audio, length_to_mask

import time


class FireRedTTS:
    def __init__(
        self, model_config_path, speaker_config_path, pretrained_path, device="cuda"
    ):
        """_summary_

        Args:
            model_config (_type_): 模型的config
            speaker_config (_type_): 说话人配置的config
            pretrained_path (_type_): 预训练模型和一些提前抽取好的特征的地址
            device (str, optional): _description_. Defaults to "cuda".
        """
        self.device = device
        self.config = json.load(open(model_config_path))
        self.pretrained_path = pretrained_path
        self.tokenizer_path = os.path.join(pretrained_path, "tokenizer")
        self.gpt_path = os.path.join(pretrained_path, "fireredtts_gpt.pt")
        self.token2wav_path = os.path.join(
            pretrained_path, "fireredtts_token2wav_v2.pt"
        )
        self.hubert_path = os.path.join(pretrained_path, "hubert.pt")
        self.codec_path = os.path.join(pretrained_path, "fireredtts_codec.bin")

        assert os.path.exists(self.tokenizer_path)
        assert os.path.exists(self.token2wav_path)
        assert os.path.exists(self.gpt_path)
        assert os.path.exists(self.hubert_path)
        assert os.path.exists(self.codec_path)

        # tokenizer; text_normalizer; speaker extractor;
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.text_normalizer = TextNormalizer()
        # self.semantic_tokenizer = SemanticTokenizer.from_pretrained(
        #     self.config['semantic_tokenizer'], self.hubert_path, self.codec_path
        # ).to(device)
        self.mel_extractor = MelSpectrogramExtractorV2()

        # load gpt model
        self.EOS_TOKEN = self.config["gpt"]["stop_audio_token"]
        self.gpt = Speech_LLM_GPT2(
            start_text_token=self.config["gpt"]["start_text_token"],
            stop_text_token=self.config["gpt"]["stop_text_token"],
            num_text_tokens=self.config["gpt"]["num_text_tokens"],
            start_audio_token=self.config["gpt"]["start_audio_token"],
            stop_audio_token=self.config["gpt"]["stop_audio_token"],
            num_audio_tokens=self.config["gpt"]["num_audio_tokens"],
            llm_hidden_size=self.config["gpt"]["n_embd"],
            llm_intermediate_size=self.config["gpt"]["n_inner"],
            llm_num_layers=self.config["gpt"]["n_layer"],
            llm_num_heads=self.config["gpt"]["n_head"],
            llm_max_audio_seq_len=self.config["gpt"]["llm_max_audio_seq_len"],
            llm_max_text_seq_len=self.config["gpt"]["llm_max_text_seq_len"],
            llm_max_prompt_len=self.config["gpt"]["llm_max_prompt_len"],
            code_stride_len=self.config["gpt"]["code_stride_len"],
        )

        sd = torch.load(self.gpt_path, map_location=device)["model"]
        self.gpt.load_state_dict(sd, strict=False)
        self.gpt = self.gpt.to(device=device)
        self.gpt.eval()
        self.gpt.init_gpt_for_inference(kv_cache=True)

        # self.gpt.half()

        # load token2wav model
        self.token2wav = Token2Wav_v2.from_pretrained(self.config, self.token2wav_path)
        self.token2wav = self.token2wav.to(device)

        # parse speaker config
        self.speaker_features = self.parse_speaker_config(
            speaker_config_path=speaker_config_path
        )

        # print("---self.speaker_features:\n", self.speaker_features)

    def parse_speaker_config(self, speaker_config_path):
        """_summary_

        Args:
            speaker_config (_type_): speaker config
        """
        print("#### Parse Speaker Configs...#####")
        parsed_features = {}
        speaker_config = json.load(open(speaker_config_path))
        print("---speaker_config:\n", speaker_config)
        for speaker in list(speaker_config.keys()):
            temp_dict = {}
            # load spk
            token_path = os.path.join(
                self.pretrained_path, speaker_config[speaker]["token"]
            )
            spk_path = os.path.join(
                self.pretrained_path, speaker_config[speaker]["spk"]
            )
            mel_path = os.path.join(
                self.pretrained_path, speaker_config[speaker]["mel"]
            )
            # print("---spk_path:", spk_path)

            assert os.path.exists(token_path)
            assert os.path.exists(spk_path)
            assert os.path.exists(mel_path)

            token = torch.load(token_path).to(self.device)
            spk = torch.load(spk_path).view(1, 1, -1).to(self.device)
            mel = torch.load(mel_path).to(self.device)

            # print("---spk:", spk, spk.shape)
            # print("---mel:", mel, mel.shape)

            temp_dict["token"] = token
            temp_dict["spk"] = spk
            temp_dict["mel"] = mel
            temp_dict["offset"] = int(speaker_config[speaker]["offset"])

            # print("---temp_dict:\n", temp_dict)

            parsed_features[speaker] = temp_dict
        return parsed_features

    def split_and_normalize(self, text, lang="auto"):
        text_lang_pairs = self.text_normalizer.tn(text=text)
        lines = [a for (a, b) in text_lang_pairs]
        # additional rules
        lines = [line.replace("<sbreak>", "") for line in lines]
        lines = [line.replace("?。", "。") for line in lines]
        return lines

    def merge_text_tokens(self, text_tokens):
        merged_tokens = []
        temp_token = []
        for i in range(len(text_tokens)):
            if i == len(text_tokens) - 1:
                temp_token += text_tokens[i]
                if len(temp_token) <= 8:
                    if len(merged_tokens) >= 1:
                        merged_tokens[-1] += temp_token
                    else:
                        merged_tokens.append(temp_token)
                else:
                    merged_tokens.append(temp_token)
                break
            if len(temp_token) < 25:
                temp_token += text_tokens[i]
            else:
                merged_tokens.append(temp_token)
                temp_token = []
                temp_token += text_tokens[i]

        # print("---merged_tokens:", merged_tokens, len(merged_tokens))
        # print("\n\n")

        return merged_tokens

    def do_gpt_inference(self, spk_gpt, text_tokens):
        """_summary_

        Args:
            spk_gpt (_type_): speaker embeddidng in gpt
            text_tokens (_type_): text tokens

        Returns:
            _type_: _description_
        """
        text_token = text_tokens[0]
        text_token = torch.IntTensor(text_token).unsqueeze(0).to(self.device)  # [1,t]
        assert text_token.shape[-1] < 400

        with torch.no_grad():
            gpt_codes = self.gpt.generate(
                cond_latents=spk_gpt,
                text_inputs=text_token,
                input_tokens=None,
                do_sample=True,
                top_p=0.85,
                top_k=30,
                temperature=0.75,
                num_return_sequences=5,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=2.0,
                output_attentions=False,
            )

            print("---gpt_codes:\n", gpt_codes)

            seqs = []
            for seq in gpt_codes:
                index = (seq == self.EOS_TOKEN).nonzero(as_tuple=True)[0][0]
                seq = seq[:index]
                seqs.append(seq)

            # 根据长度进行排序
            sorted_seqs = sorted(seqs, key=lambda i: len(i), reverse=False)
            sorted_len = [len(l) for l in sorted_seqs]
            print("---sorted_len:", sorted_len)
            gpt_codes = sorted_seqs[2].unsqueeze(0)
            # gpt_codes = gpt_codes[:, :-1]  # 丢掉eos token

        return gpt_codes

    def do_gpt_inference_batch(self, spk_gpt, text_tokens):
        """_summary_

        Args:
            spk_gpt (_type_): _description_
            text_tokens (_type_): _description_
        """
        # step1.构造text batch
        text_tokens = [
            torch.tensor(text_tokens[i], dtype=torch.int32)
            for i in range(len(text_tokens))
        ]

        # print("---text_tokens:", text_tokens)

        text_lengths = torch.tensor([i.size(0) for i in text_tokens], dtype=torch.int32)

        # print("---text_lengths:\n", text_lengths)

        text_tokens = pad_sequence(
            sequences=text_tokens,
            batch_first=True,
            padding_value=self.config["gpt"]["stop_text_token"],
        ).to(self.device)

        # print("---text_tokens:", text_tokens, text_tokens.shape)

        # spk_gpt repeat batch times
        spk_gpt = torch.repeat_interleave(
            input=spk_gpt, repeats=text_tokens.shape[0], dim=0
        )
        # print("---spk_gpt:\n", spk_gpt, spk_gpt.shape)

        seqs = []
        with torch.no_grad():
            gpt_codes = self.gpt.generate_batch(
                cond_latents=spk_gpt,
                text_inputs=text_tokens,
                text_lengths=text_lengths,
                input_tokens=None,
                do_sample=True,
                top_p=0.85,
                top_k=20,
                temperature=0.75,
                num_return_sequences=1,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=2.0,
                output_attentions=False,
            )

            for seq in gpt_codes:
                index = (seq == self.EOS_TOKEN).nonzero(as_tuple=True)[0][0]
                seq = seq[:index]
                seqs.append(seq)

        return seqs

    def do_gpt_inference_short(self, spk_gpt, text_tokens, prompt_lang, trunk_offset):
        """_summary_

        Args:
            spk_gpt (_type_): _description_
            text_tokens (_type_): _description_
            trunk_offset (_type_): _description_
        """
        if prompt_lang == "zh":
            # head_sentence = "这里是小红书服务。<|pad|>"
            head_sentence = "你好呀，我是你最可爱的语音助手。<|pad|>"
        elif prompt_lang == "en":
            head_sentence = (
                "welcome to this beatiful and most fantistic brand new world.<|pad|>"
            )

        # head_sentence = "if you are worried that the video you posted on world will disappear soon.<|pad|>"

        head_text_token = self.text_tokenizer.encode(text=head_sentence)

        text_token = text_tokens[0]

        text_token = head_text_token + text_token
        print("---decode:", [self.text_tokenizer.decode([c]) for c in text_token])

        text_token = torch.IntTensor(text_token).unsqueeze(0).to(self.device)
        assert text_token.shape[-1] < 400
        with torch.no_grad():
            gpt_codes = self.gpt.generate(
                cond_latents=spk_gpt,
                text_inputs=text_token,
                input_tokens=None,
                do_sample=True,
                top_p=0.85,
                top_k=20,
                temperature=0.75,
                num_return_sequences=5,
                num_beams=1,
                length_penalty=1.0,
                repetition_penalty=2.0,
                output_attentions=False,
            )

            seqs = []
            for seq in gpt_codes:
                index = (seq == self.EOS_TOKEN).nonzero(as_tuple=True)[0][0]
                seq = seq[:index]
                seqs.append(seq)

            # 根据长度进行排序
            sorted_seqs = sorted(seqs, key=lambda i: len(i), reverse=False)
            sorted_len = [len(l) for l in sorted_seqs]
            # print("---sorted_len:", sorted_len)
            gpt_codes = sorted_seqs[2].unsqueeze(0)
            # print("---gpt_codes:\n", gpt_codes)

        return gpt_codes

    def spon_label_replace(self, text):
        # label 替换
        text = text.replace("<stress>", "<|im_start|>")
        text = text.replace("<prolong>", "<|im_end|>")
        text = text.replace("<pause>", "<|semantic|>，")
        text = text.replace("<br>", "<|mel|>")
        text = text.replace("<laugh>", "<|reserve_0|>")
        return text

    # def extract_features(self, prompt_wav_path: str):
    #     audio, audio_sr = torchaudio.load(prompt_wav_path)
    #     audio = audio[:1]
    #     with torch.no_grad():
    #         token, spk = self.semantic_tokenizer(audio, audio_sr)
    #     spk = spk.view(1, 1, -1)
    #     mel = self.mel_extractor(audio, audio_sr).transpose(1, 2)
    #     token, spk, mel = map(lambda ts: ts.to(self.device), (token, spk, mel))
    #     return token, spk, mel

    def token2audio_inference(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_spk: torch.Tensor,
        prompt_mel: torch.Tensor,
    ):
        # NOTE modified by sfy
        target_mel_length = prompt_token.shape[1] * 2
        if target_mel_length > prompt_mel.shape[1]:
            prompt_mel = F.pad(
                prompt_mel,
                (0, 0, 0, target_mel_length - prompt_mel.shape[1]),
                mode="constant",
                value=-11.5,
            )
        elif target_mel_length < prompt_mel.shape[1]:
            prompt_mel = prompt_mel[:, :target_mel_length]
        # prompt_mel = F.interpolate(
        #     prompt_mel.transpose(1, 2),
        #     size=prompt_token.shape[1] * 2,
        #     mode="nearest",
        # ).transpose(1, 2)
        audio = self.token2wav.inference(
            prompt_token=prompt_token,
            prompt_xvec=prompt_spk.view(1, -1),
            prompt_mel=prompt_mel,
            token=token,
        )
        return audio

    def synthesize_online(self, speaker_id, text, lang="auto"):
        """_summary_

        Args:
            speaker_id (_type_): _description_
            text (_type_): _description_
            lang (str, optional): _description_. Defaults to "auto".
        """
        assert lang in ["zh", "en", "auto"]
        assert speaker_id in list(self.speaker_features.keys())

        # extract speaker embedding & prompt mel-spectrogram compute
        prompt_token = self.speaker_features[speaker_id]["token"]
        prompt_spk = self.speaker_features[speaker_id]["spk"]
        prompt_mel = self.speaker_features[speaker_id]["mel"]
        offset = int(self.speaker_features[speaker_id]["offset"])
        with torch.no_grad():
            spk_gpt = self.gpt.reference_embedding(prompt_spk)  # [1,1,1024]

        # text to tokens
        # text = self.spon_label_replace(text=text)
        lines = self.split_and_normalize(text=text, lang=lang)
        lines = [self.spon_label_replace(text=line) for line in lines]
        print("lines:\n", lines)
        text_tokens = [self.text_tokenizer.encode(text=line) for line in lines]

        print("---text_tokens:\n", text_tokens, len(text_tokens))

        out_wavs = []
        # normal inference
        if len(text_tokens) > 1:
            # merge text tokens
            text_tokens = self.merge_text_tokens(text_tokens=text_tokens)

            # batch inference
            gpt_batch_start_time = time.time()
            gpt_coeds_seqs = self.do_gpt_inference_batch(
                spk_gpt=spk_gpt, text_tokens=text_tokens
            )
            gpt_batch_end_time = time.time()
            gpt_batch_dur = gpt_batch_end_time - gpt_batch_start_time
            for gpt_codes in gpt_coeds_seqs:
                # print("---gpt_coeds:\n", gpt_codes, gpt_codes.shape)
                gpt_codes = gpt_codes.unsqueeze(0)

                # convert token to waveform (b=1, t)
                # voc_start_time = time.time()
                rec_wavs = self.token2audio_inference(
                    gpt_codes,
                    prompt_token,
                    prompt_spk,
                    prompt_mel,
                )
                # voc_end_time = time.time()
                # voc_dur = voc_end_time - voc_start_time
                # all_dur = voc_end_time - gpt_start_time

                # rtf compute
                # audio_dur = rec_wavs.shape[-1] / 24000
                # rtf_gpt = gpt_dur / audio_dur
                # rtf_voc = voc_dur / audio_dur
                # rtf_all = all_dur / audio_dur
                out_wavs.append(rec_wavs.detach().cpu())

            out_wav = torch.concat(out_wavs, axis=-1)

            print("---gpt_batch_dur:", gpt_batch_dur)
            print("---out_wav:", out_wav, out_wav.shape, (out_wav.shape[-1] / 24000))
            print("rtf_gpt_batch:", gpt_batch_dur / (out_wav.shape[-1] / 24000))

        else:
            text_token = text_tokens[0]
            print("---len_text_token:", len(text_token))

            # gpt inference [1,time]
            gpt_start_time = time.time()
            gpt_codes = self.do_gpt_inference(spk_gpt=spk_gpt, text_tokens=text_tokens)
            gpt_end_time = time.time()
            gpt_dur = gpt_end_time - gpt_start_time

            # print("---gpt_coeds:\n", gpt_codes, gpt_codes.shape)

            # convert token to waveform (b=1, t)
            voc_start_time = time.time()
            rec_wavs = self.token2audio_inference(
                gpt_codes,
                prompt_token,
                prompt_spk,
                prompt_mel,
            )
            voc_end_time = time.time()
            voc_dur = voc_end_time - voc_start_time

            # rtf compute
            #
            # rtf_gpt = gpt_dur / audio_dur
            # rtf_voc = voc_dur / audio_dur
            # rtf_all = all_dur / audio_dur
            out_wavs.append(rec_wavs.detach().cpu())
            out_wav = torch.concat(out_wavs, axis=-1)

            audio_dur = out_wav.shape[-1] / 24000

            print("---gpt_dur:", gpt_dur)
            print("---out_wav:", out_wav, out_wav.shape)
            print("rtf_gpt:", gpt_dur / audio_dur)
            print("rtf_decoder:", voc_dur, voc_dur / audio_dur)

        return out_wav

    # def synthesize(self, prompt_wav, text, lang="auto"):
    #     """_summary_

    #     Args:
    #         prompt_wav (_type_): _description_
    #         text (_type_): _description_
    #         lang (str, optional): _description_. Defaults to "auto".

    #     Returns:
    #         _type_: _description_
    #     """
    #     # Currently only supports Chinese and English
    #     assert lang in ["zh", "en", "auto"]
    #     assert os.path.exists(prompt_wav)

    #     # extract speaker embedding & prompt mel-spectrogram compute
    #     prompt_token, prompt_spk, prompt_mel = self.extract_features(prompt_wav)
    #     with torch.no_grad():
    #         spk_gpt = self.gpt.reference_embedding(prompt_spk)  # [1,1,1024]
    #     offset = 60

    #     prompt_lang = "en"

    #     # text to tokens
    #     lines = self.split_and_normalize(text=text, lang=lang)
    #     print("---text:\n", lines)
    #     # lines = [line.replace("sbreak", "。") for line in lines]
    #     # print("---lines:\n", lines)

    #     # tokenize text
    #     text_tokens = [self.text_tokenizer.encode(text=line) for line in lines]
    #     print("---text_tokens:\n", text_tokens, len(text_tokens))

    #     # merge text tokens
    #     if (len(text_tokens)) > 1:
    #         text_tokens = self.merge_text_tokens(text_tokens=text_tokens)

    #     out_wavs = []
    #     out_gpts = []
    #     # normal inference
    #     if len(text_tokens) > 1:
    #         # for inference
    #         gpt_start_time = time.time()
    #         for text_token in text_tokens:
    #             # gpt inference
    #             gpt_codes = self.do_gpt_inference(
    #                 spk_gpt=spk_gpt, text_tokens=[text_token]
    #             )

    #             out_gpts.append(gpt_codes)

    #             # convert token to waveform (b=1, t)
    #             # voc_start_time = time.time()
    #             rec_wavs = self.token2audio_inference(
    #                 gpt_codes,
    #                 prompt_token,
    #                 prompt_spk,
    #                 prompt_mel,
    #             )
    #             # voc_end_time = time.time()
    #             # voc_dur = voc_end_time - voc_start_time
    #             # all_dur = voc_end_time - gpt_start_time

    #             # rtf compute
    #             # audio_dur = rec_wavs.shape[-1] / 24000
    #             # rtf_gpt = gpt_dur / audio_dur
    #             # rtf_voc = voc_dur / audio_dur
    #             # rtf_all = all_dur / audio_dur
    #             out_wavs.append(rec_wavs.detach().cpu())

    #         gpt_end_time = time.time()
    #         gpt_dur = gpt_end_time - gpt_start_time

    #         out_wav = torch.concat(out_wavs, axis=-1)

    #         # print("---gpt_dur:", gpt_dur)
    #         # print("---out_wav:", out_wav, out_wav.shape, (out_wav.shape[-1] / 24000))
    #         # print("rtf_gpt:", gpt_dur / (out_wav.shape[-1] / 24000))

    #     else:
    #         text_token = text_tokens[0]
    #         # print("---len_text_token:", len(text_token))
    #         if len(text_token) <= 15:
    #             # gpt inference short [1,time]
    #             gpt_start_time = time.time()
    #             gpt_codes = self.do_gpt_inference_short(
    #                 spk_gpt=spk_gpt,
    #                 text_tokens=text_tokens,
    #                 prompt_lang=prompt_lang,
    #                 trunk_offset=60,
    #             )
    #             gpt_end_time = time.time()
    #             gpt_dur = gpt_end_time - gpt_start_time
    #         else:
    #             # gpt inference [1,time]
    #             gpt_start_time = time.time()
    #             gpt_codes = self.do_gpt_inference(
    #                 spk_gpt=spk_gpt, text_tokens=text_tokens
    #             )
    #             gpt_end_time = time.time()
    #             gpt_dur = gpt_end_time - gpt_start_time

    #         # print("---gpt_coeds:\n", gpt_codes, gpt_codes.shape)

    #         out_gpts.append(gpt_codes)

    #         # convert token to waveform (b=1, t)
    #         voc_start_time = time.time()
    #         rec_wavs = self.token2audio_inference(
    #             gpt_codes,
    #             prompt_token,
    #             prompt_spk,
    #             prompt_mel,
    #         )
    #         print("---rec_wavs:\n", rec_wavs)
    #         voc_end_time = time.time()
    #         voc_dur = voc_end_time - voc_start_time

    #         if len(text_token) <= 15:
    #             # trunk head
    #             # king-160 100
    #             # db30 80
    #             rec_wavs = rec_wavs[:, 80 * 960 :]
    #             # if gpt_codes.shape[1] >= offset + 8:
    #             #     rec_wavs = rec_wavs[:, offset * 960 :]

    #         # rtf compute
    #         #
    #         # rtf_gpt = gpt_dur / audio_dur
    #         # rtf_voc = voc_dur / audio_dur
    #         # rtf_all = all_dur / audio_dur
    #         out_wavs.append(rec_wavs.detach().cpu())
    #         out_wav = torch.concat(out_wavs, axis=-1)

    #         audio_dur = out_wav.shape[-1] / 24000

    #         print("---gpt_dur:", gpt_dur)
    #         print("---out_wav:", out_wav, out_wav.shape)
    #         print("rtf_gpt:", gpt_dur / audio_dur)
    #         print("rtf_decoder:", voc_dur, voc_dur / audio_dur)

    #     out_gpts = torch.concat(out_gpts, dim=1)

    #     return out_wav, out_gpts
