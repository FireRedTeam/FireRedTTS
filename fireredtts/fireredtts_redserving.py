import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from transformers import AutoTokenizer

from fireredtts.modules.gpt.gpt import Speech_LLM_GPT2
from fireredtts.modules import Token2Wav_v2, MelSpectrogramExtractorV2

# from fireredtts.modules.codec import SemanticTokenizer
from fireredtts.modules.text_normalizer.normalize_redtn import TextNormalizer
from fireredtts.utils.utils import load_audio, length_to_mask

from redserving_llm import XformerForCausalLM, XformerEngineConfig, SchedulerConfig

import time


class FireRedTTS_Redserving:
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
        self.gpt_path = os.path.join(pretrained_path, "fireredtts_gpt.pt")
        self.tn_path = os.path.join(pretrained_path, "tn_data")

        self.token2wav_path = os.path.join(
            pretrained_path, "fireredtts_token2wav_v2.pt"
        )
        self.hubert_path = os.path.join(pretrained_path, "hubert.pt")
        self.codec_path = os.path.join(pretrained_path, "fireredtts_codec.bin")

        self.tokenizer_path = os.path.join(pretrained_path, "tokenizer")
        self.gpt_redserving_path = os.path.join(pretrained_path, "gpt2_redserving")

        assert os.path.exists(self.token2wav_path)
        assert os.path.exists(self.gpt_path)
        assert os.path.exists(self.tn_path)
        # assert os.path.exists(self.hubert_path)
        # assert os.path.exists(self.codec_path)
        assert os.path.exists(self.tokenizer_path)
        assert os.path.exists(self.gpt_redserving_path)

        # tokenizer; text_normalizer; speaker extractor;
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.text_normalizer = TextNormalizer(data_dir=self.tn_path)
        # self.semantic_tokenizer = SemanticTokenizer.from_pretrained(
        #     self.config['semantic_tokenizer'], self.hubert_path, self.codec_path
        # ).to(device)
        # self.mel_extractor = MelSpectrogramExtractorV2()

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

        # ------------------------------ load gpt redserving
        layernorm = self.gpt.gpt_inference.final_norm.half().to(device=device)

        def custom_layernorm(name, map, stream):
            if name == "last_ln2":
                stream = torch.cuda.Stream(stream_ptr=stream)
                torch.cuda.set_stream(stream)
                # torch.cuda.synchronize()
                map["norm_out"].copy_(layernorm(map["norm_in"]))
                # print(f"time is: {datetime.datetime.now()}")
                # print(f"post_ln2 output: {map['norm_out']}")
                # print(f"222 norm output ptr: {map['norm_out'].data_ptr()}")

        self.gpt_redserving = XformerForCausalLM.from_pretrained(
            self.gpt_redserving_path,
            engine_config=XformerEngineConfig(
                tensor_para_size=1,
                scheduler_config=SchedulerConfig(
                    max_model_len=2048,
                    enable_chunked_prefill=False,
                ),
            ),
            visitor=custom_layernorm,
        )
        self.gpt_redserving.position_embedding_start_from(1)
        # ------------------------------

        # load token2wav model
        self.token2wav = Token2Wav_v2.from_pretrained(
            self.config,
            self.token2wav_path,
        )
        self.token2wav = self.token2wav.to(device)

        # parse speaker config
        self.speaker_features = self.parse_speaker_config(
            speaker_config_path=speaker_config_path
        )

        # print("---self.speaker_features:\n", self.speaker_features)
        print("---说话人列表:\n", list(self.speaker_features.keys()))

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
        lines = [line.replace("、", "，") for line in lines]
        return lines

    def merge_text_tokens(self, text_tokens):
        merged_tokens = []
        temp_token = []
        for i in range(len(text_tokens)):
            temp_token += text_tokens[i]
            if i == len(text_tokens) - 1:
                if len(temp_token) <= 9:
                    if len(merged_tokens) >= 1:
                        merged_tokens[-1] += temp_token
                    else:
                        merged_tokens.append(temp_token)
                else:
                    merged_tokens.append(temp_token)
                break

            if len(temp_token) >= 8:
                merged_tokens.append(temp_token)
                temp_token = []

        # print("---merged_tokens:", merged_tokens, len(merged_tokens))
        # print("\n\n")

        return merged_tokens

    def do_gpt_inference(self, spk_gpt, text_tokens):
        """进行gpt inference, 单句话，且长度适中

        Args:
            spk_gpt (_type_): speaker embeddidng in gpt
            text_tokens (_type_): text tokens

        Returns:
            _type_: _description_
        """
        # ------------------------------- compute embeddings ---------------------------
        #  [time_step,]==> [1,time_step]
        text_token = text_tokens[0]
        text_token = torch.IntTensor(text_token).unsqueeze(0).to(self.device)
        # assert text_token.shape[-1] < 400
        text_token = self.gpt.add_bos_eos(
            input=text_token,
            start_token=self.config["gpt"]["start_text_token"],
            stop_token=self.config["gpt"]["stop_text_token"],
        )
        emb = self.gpt.text_embedding(text_token) + self.gpt.text_pos_embedding(
            text_token
        )
        emb = torch.cat([spk_gpt, emb], dim=1)  # cat speaker info
        # print("---text_tokens:\n", text_tokens)
        # print("---emb after concat spk:\n", emb, emb.shape)

        audio_start_token = (
            torch.IntTensor([self.config["gpt"]["start_audio_token"]])
            .unsqueeze(0)
            .to("cuda")
        )
        audio_bos_emb = self.gpt.audio_embedding(
            audio_start_token
        ) + self.gpt.audio_pos_embedding.get_fixed_embedding(ind=0, dev="cuda")
        initial_embeddings = torch.cat([emb, audio_bos_emb], dim=1).half().detach()
        # print("---initial_embeddings:\n", initial_embeddings, initial_embeddings.shape)
        # -----------------------------------------------------------------------------------

        # ------------------------------------ GPT infer ------------------------------------
        inputs_em = []
        for in_em in initial_embeddings:
            inputs_em.append(in_em)

        outputs = self.gpt_redserving.generate_blocking(
            inputs_embeds=inputs_em,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.85,
            top_k=30,
            temperature=0.75,
            num_return_sequences=5,
            repetition_penalty=2.0,
        )

        # 记录长度
        out_len = []
        for output in outputs:
            out_len.append(len(output.output_ids))
        out_len = np.array(out_len)
        sorted_index = np.argsort(out_len)
        index = sorted_index[2]  # 取中间长度，尽可能保证稳定性
        print("---out_len:", out_len)
        # print("---sorted_index:", sorted_index)
        gpt_codes = (
            torch.IntTensor(list(outputs[index].output_ids))
            .to(device=self.device)
            .unsqueeze(0)[:, :-1]
        )  # 丢掉最后一个eos tokens
        # --------------------------------------------------------------------------------

        return gpt_codes

    def do_gpt_inference_short(self, spk_gpt, text_tokens):
        """进行gpt inference, 单句话，超短句，长度小于N个token

        Args:
            spk_gpt (_type_): _description_
            text_tokens (_type_): _description_
        """
        # ------------------------------- compute embeddings ---------------------------
        head_sentence = "这里是小红书服务。<|pad|>"
        head_text_token = self.text_tokenizer.encode(text=head_sentence)
        text_token = text_tokens[0]
        text_token = head_text_token + text_token
        # print("---decode:", [self.text_tokenizer.decode([c]) for c in text_token])
        text_token = torch.IntTensor(text_token).unsqueeze(0).to(self.device)
        # assert text_token.shape[-1] < 400

        text_token = self.gpt.add_bos_eos(
            input=text_token,
            start_token=self.config["gpt"]["start_text_token"],
            stop_token=self.config["gpt"]["stop_text_token"],
        )
        emb = self.gpt.text_embedding(text_token) + self.gpt.text_pos_embedding(
            text_token
        )
        emb = torch.cat([spk_gpt, emb], dim=1)  # cat speaker info
        # print("---text_tokens:\n", text_tokens)
        # print("---emb after concat spk:\n", emb, emb.shape)

        audio_start_token = (
            torch.IntTensor([self.config["gpt"]["start_audio_token"]])
            .unsqueeze(0)
            .to("cuda")
        )
        audio_bos_emb = self.gpt.audio_embedding(
            audio_start_token
        ) + self.gpt.audio_pos_embedding.get_fixed_embedding(ind=0, dev="cuda")
        initial_embeddings = torch.cat([emb, audio_bos_emb], dim=1).half().detach()
        # print("---initial_embeddings:\n", initial_embeddings, initial_embeddings.shape)

        # ------------------------------------ GPT infer ------------------------------------
        inputs_em = []
        for in_em in initial_embeddings:
            inputs_em.append(in_em)

        outputs = self.gpt_redserving.generate_blocking(
            inputs_embeds=inputs_em,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.85,
            top_k=20,
            temperature=0.75,
            num_return_sequences=5,
            repetition_penalty=2.0,
        )

        # 记录长度
        out_len = []
        for output in outputs:
            out_len.append(len(output.output_ids))
        out_len = np.array(out_len)
        sorted_index = np.argsort(out_len)
        index = sorted_index[2]  # 取中间长度，尽可能保证稳定性
        # print("---out_len:", out_len)
        # print("---sorted_index:", sorted_index)
        gpt_codes = (
            torch.IntTensor(list(outputs[index].output_ids))
            .to(device=self.device)
            .unsqueeze(0)[:, :-1]
        )  # 丢掉最后一个eos tokens
        # --------------------------------------------------------------------------------

        return gpt_codes

    def do_gpt_inference_batch(self, spk_gpt, text_tokens):
        """_summary_

        Args:
            spk_gpt (_type_): _description_
            text_tokens (_type_): _description_
        """
        # step1.构造text batch
        text_tokens = [
            torch.IntTensor(text_token).unsqueeze(0).to(self.device)
            for text_token in text_tokens
        ]

        # print("---text_tokens:\n", text_tokens)

        # step2.构造initial_embeedings
        inputs_em = []
        for text_token in text_tokens:
            text_token = self.gpt.add_bos_eos(
                input=text_token,
                start_token=self.config["gpt"]["start_text_token"],
                stop_token=self.config["gpt"]["stop_text_token"],
            )
            emb = self.gpt.text_embedding(text_token) + self.gpt.text_pos_embedding(
                text_token
            )
            emb = torch.cat([spk_gpt, emb], dim=1)  # cat speaker info
            # print("---text_tokens:\n", text_tokens)
            # print("---emb after concat spk:\n", emb, emb.shape)

            audio_start_token = (
                torch.IntTensor([self.config["gpt"]["start_audio_token"]])
                .unsqueeze(0)
                .to("cuda")
            )
            audio_bos_emb = self.gpt.audio_embedding(
                audio_start_token
            ) + self.gpt.audio_pos_embedding.get_fixed_embedding(ind=0, dev="cuda")
            initial_embeddings = torch.cat([emb, audio_bos_emb], dim=1).half().detach()
            # print(
            #     "---initial_embeddings:\n", initial_embeddings, initial_embeddings.shape
            # )
            inputs_em.append(initial_embeddings.squeeze(0))

        # ------------------------------------ GPT infer ------------------------------------
        outputs = self.gpt_redserving.generate_blocking(
            inputs_embeds=inputs_em,
            max_new_tokens=600,
            do_sample=True,
            top_p=0.85,
            top_k=20,
            temperature=0.75,
            num_return_sequences=1,
            repetition_penalty=2.0,
        )
        # print("---outputs:\n", outputs)

        seqs = []
        for output in outputs:
            gpt_codes = (
                torch.IntTensor(list(output.output_ids))
                .to(device=self.device)
                .unsqueeze(0)[:, :-1]
            )  # 丢掉最后一个eos tokens
            seqs.append(gpt_codes)

        return seqs

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
        audio = self.token2wav.inference(
            prompt_token=prompt_token,
            prompt_xvec=prompt_spk.view(1, -1),
            prompt_mel=prompt_mel,
            token=token,
        )
        return audio

    def synthesize_online(self, speaker_id, text, lang="auto"):
        """在线合成函数

        Args:
            speaker_id (_type_): 说话人名称
            text (_type_): 输入文本
            lang (str, optional): 文本语言

        返回，错误码和音频
        402	输入参数错误	输入参数有误:text为空；speaker未定义；其他参数不符合约定的问题(一般开发联调时出现，服务时不会发生)	音频生成失败，请重新尝试
        501	服务内部错误	推理过程出现问题：显存溢出其他推理和计算过程中错误	    音频生成失败，请重新尝试
        701（仅用于 capa服务）	输入文本过长	输入文本过长：仅用于capa服务，输入文本超过350字	    输入文本过长，请编辑后重新尝试

        """
        # 输入语言错误
        if lang not in ["zh", "en", "auto"]:
            return "402", None
        # 输入说话人错误
        if speaker_id not in list(self.speaker_features.keys()):
            return "402", None

        # extract speaker embedding & prompt mel-spectrogram compute
        prompt_token = self.speaker_features[speaker_id]["token"]
        prompt_spk = self.speaker_features[speaker_id]["spk"]
        prompt_mel = self.speaker_features[speaker_id]["mel"]
        offset = int(self.speaker_features[speaker_id]["offset"])
        with torch.no_grad():
            spk_gpt = self.gpt.reference_embedding(prompt_spk)  # [1,1,1024]

        # text to tokens
        lines = self.split_and_normalize(text=text, lang=lang)
        text_tokens = [self.text_tokenizer.encode(text=line) for line in lines]
        print("---lines:\n", lines)
        print("---text_tokens:\n", text_tokens, len(text_tokens))

        try:
            # inference
            out_wavs = []
            if len(text_tokens) > 1:  # batch inference
                # merge text tokens
                text_tokens = self.merge_text_tokens(text_tokens=text_tokens)
                print("---text_tokens after merged:\n", text_tokens, len(text_tokens))
                gpt_batch_start_time = time.time()
                gpt_coeds_seqs = self.do_gpt_inference_batch(
                    spk_gpt=spk_gpt, text_tokens=text_tokens
                )
                gpt_batch_end_time = time.time()
                gpt_batch_dur = gpt_batch_end_time - gpt_batch_start_time
                for gpt_codes in gpt_coeds_seqs:
                    # convert token to waveform (b=1, t)
                    rec_wavs = self.token2audio_inference(
                        gpt_codes,
                        prompt_token,
                        prompt_spk,
                        prompt_mel,
                    )
                    out_wavs.append(rec_wavs.detach().cpu())

                out_wav = torch.concat(out_wavs, axis=-1)

                print("---gpt_batch_dur:", gpt_batch_dur)
                print(
                    "---out_wav:", out_wav, out_wav.shape, (out_wav.shape[-1] / 24000)
                )
                print("rtf_gpt_batch:", gpt_batch_dur / (out_wav.shape[-1] / 24000))

            else:  # normal inference
                text_token_len = len(text_tokens[0])
                print("---len_text_token:", text_token_len)

                gpt_start_time = time.time()
                gpt_codes = self.do_gpt_inference(
                    spk_gpt=spk_gpt,
                    text_tokens=text_tokens,
                )
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

                out_wavs.append(rec_wavs.detach().cpu())
                out_wav = torch.concat(out_wavs, axis=-1)
                audio_dur = out_wav.shape[-1] / 24000

                print("---gpt_dur:", gpt_dur)
                print("---out_wav:", out_wav, out_wav.shape)
                print("---rtf_gpt:", gpt_dur / audio_dur)
                print("---rtf_decoder:", voc_dur, voc_dur / audio_dur)

            return None, out_wav
        except:
            return "501", None

