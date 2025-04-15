import os
import json
import time
import torch

from transformers import AutoTokenizer

from fireredtts.utils.utils import load_audio
from fireredtts.modules.text_normalizer.utils import text_split
from fireredtts.utils.spliter import clean_text
from fireredtts.modules.text_normalizer.normalize import TextNormalizer
from fireredtts.modules.semantic_tokenizer import SemanticTokenizer
from fireredtts.modules.semantic_llm.llm_gpt2 import Speech_LLM_GPT2
from fireredtts.modules.acoustic_llm import AcousticLLM
from fireredtts.modules.acoustic_codec import AcousticCodec


class TwoStageCodec:

    def __init__(self, config):
        self.acoustic_llm = AcousticLLM(**config["acoustic_llm"])
        self.acoustic_codec = AcousticCodec(**config["acoustic_codec"])

    def __call__(
        self, semantic_token, prompt_semantic_token, prompt_acoustic_token, spk_gpt
    ):
        # print('Before: ', semantic_token.shape)
        token_pred = torch.cat((prompt_semantic_token, semantic_token), dim=1)

        # Fine LLM inference
        token_pred = self.acoustic_llm.inference_speech(
            speech_conditioning_latent=spk_gpt,
            text_inputs=token_pred,
            num_return_sequences=1,
            input_tokens=prompt_acoustic_token,
        )[0]

        if isinstance(token_pred, (tuple, list)):
            token_pred = [x.unsqueeze(0) for x in token_pred]
        else:
            token_pred = token_pred.unsqueeze(0)

        acoustic_outputs = self.acoustic_codec.reconstruct_wav(token=token_pred)
        wav = acoustic_outputs["wav_pred"].squeeze(1)

        return wav

    def extract(self, wavs, wav_lengths, spk):
        if torch.cuda.is_available():
            wavs = wavs.cuda()
        cond_tok = self.acoustic_codec.extract_speech_tokens(wavs, wav_lengths)[
            "token"
        ][0]
        spk_gpt = self.acoustic_llm.get_conditioning(spk)
        return cond_tok, spk_gpt


class FireRedTTS:
    def __init__(self, config_path, pretrained_path, device="cuda"):
        self.device = device
        self.config = json.load(open(config_path))
        self.EOS_TOKEN = self.config["semantic_llm"]["EOS_TOKEN"]

        # pretrained models
        self.tokenizer_path = os.path.join(pretrained_path, "tokenizer")
        self.speech_tokenizer_path = os.path.join(pretrained_path, "speech_tokenizer")
        self.semantic_llm_path = os.path.join(pretrained_path, "semantic_llm.pt")
        self.acoustic_llm_path = os.path.join(pretrained_path, "acoustic_llm.bin")
        self.acoustic_codec_path = os.path.join(pretrained_path, "acoustic_codec.bin")

        assert os.path.exists(self.tokenizer_path)
        assert os.path.exists(self.speech_tokenizer_path)
        assert os.path.exists(self.semantic_llm_path)
        assert os.path.exists(self.acoustic_llm_path)
        assert os.path.exists(self.acoustic_codec_path)

        # text normalizer
        self.text_normalizer = TextNormalizer()
        # text tokenizer
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # semantic llm
        self.semantic_llm = Speech_LLM_GPT2(
            start_text_token=self.config["semantic_llm"]["start_text_token"],
            stop_text_token=self.config["semantic_llm"]["stop_text_token"],
            num_text_tokens=self.config["semantic_llm"]["num_text_tokens"],
            start_audio_token=self.config["semantic_llm"]["start_audio_token"],
            stop_audio_token=self.config["semantic_llm"]["stop_audio_token"],
            num_audio_tokens=self.config["semantic_llm"]["num_audio_tokens"],
            llm_hidden_size=self.config["semantic_llm"]["llm_hidden_size"],
            llm_intermediate_size=self.config["semantic_llm"]["llm_intermediate_size"],
            llm_num_layers=self.config["semantic_llm"]["llm_num_layers"],
            llm_num_heads=self.config["semantic_llm"]["llm_num_heads"],
            llm_max_audio_seq_len=self.config["semantic_llm"]["llm_max_audio_seq_len"],
            llm_max_text_seq_len=self.config["semantic_llm"]["llm_max_text_seq_len"],
            llm_max_prompt_len=self.config["semantic_llm"]["llm_max_prompt_len"],
            code_stride_len=self.config["semantic_llm"]["code_stride_len"],
        )

        sd = torch.load(self.semantic_llm_path, map_location=device)["model"]
        self.semantic_llm.load_state_dict(sd, strict=True)
        self.semantic_llm = self.semantic_llm.to(device=device)
        self.semantic_llm.eval()
        self.semantic_llm.init_gpt_for_inference(kv_cache=True)

        # Speech tokenizer
        self.speech_tokenizer = SemanticTokenizer(
            config=self.config["semantic_tokenizer"], path=self.speech_tokenizer_path
        )

        # Acoustic decoder
        self.acoustic_decoder = TwoStageCodec(self.config)

        self.acoustic_decoder.acoustic_llm.load_state_dict(
            torch.load(self.acoustic_llm_path, map_location="cpu"), strict=True
        )
        self.acoustic_decoder.acoustic_llm = self.acoustic_decoder.acoustic_llm.to(
            device=device
        )
        self.acoustic_decoder.acoustic_llm.eval()

        self.acoustic_decoder.acoustic_codec.load_state_dict(
            torch.load(self.acoustic_codec_path, map_location="cpu"), strict=True
        )
        self.acoustic_decoder.acoustic_codec = self.acoustic_decoder.acoustic_codec.to(
            device=device
        )
        self.acoustic_decoder.acoustic_codec.eval()

    def extract_spk_embeddings(self, prompt_wav):
        audio, lsr, audio_resampled = load_audio(
            audiopath=prompt_wav,
            sampling_rate=16000,
        )

        audio_resampled = audio_resampled.to(self.device)
        audio_len = torch.tensor(
            data=[audio_resampled.shape[1]], dtype=torch.long, requires_grad=False
        )

        # spk_embeddings：[1, 512]
        prompt_tokens, token_lengths, spk_embeddings = self.speech_tokenizer(
            audio_resampled, audio_len
        )

        prompt_acoustic_tokens, acoustic_llm_spk = self.acoustic_decoder.extract(
            audio_resampled, audio_len, spk_embeddings.unsqueeze(0)
        )

        return prompt_tokens, spk_embeddings, prompt_acoustic_tokens, acoustic_llm_spk

    def synthesize_base(
        self,
        prompt_semantic_tokens,
        prompt_acoustic_tokens,
        spk_semantic_llm,
        spk_acoustic_llm,
        prompt_text,
        text,
        lang="auto",
    ):
        """_summary_

        Args:
            prompt_wav (_type_): _description_
            prompt_text (_type_): _description_
            text (_type_): _description_
            lang (str, optional): _description_. Defaults to "auto".

        Returns:
            _type_: _description_
        """
        if lang == "en":
            text = prompt_text + " " + text
        else:
            text = prompt_text + text

        # Pre-process prompt tokens
        # text to tokens
        text_tokens = self.text_tokenizer.encode(
            text=text,
            add_special_tokens=False,
            max_length=10**6,
            truncation=False,
        )
        # print("---decode", [self.text_tokenizer.decode([c]) for c in text_tokens])

        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0).to(self.device)

        assert text_tokens.shape[-1] < 200
        with torch.no_grad():
            gpt_codes = self.semantic_llm.generate_ic(
                cond_latents=spk_semantic_llm,
                text_inputs=text_tokens,
                prompt_tokens=prompt_semantic_tokens[:, :-3],
                do_sample=True,
                top_p=0.85,
                top_k=30,
                temperature=0.75,
                num_return_sequences=7,
                num_beams=1,
                length_penalty=2.0,
                repetition_penalty=5.0,
                output_attentions=False,
            )

        seqs = []
        for seq in gpt_codes:
            index = (seq == self.EOS_TOKEN).nonzero(as_tuple=True)[0][0]
            seq = seq[:index]
            seqs.append(seq)

        sorted_seqs = sorted(seqs, key=lambda i: len(i), reverse=False)
        sorted_len = [len(l) for l in sorted_seqs]

        gpt_codes = sorted_seqs[2].unsqueeze(0)

        # Acoustic decoder
        rec_wavs = self.acoustic_decoder(
            gpt_codes, prompt_semantic_tokens, prompt_acoustic_tokens, spk_acoustic_llm
        )

        rec_wavs = rec_wavs.detach().cpu()
        return rec_wavs

    @torch.no_grad
    def synthesize(self, prompt_wav, prompt_text, text, lang="auto", use_tn=False):
        """audio synthesize

        Args:
            prompt_wav (_type_): _description_
            prompt_text (_type_): _description_
            text (_type_): _description_
            lang (str, optional): _description_. Defaults to "auto".

        Returns:
            _type_: _description_
        """
        assert lang in ["zh", "en", "auto"]
        assert os.path.exists(prompt_wav)

        (
            prompt_semantic_tokens,
            spk_embeddings,
            prompt_acoustic_tokens,
            spk_acoustic_llm,
        ) = self.extract_spk_embeddings(prompt_wav=prompt_wav)

        spk_embeddings = spk_embeddings.unsqueeze(0)
        spk_semantic_llm = self.semantic_llm.reference_embedding(spk_embeddings)

        # clean text
        prompt_text = clean_text(prompt_text)
        text = clean_text(text=text)

        if use_tn:
            substrings = text_split(text=text)

            out_wavs = []
            try:
                for sub in substrings:

                    sub_tn, res_lang = self.text_normalizer.tn(text=sub)[0]

                    chunk = self.synthesize_base(
                        prompt_semantic_tokens=prompt_semantic_tokens,
                        prompt_acoustic_tokens=prompt_acoustic_tokens,
                        spk_semantic_llm=spk_semantic_llm,
                        spk_acoustic_llm=spk_acoustic_llm,
                        prompt_text=prompt_text,
                        text=sub,
                        lang=res_lang,
                    )

                    out_wavs.append(chunk)
                out_wav = torch.concat(out_wavs, axis=-1)
                return out_wav
            except:
                return None
        else:
            out_wavs = []
            try:
                res_lang = self.text_normalizer.tn(text=text)[1]

                chunk = self.synthesize_base(
                    prompt_semantic_tokens=prompt_semantic_tokens,
                    prompt_acoustic_tokens=prompt_acoustic_tokens,
                    spk_semantic_llm=spk_semantic_llm,
                    spk_acoustic_llm=spk_acoustic_llm,
                    prompt_text=prompt_text,
                    text=text,
                    lang=res_lang,
                )

                out_wavs.append(chunk)
                out_wav = torch.concat(out_wavs, axis=-1)
                return out_wav
            except:
                return None
