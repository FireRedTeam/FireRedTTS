import torch
import torch.nn.functional as F
from fireredtts.modules.acoustic_llm import AcousticLLM
from fireredtts.modules.acoustic_codec import AcousticCodec
from fireredtts.modules.flowmatching import FlowToken2Mel
from fireredtts.modules.bigvgan import BigVGAN, MelExtractor


class TwoStageCodec(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.acoustic_llm = AcousticLLM(**config["acoustic_llm"])
        self.acoustic_codec = AcousticCodec(**config["acoustic_codec"])
    
    def load_model(self, acoustic_llm_path, acoustic_codec_path):
        self.acoustic_llm.load_state_dict(
            torch.load(acoustic_llm_path, map_location="cpu"), strict=True
        )
        self.acoustic_codec.load_state_dict(
            torch.load(acoustic_codec_path, map_location="cpu"), strict=True
        )

    @torch.inference_mode()
    def forward(
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


"""For FlowToken2Audio, keep interface consistant with TwoStageCodec to minimize code changes.
prompt_acoustic_token alias to prompt_mel
spk_gpt alias to spk_embeddings
"""
class FlowToken2Audio(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flow = FlowToken2Mel(config['flow'])
        self.bigvgan = BigVGAN(**config['bigvgan'])
        self.mel_extractor = MelExtractor(**config['mel'])

    def load_model(self, flow_path, bigvgan_path):
        self.flow.load_state_dict(
            torch.load(flow_path, map_location="cpu"), strict=True
        )
        self.bigvgan.load_state_dict(
            torch.load(bigvgan_path, map_location="cpu")['generator'], strict=True
        )
        self.bigvgan.remove_weight_norm()

    @torch.inference_mode()
    def forward(
        self, semantic_token, prompt_semantic_token, prompt_acoustic_token, spk_gpt
    ):
        # Align prompt token & prompt_mel
        target_mel_length = prompt_semantic_token.shape[1] * 2
        if target_mel_length > prompt_acoustic_token.shape[1]:
            prompt_acoustic_token = F.pad(
                prompt_acoustic_token, (0, 0, 0, target_mel_length-prompt_acoustic_token.shape[1]),
                mode='constant', value=-11.5
            )
        elif target_mel_length < prompt_acoustic_token.shape[1]:
            prompt_acoustic_token = prompt_acoustic_token[:, :target_mel_length]
        # prompt_acoustic_token = F.interpolate(
        #     prompt_acoustic_token.transpose(1, 2),
        #     size=prompt_semantic_token.shape[1] * 2, mode='nearest'
        # ).transpose(1, 2)
        mel_pred = self.flow.inference(
            prompt_token=prompt_semantic_token,
            prompt_xvec=spk_gpt,
            prompt_feat=prompt_acoustic_token,
            token=semantic_token
        )
        wav = self.bigvgan(mel_pred.transpose(1, 2)).squeeze(1)
        return wav

    def extract(self, wavs, wav_lengths, spk):
        mel = self.mel_extractor(wavs, 24000).transpose(1, 2)
        if torch.cuda.is_available():
            mel = mel.cuda()
        return mel, spk.squeeze(0)
