from einops import rearrange
from time import time
from torch.utils.checkpoint import checkpoint
from transformers import (
    GPT2Config,
    GPT2Model,
    GPT2PreTrainedModel,
    LogitsProcessorList,
    LogitsWarper,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerationConfig,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from typing import Any, Dict, Optional, Tuple, Union

import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadRepetitionPenaltyLogitsProcessor(LogitsWarper):

    def __init__(
        self, penalty: float = 2.0, n_heads: int = 4, n_frames: int = -1, start_index=0
    ):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty
        self.n_heads = n_heads
        self.n_frames = n_frames
        self.start_index = start_index

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        input_ids = input_ids[:, self.start_index :]
        if input_ids.size(1) == 0:
            return scores

        if self.n_frames <= 0:
            input_ids = torch.flip(input_ids, [1])[:, self.n_heads - 1 :: self.n_heads]
        else:
            input_ids = torch.flip(input_ids, [1])[
                :, self.n_heads - 1 : self.n_heads * self.n_frames : self.n_heads
            ]
        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        if self.penalty > 100:
            score = torch.full_like(score, -1e3)
        else:
            score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, input_ids, score)
        return scores


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class FixedStoppingCriteria(StoppingCriteria):

    def __init__(self, running_steps, start_index=0):
        self.running_steps = running_steps
        self.start_index = start_index

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        assert input_ids.shape[0] == 1, input_ids.shape
        if input_ids.shape[1] - self.start_index >= self.running_steps:
            return torch.tensor([True]).to(input_ids.device)
        return torch.tensor([False]).to(input_ids.device)


class DelayStoppingCriteria(StoppingCriteria):

    def __init__(self, eos_token_id, delay_steps):
        self.delay_steps = delay_steps
        self.eos_token_id = torch.tensor(eos_token_id)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> torch.BoolTensor:
        assert input_ids.shape[0] == 1, input_ids.shape

        if (input_ids == self.eos_token_id).any():
            index = (input_ids[0] == self.eos_token_id).nonzero(as_tuple=True)[0][0]
            if index + self.delay_steps < input_ids.shape[1]:
                return torch.tensor([True]).to(input_ids.device)
        return torch.tensor([False]).to(input_ids.device)


class SuppressionLogitsProcessor(LogitsWarper):

    def __init__(self, suppressed_ids=[]):
        self.suppressed_ids = suppressed_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for sid in self.suppressed_ids:
            scores[..., sid] = scores.min()
        return scores


class MHGPT2InferenceModel(GPT2PreTrainedModel):

    def __init__(
        self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=True
    ):
        super().__init__(config)
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.lm_head = nn.ModuleList([norm, linear])  # nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

        # Multi-head configuration
        self.n_heads = len(linear)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None
        self.cached_mel_parallel_emb = None

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def store_mel_parallel_emb(self, mel_emb):
        self.cached_mel_parallel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        attention_mask = None
        position_ids = None

        if input_ids.shape[1] != 1 and past_key_values is None:
            text_inputs = input_ids[:, mel_len:]
            text_emb = sum(
                [self.embeddings[i](text_inputs[:, :, i]) for i in range(self.n_heads)]
            )
            text_emb = text_emb + self.text_pos_embedding(text_emb)

            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(
                    text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                )
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb

            if self.cached_mel_parallel_emb is not None:
                text_emb = (
                    text_emb + self.cached_mel_parallel_emb[:, : text_emb.shape[1]]
                )

            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:  # KV-cache mode
            text_inputs = input_ids[:, mel_len:]
            emb = sum(
                [self.embeddings[i](text_inputs[:, -1, i]) for i in range(self.n_heads)]
            )
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                text_inputs.shape[1] - 1, emb.device
            )

            if self.cached_mel_parallel_emb is not None:
                emb = emb + self.cached_mel_parallel_emb[:, text_inputs.shape[1] - 1]

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,  # output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        past_key_values = transformer_outputs.past_key_values
        output_hidden_states = transformer_outputs.hidden_states
        output_attentions = transformer_outputs.attentions

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head[0](hidden_states)
        lm_logits = [head(lm_logits) for head in self.lm_head[1]]
        lm_logits = torch.stack(lm_logits, dim=2)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        output = CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=output_hidden_states,
            attentions=output_attentions,
        )
        return output

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len, num_streams = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
            cur_len=cur_len,
            max_length=max_length,
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits.clone()[:, -1].float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            batch_size, seq_len, num_streams = input_ids.shape
            rearrange_input_ids = rearrange(input_ids, "b l n -> (b n) l")
            next_token_logits = rearrange(next_token_logits, "b n d -> (b n) d")
            next_token_scores = logits_processor(rearrange_input_ids, next_token_logits)
            next_token_scores = rearrange(
                next_token_scores, "(b n) d -> b n d", b=batch_size
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                probs = probs.view(-1, probs.shape[-1])
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_tokens = next_tokens.view(*next_token_scores.shape[:-1])
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


class LearnedPositionEmbeddings(nn.Module):

    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(
    layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing
):
    gpt_config = GPT2Config(
        vocab_size=256,
        n_positions=max_mel_seq_len + max_text_seq_len,
        n_ctx=max_mel_seq_len + max_text_seq_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        use_cache=not checkpointing,
        scale_attn_by_inverse_layer_idx=True,
        reorder_and_upcast_attn=True,
        attn_implementation="sdpa",
    )
    gpt = GPT2Model(gpt_config)

    if checkpointing:
        gpt.gradient_checkpointing_enable()

    del gpt.wpe, gpt.wte
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    mel_pos_embs = LearnedPositionEmbeddings(max_mel_seq_len, model_dim)

    return gpt, mel_pos_embs


class AcousticLLM(nn.Module):

    def __init__(
        self,
        # Model
        n_stacks=2,
        layers=12,
        model_dim=1024,
        heads=16,
        # Text
        max_text_tokens=120,
        number_text_tokens=8194,
        start_text_token=8192,
        stop_text_token=8193,
        # Speech
        n_frames_per_step=4,
        n_heads_per_frame=1,
        max_speech_tokens=250,
        number_speech_tokens=8194,
        start_speech_token=8192,
        stop_speech_token=8193,
        # CoS Prediction
        streaming=False,
        streaming_delayed_frames=4,
        accumulative_speech_embedding=False,
        upsample_factors=2,
        # Reference embedding
        max_conditioning_inputs=1,
        speaker_embedding_pretrained=True,
        speaker_embedding_ckpt=None,
        speaker_embedding_dim=256,
        # For training
        checkpointing=True,
        loss_weights=1.0,
        # For inference
        delay_prediction=1,
        temperature=0.3,
        length_penalty=1.0,
        repetition_penalty=2.0,
        top_p=0.2,
        top_k=50,
    ):
        super().__init__()
        self.n_stacks = n_stacks
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_speech_tokens = number_speech_tokens
        self.start_speech_token = start_speech_token
        self.stop_speech_token = stop_speech_token
        self.layers = layers
        self.heads = heads

        self.streaming = streaming
        self.streaming_delayed_frames = streaming_delayed_frames
        self.accumulative_speech_embedding = accumulative_speech_embedding
        self.upsample_factors = upsample_factors

        self.n_frames_per_step = n_frames_per_step
        self.n_heads_per_frame = n_heads_per_frame
        self.number_speech_heads = n_heads_per_frame * n_frames_per_step

        self.max_speech_tokens = max_speech_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs

        self.speaker_embedding_pretrained = speaker_embedding_pretrained
        self.speaker_embedding_ckpt = speaker_embedding_ckpt
        self.speaker_embedding_dim = speaker_embedding_dim

        # For training
        self.loss_weights = loss_weights

        # For inference
        self.delay_prediction = delay_prediction
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.top_k = top_k

        # Conditional embedding
        self.reference_embedding = nn.Sequential(
            nn.Linear(speaker_embedding_dim, 256),
            nn.Tanh(),
            nn.Linear(256, model_dim),
        )

        self.text_embedding = nn.Embedding(self.number_text_tokens + 1, model_dim)
        self.text_embedding.weight.data.normal_(mean=0.0, std=0.02)

        self.mel_embedding = nn.ModuleList(
            [
                nn.Embedding(self.number_speech_tokens, model_dim)
                for _ in range(self.number_speech_heads)
            ]
        )
        for module in self.mel_embedding:
            module.weight.data.normal_(mean=0.0, std=0.02)

        # Build GPTs
        self.gpt, self.mel_pos_embedding = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_speech_tokens + 2 + self.max_conditioning_inputs,
            self.max_text_tokens + 2,
            checkpointing,
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.mel_head = nn.ModuleList(
            [
                nn.Linear(model_dim, self.number_speech_tokens)
                for _ in range(self.number_speech_heads)
            ]
        )

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=True, half=False):
        seq_length = self.max_speech_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.max_speech_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )

        self.inference_model = MHGPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        self.inference_model.eval()

    def build_aligned_inputs_and_targets(
        self, seqs, lens, start_token, stop_token, delay=0
    ):
        for i in range(seqs.shape[0]):
            seqs[i, lens[i] :] = stop_token

        if len(seqs.shape) == 2:
            inp = F.pad(
                seqs, (self.streaming_delayed_frames, 0), value=start_token
            ).type_as(seqs)
            inp = F.pad(inp, (0, 1), value=stop_token).type_as(inp)
            tar = F.pad(inp[:, 1:], (0, 1), value=stop_token).type_as(seqs)
        else:
            inp = F.pad(
                seqs, (0, 0, self.streaming_delayed_frames, 0), value=start_token
            ).type_as(seqs)
            inp = F.pad(inp, (0, 0, 0, 1), value=stop_token).type_as(inp)
            tar = F.pad(inp[:, 1:], (0, 0, 0, 1), value=stop_token).type_as(seqs)

        if delay > 0:
            pad_size = delay * (inp.shape[2] - 1)
            L = inp.shape[1] + pad_size
            inp = F.pad(inp, (0, 0, pad_size, 0), value=start_token).type_as(inp)
            inp = F.pad(inp, (0, 0, 0, pad_size), value=stop_token).type_as(inp)
            inp = torch.stack(
                [
                    inp[:, pad_size - i * delay : pad_size - i * delay + L, i]
                    for i in range(inp.shape[-1])
                ],
                dim=-1,
            )

            tar = F.pad(tar, (0, 0, pad_size, 0), value=start_token).type_as(tar)
            tar = F.pad(tar, (0, 0, 0, pad_size), value=stop_token).type_as(tar)
            tar = torch.stack(
                [
                    tar[:, pad_size - i * delay : pad_size - i * delay + L, i]
                    for i in range(tar.shape[-1])
                ],
                dim=-1,
            )

            lens += pad_size

        return inp, tar, lens + self.streaming_delayed_frames + 1

    def get_logits(
        self,
        final_norm,
        first_inputs,
        first_head,
        speech_conditioning_inputs=None,
        attention_mask=None,
        get_attns=False,
        return_latent=False,
    ):
        emb = first_inputs
        if speech_conditioning_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, emb], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            attention_mask=attention_mask,
            output_attentions=get_attns,
        )

        enc = gpt_out.last_hidden_state
        if speech_conditioning_inputs is not None:
            enc = enc[:, 1:]
        enc = final_norm(enc)

        first_logits = [head(enc).permute(0, 2, 1) for head in first_head]

        return first_logits

    @torch.cuda.amp.autocast()
    def get_conditioning(self, speech_conditioning_input):
        if hasattr(self, "reference_encoder"):
            if len(speech_conditioning_input.shape) == 2:
                speech_conditioning_input = speech_conditioning_input.unsqueeze(1)
            speech_conditioning_input = self.reference_encoder(
                speech_conditioning_input
            )
        conds = self.reference_embedding(speech_conditioning_input)
        return conds

    def inference_speech(
        self,
        speech_conditioning_latent,
        text_inputs,
        input_tokens=None,
        num_return_sequences=1,
        max_generate_length=None,
        **hf_generate_kwargs,
    ):
        if not hasattr(self, "inference_model"):
            self.post_init_gpt2_config()

        # Cond
        emb = speech_conditioning_latent
        self.inference_model.store_mel_emb(emb)

        # Text
        text = torch.repeat_interleave(text_inputs, self.upsample_factors, dim=1)
        text = F.pad(
            text,
            (0, self.streaming_delayed_frames + self.number_speech_heads - 1),
            value=self.stop_speech_token,
        )
        text_embedding = self.text_embedding(text)
        self.inference_model.store_mel_parallel_emb(text_embedding)

        fake_inputs = torch.full(
            (
                emb.shape[0],  # should be 1 for stable inference
                emb.shape[1] + 1,  # + 1 for the start_speech_token
                self.number_speech_heads,
            ),
            fill_value=self.start_speech_token,
            dtype=torch.long,
            device=text_inputs.device,
        )
        if input_tokens is None:
            inputs = fake_inputs
            prompt_index = 0
        else:
            prompt, _, _ = self.build_aligned_inputs_and_targets(
                input_tokens,
                torch.Tensor([len(input_tokens[0])]).int(),
                self.start_speech_token,
                self.stop_speech_token,
                self.delay_prediction,
            )
            prompt = prompt[:, 1 : 1 + input_tokens.shape[1]]
            inputs = torch.cat([fake_inputs, prompt], dim=1)
            prompt_index = input_tokens.shape[1]
        trunc_index = fake_inputs.shape[1]

        stop_criteria = StoppingCriteriaList(
            [FixedStoppingCriteria(text_embedding.shape[1], start_index=emb.shape[1])]
        )

        logits_processor = (
            LogitsProcessorList(
                [
                    MultiHeadRepetitionPenaltyLogitsProcessor(
                        penalty=self.repetition_penalty,
                        n_heads=self.n_heads_per_frame,
                        n_frames=-1,
                        start_index=trunc_index + prompt_index,
                    )
                ]
            )
            if self.repetition_penalty > 1.0
            else LogitsProcessorList()
        )
        logits_processor.append(
            SuppressionLogitsProcessor(suppressed_ids=[self.stop_speech_token])
        )

        max_length = (
            trunc_index + self.max_speech_tokens - 1
            if max_generate_length is None
            else trunc_index + max_generate_length
        )

        # Recommandation of temp & top_p: (0.8, 0.8), (0.5, 0.5), (0.3, 0.2), (0.2, 0.1)
        gen = self.inference_model.generate(
            inputs,
            bos_token_id=self.start_speech_token,
            pad_token_id=self.stop_speech_token,
            eos_token_id=self.stop_speech_token + 2,
            max_length=max_length,
            stopping_criteria=stop_criteria,
            logits_processor=logits_processor,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=self.temperature,
            length_penalty=self.length_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
            **hf_generate_kwargs,
        )

        seq = gen[0][trunc_index:]

        start, heads = 0, []
        for j in range(self.number_speech_heads):
            head = seq[j * self.delay_prediction :, j]
            start_indices = (head == self.start_speech_token).nonzero(as_tuple=True)[0]
            start = max(start, start_indices[-1] + 1 if len(start_indices) > 0 else 0)
            stop = (head == self.stop_speech_token).nonzero(as_tuple=True)[0]
            stop = stop[0] if len(stop) > 0 else len(head)
            heads.append(head[:stop])

        min_length = min([len(x) for x in heads])
        seq = torch.stack(
            [head[start + prompt_index : min_length] for head in heads], dim=-1
        )
        return [seq]
