import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import functools
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from fireredtts.utils.utils import length_to_mask


# GPT2 NROMAL INFERENCE MODE
class GPT2InferenceModel(GPT2PreTrainedModel):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear, kv_cache):
        super().__init__(config)
        self.transformer = gpt
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
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
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            gen_inputs = input_ids[:, prefix_len:]
            gen_emb = self.embeddings(gen_inputs)
            gen_emb = gen_emb + self.pos_embedding(gen_emb)
            # print("---gen_emb:\n", gen_emb, gen_emb.shape)
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
                )
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            emb = torch.cat([prefix_emb, gen_emb], dim=1)
            # print("---initial_emb:\n", emb, emb.shape)
        else:
            emb = self.embeddings(input_ids)

            # print("---dynamic_pos:", attention_mask.shape[1] - (prefix_len + 1))

            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - (prefix_len + 1), attention_mask.device
            )
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


# GPT2 INDEX-CONTEXT INFERENCE MODE
class GPT2ICInferenceModel(GPT2PreTrainedModel):
    """Override GPT2LMHeadModel to allow for prefix conditioning."""

    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear, kv_cache):
        super().__init__(config)
        self.transformer = gpt
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

    def store_prefix_emb(self, prefix_emb):
        self.cached_prefix_emb = prefix_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
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
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # assert len(past_key_values) + len(input_ids) == attention_mask.shape[1]

        # Create embedding
        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            # gen_inputs = input_ids[:, prefix_len:]
            # gen_emb = self.embeddings(gen_inputs)
            # gen_emb = gen_emb + self.pos_embedding(gen_emb)
            gen_emb = self.cached_prefix_emb
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
                )
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            # emb = torch.cat([prefix_emb, gen_emb], dim=1)
            emb = gen_emb
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - (prefix_len + 1), attention_mask.device
            )
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        # nn.Embedding
        self.emb = torch.nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(
    layers,
    model_dim,
    heads,
    max_mel_seq_len,
    max_text_seq_len,
    max_prompt_len,
    checkpointing,
):
    """
    GPT-2 implemented by the HuggingFace library.
    """

    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_ctx=max_mel_seq_len + max_text_seq_len + max_prompt_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte

    mel_pos_emb = (
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    text_pos_emb = (
        LearnedPositionEmbeddings(max_text_seq_len, model_dim)
        if max_mel_seq_len != -1
        else functools.partial(null_position_embeddings, dim=model_dim)
    )
    # gpt = torch.compile(gpt, mode="reduce-overhead", fullgraph=True)
    return gpt, mel_pos_emb, text_pos_emb, None, None


class Speech_LLM_GPT2(nn.Module):
    def __init__(
        self,
        start_text_token,
        stop_text_token,
        num_text_tokens,
        start_audio_token,
        stop_audio_token,
        num_audio_tokens,
        llm_hidden_size,
        llm_intermediate_size,
        llm_num_layers,
        llm_num_heads,
        llm_max_audio_seq_len,
        llm_max_text_seq_len,
        llm_max_prompt_len,
        code_stride_len=640,
        max_conditioning_inputs=1,
        label_smoothing=0.0,
        checkpointing=False,
    ):
        """
        Args:

        """
        super().__init__()

        self.label_smoothing = label_smoothing
        # text token config
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.num_text_tokens = num_text_tokens

        # audio token config
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token
        self.num_audio_tokens = num_audio_tokens

        # prompts token config
        self.start_prompt_token = start_audio_token
        self.stop_prompt_token = stop_audio_token

        # other config
        self.max_conditioning_inputs = max_conditioning_inputs

        # length configs
        self.max_text_len = llm_max_text_seq_len + 2  # add <bos> <eos>
        self.max_prompt_len = llm_max_prompt_len
        self.max_audio_len = llm_max_audio_seq_len + 2 + self.max_conditioning_inputs
        self.max_gen_audio_tokens = (
            llm_max_audio_seq_len - self.max_conditioning_inputs - 2
        )
        self.code_stride_len = code_stride_len

        # model config
        self.llm_hidden_size = llm_hidden_size
        self.llm_intermediate_size = llm_intermediate_size
        self.llm_num_layers = llm_num_layers
        self.llm_num_heads = llm_num_heads

        # text embedding and audio embeddings
        self.text_embedding = nn.Embedding(self.num_text_tokens, self.llm_hidden_size)
        self.audio_embedding = nn.Embedding(self.num_audio_tokens, self.llm_hidden_size)

        # low-level llm model
        self.gpt2, self.audio_pos_embedding, self.text_pos_embedding, _, _ = (
            build_hf_gpt_transformer(
                layers=self.llm_num_layers,
                model_dim=self.llm_hidden_size,
                heads=self.llm_num_heads,
                max_mel_seq_len=self.max_audio_len,
                max_text_seq_len=self.max_text_len,
                max_prompt_len=self.max_prompt_len,
                checkpointing=checkpointing,
            )
        )

        # text and audio linear
        self.final_norm = nn.LayerNorm(self.llm_hidden_size)
        self.text_head = nn.Linear(self.llm_hidden_size, self.num_text_tokens)
        self.audio_head = nn.Linear(self.llm_hidden_size, self.num_audio_tokens)

        # speaker特征变换
        self.reference_embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, self.llm_hidden_size),
        )

    def init_gpt_for_inference(self, kv_cache=True, use_deepspeed=False):
        """_summary_

        Args:
            kv_cache (bool, optional): _description_. Defaults to True.
            use_deepspeed (bool, optional): _description_. Defaults to False.
        """
        seq_length = self.max_audio_len + self.max_text_len + self.max_prompt_len + 1

        gpt_config = GPT2Config(
            vocab_size=self.num_audio_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.llm_hidden_size,
            n_layer=self.llm_num_layers,
            n_head=self.llm_num_heads,
            gradient_checkpointing=False,
            use_cache=True,
        )

        # normal inference model
        self.gpt_inference = GPT2InferenceModel(
            config=gpt_config,
            gpt=self.gpt2,
            pos_emb=self.audio_pos_embedding,
            embeddings=self.audio_embedding,
            norm=self.final_norm,
            linear=self.audio_head,
            kv_cache=kv_cache,
        )

        # in-context inference model
        self.gpt_inference_ic = GPT2ICInferenceModel(
            config=gpt_config,
            gpt=self.gpt2,
            pos_emb=self.audio_pos_embedding,
            embeddings=self.audio_embedding,
            norm=self.final_norm,
            linear=self.audio_head,
            kv_cache=kv_cache,
        )

        self.gpt2.wte = self.audio_embedding

    def add_bos_eos(self, input, start_token=None, stop_token=None):
        if start_token is not None:
            input = F.pad(input, (1, 0), value=start_token)
        if stop_token is not None:
            input = F.pad(input, (0, 1), value=stop_token)
        return input

    # --------------------------- normal inference ---------------------------
    def inference(self, cond_latents, text_inputs, **hf_generate_kwargs):
        self.compute_embeddings(cond_latents, text_inputs)
        return self.generate(cond_latents, text_inputs, **hf_generate_kwargs)

    def compute_embeddings(self, cond_latents, text_inputs):

        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        # print("---emb:\n", emb, emb.shape)

        emb = torch.cat([cond_latents, emb], dim=1)
        # print("---emb after concat spk:\n", emb, emb.shape)
        self.gpt_inference.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_audio_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_audio_token
        return gpt_inputs

    def generate(self, cond_latents, text_inputs, **hf_generate_kwargs):
        gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)
        gen = self.gpt_inference.generate(
            gpt_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_audio_tokens + gpt_inputs.shape[-1],
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1] :], gen
        return gen[:, gpt_inputs.shape[1] :]

    def compute_embeddings_batch(self, cond_latents, text_inputs, text_lengths):
        """_summary_

        Args:
            cond_latents (_type_): _description_
            text_inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        text_inputs = self.add_bos_eos(
            input=text_inputs,
            start_token=self.start_text_token,
            stop_token=self.stop_text_token,
        )
        # print("---text_inputs after add bos eos:", text_inputs, text_inputs.shape)

        # print("---text_lengths:", text_lengths)

        text_lengths = text_lengths + 2
        # print("---text_lengths after add bos eos:", text_lengths)

        # construct attn_mask
        attention_mask = length_to_mask(length=text_lengths).to(text_inputs.device)
        # print("---attention_mask:\n", attention_mask, attention_mask.shape)

        # get text embeddings
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        emb = torch.cat([cond_latents, emb], dim=1)
        self.gpt_inference.store_prefix_emb(emb)
        # print("---emb:\n", emb, emb.shape)

        # 因为添加了一个spk，所以更新mask
        attn_mask_cond = torch.ones(
            attention_mask.shape[0], 1, dtype=torch.bool, device=attention_mask.device
        )

        attn_mask_audio = torch.ones(
            attention_mask.shape[0], 1, dtype=torch.bool, device=attention_mask.device
        )

        attention_mask = torch.cat(
            [attn_mask_cond, attention_mask, attn_mask_audio], dim=1
        )

        # print("---attention_mask:\n", attention_mask, attention_mask.shape)

        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_audio_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_audio_token

        return gpt_inputs, attention_mask

    def generate_batch(
        self, cond_latents, text_inputs, text_lengths, **hf_generate_kwargs
    ):
        """_summary_

        Args:
            cond_latents (_type_): _description_
            text_inputs (_type_): _description_

        Returns:
            _type_: _description_
        """
        # print("---cond_latents:\n", cond_latents, cond_latents.shape)
        # print("---text_inputs:\n", text_inputs)

        gpt_inputs, attention_mask = self.compute_embeddings_batch(
            cond_latents=cond_latents,
            text_inputs=text_inputs,
            text_lengths=text_lengths,
        )
        gen = self.gpt_inference.generate(
            gpt_inputs,
            attention_mask=attention_mask,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_audio_tokens + gpt_inputs.shape[-1],
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1] :], gen

        # print("---gen:\n", gen)
        # print("---gen[:, gpt_inputs.shape[1] :]:\n", gen[:, : gpt_inputs.shape[1]])
        # print(
        #     "---gen[:, gpt_inputs.shape[1]+1 :]:\n", gen[:, : gpt_inputs.shape[1] + 1]
        # )

        return gen[:, gpt_inputs.shape[1] :]

    # --------------------------- normal inference --------------------------

    # --------------------------- IC inference ---------------------------
    def compute_embeddings_ic(self, cond_latents, text_inputs, prompt_tokens):
        """_summary_

        Args:
            cond_latents (_type_): speaker embedding
            text_inputs (_type_): text tokens
            prompt_tokens (_type_): prompts_tokens

        Returns:
            _type_: _description_
        """

        print("---text_inputs:\n", text_inputs, text_inputs.shape)

        # text embeddings
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        print("---text_inputs after pad:\n", text_inputs, text_inputs.shape)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs
        )
        print("---text_emb:\n", text_emb, text_emb.shape)

        # prompt_tokens
        print("---prompt_tokens:", prompt_tokens, prompt_tokens.shape)
        prompt_tokens = F.pad(prompt_tokens, (1, 0), value=self.start_audio_token)
        audio_emb = self.audio_embedding(prompt_tokens) + self.audio_pos_embedding(
            prompt_tokens
        )

        print("---prompt_tokens after pad:", prompt_tokens, prompt_tokens.shape)
        print("---audio_emb:\n", audio_emb, audio_emb.shape)

        emb = torch.cat([cond_latents, text_emb, audio_emb], dim=1)

        print("---emb:\n", emb.shape)

        self.gpt_inference_ic.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (emb.shape[0], emb.shape[1]),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        return gpt_inputs

    def generate_ic(
        self, cond_latents, text_inputs, prompt_tokens, **hf_generate_kwargs
    ):
        """_summary_

        Args:
            cond_latents (_type_): _description_
            text_inputs (_type_): _description_
            prompt_tokens (_type_): _description_

        Returns:
            _type_: _description_
        """
        gpt_inputs = self.compute_embeddings_ic(
            cond_latents, text_inputs, prompt_tokens
        )
        gen = self.gpt_inference_ic.generate(
            gpt_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_audio_tokens + gpt_inputs.shape[-1],
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1] :], gen

        print("---gen:\n", gen)

        return gen[:, gpt_inputs.shape[1] :]

    # --------------------------- IC inference ---------------------------

    def get_generator(self, fake_inputs, **hf_generate_kwargs):
        return self.gpt_inference.generate_stream(
            fake_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + fake_inputs.shape[-1],
            do_stream=True,
            **hf_generate_kwargs,
        )
