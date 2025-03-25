# Modified from:
#   LlamaGen:   https://github.com/FoundationVision/LlamaGen/
#   YOCO:       https://github.com/microsoft/unilm/tree/master/YOCO

import math
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from einops import rearrange
from typing import Dict, List, Optional
from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


def batch_seq_shuffle(x, orders=None):
    assert x.ndim >= 2, "The input should contain at least two dimensions, batch and length"
    bs, seq_len = x.shape[:2]
    
    if orders is None:
        orders = torch.rand(bs, seq_len, device=x.device).argsort(dim=1)
    
    orders_expand = orders.view(*orders.shape, *(1,) * (x.ndim - orders.ndim))
    shuffled_data = torch.gather(x, 1, orders_expand.expand(*x.shape))
    
    return shuffled_data, orders


# @dataclass
class ModelArgs(PretrainedConfig):
    def __init__(
        self,
        dim: int = 4096,
        n_layer: int = 32,
        n_head: int = 32,
        multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: Optional[float] = None,
        rope_base: float = 10000,
        norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        token_dropout_p: float = 0.1,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        drop_path_rate: float = 0.0,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        model_type: str = 'c2i',
        vocab_size: int = 16384,
        cls_token_num: int = 1,
        block_size: int = 256,
    ):
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.initializer_range = initializer_range
            
        self.token_dropout_p = token_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.drop_path_rate = drop_path_rate

        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.cls_token_num = cls_token_num
        self.block_size = block_size


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.n_head = config.n_head
        self.head_dim = config.dim // config.n_head

        self.to_q = nn.Linear(config.dim, config.dim, bias=False)
        self.to_k = nn.Linear(config.dim, config.dim, bias=False)
        self.to_v = nn.Linear(config.dim, config.dim, bias=False)
        
        self.proj = nn.Linear(config.dim, config.dim, bias=False)

        self.attn_drop = config.attn_dropout_p
        self.proj_drop = nn.Dropout(config.resid_dropout_p)
        
        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None
        
    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None
        
    def update_kv_cache(self, k: torch.Tensor, v: torch.Tensor):
        if self.k_cache is None and self.v_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([self.k_cache, k], dim=-2)
            v_cache = torch.cat([self.v_cache, v], dim=-2)

        self.k_cache = k_cache
        self.v_cache = v_cache

        return k_cache, v_cache

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None
    ):

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.n_head), (q, k, v))
        
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache:
            k, v = self.update_kv_cache(k, v)
            
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            is_causal=True if self.training else False,
            dropout_p=self.attn_drop if self.training else 0
        )            
        output = rearrange(output, 'b h n d -> b n (h d)').contiguous()
        output = self.proj_drop(self.proj(output))
        return output
    
    
class CrossAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.n_head = config.n_head
        self.head_dim = config.dim // config.n_head

        self.to_q = nn.Linear(config.dim, config.dim, bias=False)
        
        self.proj = nn.Linear(config.dim, config.dim, bias=False)

        self.attn_drop = config.attn_dropout_p
        self.proj_drop = nn.Dropout(config.resid_dropout_p)
        
        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None
        
    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None
        
    def update_kv_cache(self, k: torch.Tensor, v: torch.Tensor):
        if self.k_cache is None and self.v_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([self.k_cache, k], dim=-2)
            v_cache = torch.cat([self.v_cache, v], dim=-2)

        self.k_cache = k_cache
        self.v_cache = v_cache

        return k_cache, v_cache

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: torch.Tensor = None
    ):
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b n h d', h=self.n_head)
        
        # target-aware
        q = apply_rotary_emb(q, freqs_cis[:, -q.shape[1]:, ...])

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache:
            k, v = self.update_kv_cache(k, v)
            
        output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            is_causal=True if self.training else False,
            dropout_p=self.attn_drop if self.training else 0
        )            
        output = rearrange(output, 'b h n d -> b n (h d)').contiguous()
        output = self.proj_drop(self.proj(output))
        return output


class SelfDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attn = Attention(config)
        self.ffn = FeedForward(config)
        
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None
    ):
        h = x + self.attn(x=self.attn_norm(x), freqs_cis=freqs_cis[:, :x.shape[1], ...])
        out = h + self.ffn(self.ffn_norm(h))
        
        return out
    

class CrossDecoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.attn = CrossAttention(config)
        self.ffn = FeedForward(config)
        
        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        
    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: torch.Tensor = None
    ):
        h = x + self.attn(x=self.attn_norm(x), k=k, v=v, freqs_cis=freqs_cis)
        out = h + self.ffn(self.ffn_norm(h))
        
        return out
    
    
class Decoder_Decoder(nn.Module):
    def __init__(self, config: ModelArgs, n_layer):
        super().__init__()
        self.config = config
        self.self_dec = nn.ModuleList([SelfDecoder(config) for _ in range(n_layer//2)])
        self.cross_dec = nn.ModuleList([CrossDecoder(config) for _ in range(n_layer//2)])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.to_k = nn.Linear(config.dim, config.dim, bias=False)
        self.to_v = nn.Linear(config.dim, config.dim, bias=False)
        
        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None
        
    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None
        
    def update_kv_cache(self, k: torch.Tensor, v: torch.Tensor, head_first=False):
        t_dim = 2 if head_first else 1
        
        if self.k_cache is None and self.v_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([self.k_cache, k], dim=t_dim)
            v_cache = torch.cat([self.v_cache, v], dim=t_dim)

        self.k_cache = k_cache
        self.v_cache = v_cache

        return k_cache, v_cache

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        freqs_cis: torch.Tensor = None
    ):
        for layer in self.self_dec:
            x = layer(x=x, freqs_cis=freqs_cis)
            
        x_norm = self.norm(x)
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.config.n_head), (k, v))
        k = apply_rotary_emb(k, freqs_cis[:, :k.shape[1], ...])

        if self.kv_cache:
            k, v = self.update_kv_cache(k, v)
        
        for layer in self.cross_dec:
            q = layer(x=q, k=k, v=v, freqs_cis=freqs_cis)
            
        return q


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.image_seq_len = config.block_size
        
        """
        ref: https://github.com/bytedance/1d-tokenizer/blob/main/modeling/rar.py
        Token space:
            [0, vocab_size - 1]                         : those are the learned quantized image tokens
            [vocab_size]                                : the mask token id
            [vocab_size + 1, vocab_size + num_classes]  : the imagenet class tokens
            [vocab_size + num_classes + 1]              : the class drop label
            [vocab_size + num_classes + 2]              : the drop token for scg
        """
        self.embeddings = nn.Embedding(config.vocab_size + 1 + config.num_classes + 1 + 1, config.dim)
        self.embed_drop = nn.Dropout(config.token_dropout_p)
        
        self.mask_token_id = config.vocab_size
        self.none_conds_id = config.vocab_size + config.num_classes + 1
        self.none_token_id = config.vocab_size + config.num_classes + 2
        
        # 2-pass decoder
        self.layers = Decoder_Decoder(config, config.n_layer)

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.image_seq_len ** 0.5)
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, config.dim // config.n_head, config.rope_base, config.cls_token_num)

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.head.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_kv_cache(self, enable=True):
        for block in self.layers.self_dec:
            block.attn.kv_cache = enable
            block.attn.reset_kv_cache()

        self.layers.kv_cache = enable
        self.layers.reset_kv_cache()
    
    def preprocess_condition(self, condition, cond_drop_prob=0.0):
        # Set class condition to None condition
        drop_label_mask = torch.rand_like(condition, dtype=torch.float) < cond_drop_prob
        condition = condition + self.config.vocab_size + 1  # [0, 999] -> [codebook_size + 1, codebook_size + 999]
        condition[drop_label_mask] = self.none_conds_id
        
        if condition.ndim == 1:
            condition = condition.unsqueeze(-1)
        
        return condition
    
    def forward_shared(self, input_ids, freqs_cis, num_query=None):
        embedds = self.embeddings(input_ids)
        
        x = self.embed_drop(embedds)
        num_query = input_ids.shape[-1] if num_query == None else num_query
        queries = self.embeddings(torch.full((input_ids.shape[0], num_query), self.mask_token_id, device=input_ids.device))
        
        x = self.layers(x, queries, freqs_cis=freqs_cis)
        logits = self.head(self.norm(x)).float()
        
        return logits

    def forward(self, input_ids, condition, targets=None, debug=False):
        # shift class id and dropout for classifier-free guidance
        condition = self.preprocess_condition(condition, cond_drop_prob=self.config.class_dropout_prob)
        
        # shuffle input
        shuffled_ids, orders = batch_seq_shuffle(input_ids)
        
        # shuffle RoPE
        freqs_cis = self.freqs_cis.unsqueeze(0).repeat(input_ids.shape[0], 1, 1, 1).to(input_ids.device)
        fixed_freqs_cis = freqs_cis[:, :1, ...]
        shuffled_freqs_cis = batch_seq_shuffle(freqs_cis[:, 1:, ...], orders)[0]
        freqs_cis = torch.cat([fixed_freqs_cis, shuffled_freqs_cis], dim=1)

        # teacher-forcing input
        logits = self.forward_shared(torch.cat([condition, shuffled_ids[:, :-1]], dim=-1), freqs_cis)
        
        loss = None
        if targets is not None:
            targets = batch_seq_shuffle(targets, orders)[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        condition,
        guidance_scale=4.0,
        cfg_schedule='linear',
        sample_schedule='arccos',
        temperature=1.0,
        top_k=0,
        top_p=1,
        seq_len=256,
        num_iter=64,
    ):
        device = condition.device
        num_samples = condition.shape[0]
        freqs_cis_ = self.freqs_cis.unsqueeze(0).to(device)
        
        # shift condition id
        condition = self.preprocess_condition(condition, cond_drop_prob=0.0)

        # generate a random order
        orders = torch.rand(256, device=device).argsort(dim=0) + 1

        last_pos = 0
        last_range = range(0, 1)  # for class token, hardcode
        sequences = []
        
        self.setup_kv_cache(enable=True)
        for step in range(num_iter):
            if sample_schedule == 'arccos':
                mask_ratio = np.arccos(1. * (step + 1) / num_iter) / (math.pi * 0.5)
            elif sample_schedule == 'cosine':
                mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            else:
                raise NotImplementedError

            mask_len = int(seq_len * mask_ratio)
            mask_len = max(1, min(seq_len - last_pos - 1, mask_len))
            
            num_pred = seq_len - last_pos - mask_len
            if step == num_iter - 1:
                num_pred = seq_len - last_pos

            next_range = orders[range(last_pos, last_pos + num_pred)]
            last_pos += num_pred
            
            if cfg_schedule == 'linear':
                cfg_scale = 1.0 + (guidance_scale - 1.0) * last_pos / seq_len
            elif cfg_schedule == 'constant':
                cfg_scale = guidance_scale
            else:
                raise NotImplementedError

            """
            1. Since the cached key has already had rotary embedding applied,
               we only need to input the current position's frequencies for key.
            2. We need the next position's frequencies for query to achieve target-aware guidance.
            """
            freqs_cis = torch.cat([
                freqs_cis_[:, last_range, ...],
                freqs_cis_[:, next_range, ...]], dim=1
            )
            if guidance_scale != 0:
                if step == 0:
                    input_ids = torch.cat([condition, torch.full_like(condition, self.none_conds_id)], dim=0)
                else:
                    input_ids = torch.cat([sequences[-1], sequences[-1]], dim=0)

                logits = self.forward_shared(input_ids, freqs_cis, num_pred)
                cond_logits, uncond_logits = logits[:num_samples], logits[num_samples:]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                raise NotImplementedError

            # keep the logits of last n-tokens
            logits = logits[:, -num_pred:] / max(temperature, 1e-5)

            if top_k > 0 or top_p < 1.0:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.flatten(0, 1), num_samples=1)
            sequences.append(sampled.reshape(num_samples, -1))
            
            last_range = next_range
            
        self.setup_kv_cache(enable=False)

        sequences = torch.cat(sequences, dim=-1)
        return sequences[:, orders.argsort(dim=0)]


# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def ARPG_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))

def ARPG_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))

def ARPG_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))


ARPG_models = {'ARPG-L': ARPG_L, 'ARPG-XL': ARPG_XL, 'ARPG-XXL': ARPG_XXL}