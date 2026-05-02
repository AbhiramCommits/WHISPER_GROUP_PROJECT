"""
Custom Whisper-style model with two architectural improvements:

1. RoPE (Rotary Positional Embeddings) instead of sinusoidal positional embeddings
2. FlashAttention via torch.nn.functional.scaled_dot_product_attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from whisper.model import ModelDimensions



## helpers
def precompute_rope_freqs(dim: int, max_seq_len: int, base: float = 10000.0):
    theta = 1.0 / (base ** (torch.arange(0, dim // 2, dtype=torch.float32) / (dim // 2)))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)



class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, max_seq_len: int = 1500):
        super().__init__()
        self.n_head   = n_head
        self.n_state  = n_state
        self.head_dim = n_state // n_head

        self.query = nn.Linear(n_state, n_state, bias=False)
        self.key   = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state, bias=False)
        self.out   = nn.Linear(n_state, n_state, bias=False)

        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, xa=None, mask=None, kv_cache=None):

        is_cross = xa is not None
        kv_src = xa if is_cross else x
        q = self.query(x)
        k = self.key(kv_src)
        v = self.value(kv_src)

        if kv_cache is not None:
            kid = f"{id(self)}_k"
            vid = f"{id(self)}_v"
            if kid in kv_cache:
                k = torch.cat([kv_cache[kid], k], dim=1)
                v = torch.cat([kv_cache[vid], v], dim=1)
            kv_cache[kid] = k
            kv_cache[vid] = v
        sk = k.shape[1]
        b, sq, _ = q.shape
        
        q = q.view(b, sq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, sk, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, sk, self.n_head, self.head_dim).transpose(1, 2)

    # Apply RoPE only to self-attention (not cross-attention)
        if not is_cross:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        # FlashAttention with PyTorch SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=(not is_cross) and (mask is not None)
        )

        out = out.transpose(1, 2).contiguous().view(b, sq, self.n_state)
        return self.out(out), None


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class RoPEResidualAttentionBlock(nn.Module):
    def __init__(self, num_state, n_head, cross_attention, max_seq_len=1500):
        super().__init__()
        self.attn    = RoPEMultiHeadAttention(num_state, n_head, max_seq_len)
        self.attn_ln = nn.LayerNorm(num_state)

        self.cross_attn    = ''
        self.cross_attn_ln = ''
        if cross_attention:
            self.cross_attn    = RoPEMultiHeadAttention(num_state, n_head, max_seq_len)
            self.cross_attn_ln = nn.LayerNorm(num_state)

        self.mlp = nn.Sequential(
            nn.Linear(num_state, num_state * 4),
            nn.ReLU(),   
            nn.Linear(num_state * 4, num_state),
        )
        self.mlp_ln = nn.LayerNorm(num_state)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn is not None and xa is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa=xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


# ---------------------------------------------------------------------------
# Audio Encoder
# ---------------------------------------------------------------------------

class RoPEAudioEncoder(nn.Module):
    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        self.positional_embedding = nn.Parameter(torch.zeros(n_ctx, n_state))
        nn.init.normal_(self.positional_embedding, std=0.02)

        self.blocks  = nn.ModuleList([
            RoPEResidualAttentionBlock(n_state, n_head,
                                       cross_attention=False,
                                       max_seq_len=n_ctx)
            for _ in range(n_layer)
        ])
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x):
        x = F.relu(self.conv1(x))   
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)                        
        assert x.shape[1] <= self.positional_embedding.shape[0], (
            f"Audio sequence length {x.shape[1]} exceeds "
            f"n_audio_ctx {self.positional_embedding.shape[0]}"
        )
        x = x + self.positional_embedding[:x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.ln_post(x)                          


# ---------------------------------------------------------------------------
# Text Decoder
# ---------------------------------------------------------------------------

class RoPETextDecoder(nn.Module):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.n_ctx = n_ctx
        self.token_embedding      = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.zeros(n_ctx, n_state))
        nn.init.normal_(self.positional_embedding, std=0.02)

        self.blocks = nn.ModuleList([
            RoPEResidualAttentionBlock(n_state, n_head,
                                       cross_attention=True,
                                       max_seq_len=n_ctx)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_state)

        # Tie output projection weights to token embeddings
        self.output_projection        = nn.Linear(n_state, n_vocab, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def forward(self, tokens, xa, kv_cache=None):
        """
        tokens: (batch, seq)  — integer token ids
        xa:     (batch, audio_ctx, n_state)  — encoder output
        """
        # tokens is (batch, seq) — get the token sequence length
        seq = tokens.shape[-1]

        offset = 0
        if kv_cache is not None and len(kv_cache) > 0:
            offset = next(iter(kv_cache.values())).shape[1]

        assert offset + seq <= self.n_ctx, (
            f"Token sequence too long: offset={offset} seq={seq} n_ctx={self.n_ctx}"
        )

        # Embed tokens and add positional embedding
        if tokens.ndim == 3:
            tokens = tokens.squeeze(-1)
        x = self.token_embedding(tokens)                          # (batch, seq, n_state)
        x = x + self.positional_embedding[offset: offset + seq]  # (batch, seq, n_state)

        # Causal mask for self-attention — shape (seq, seq)
        mask = torch.full((seq, seq), float("-inf"),
                          device=x.device, dtype=x.dtype).triu_(1)

        for block in self.blocks:
            x = block(x, xa=xa, mask=mask, kv_cache=kv_cache)

        x = self.ln(x)
        return self.output_projection(x)                          # (batch, seq, n_vocab)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class RoPEWhisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims    = dims
        self.encoder = RoPEAudioEncoder(
            n_mels=dims.n_mels,
            n_ctx=dims.n_audio_ctx,
            n_state=dims.n_audio_state,
            n_head=dims.n_audio_head,
            n_layer=dims.n_audio_layer,
        )
        self.decoder = RoPETextDecoder(
            n_vocab=dims.n_vocab,
            n_ctx=dims.n_text_ctx,
            n_state=dims.n_text_state,
            n_head=dims.n_text_head,
            n_layer=dims.n_text_layer,
        )

    def forward(self, mel, tokens):
        return self.decoder(tokens, self.encoder(mel))


# ---------------------------------------------------------------------------
# Model config — same dimensions as your original
# ---------------------------------------------------------------------------

SMALL_DIMS = ModelDimensions(
    n_mels=80,
    n_audio_ctx=1500,
    n_audio_state=384,
    n_audio_head=6,
    n_audio_layer=4,
    n_vocab=51865,
    n_text_ctx=448,
    n_text_state=384,
    n_text_head=6,
    n_text_layer=4,
)


def get_model():
    return RoPEWhisper(SMALL_DIMS)