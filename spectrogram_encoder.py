import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        """Gated Liner Unit with Swish Activation"""
        super().__init__()
        # Init up- and down- projection layers
        self.fc1 = nn.Linear(hidden_dim, 2 * intermediate_dim, bias=False)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU to input data.

        Args:
            x: input tensor, shape [bs, seq len, hidden dim]
        Returns:
            result tensor, shape [bs, seq len, hidden dim]
        """
        # todo()
        x1, x3 = torch.chunk(self.fc1(x), 2, dim=-1)
        output = self.fc2(F.silu(x1) * x3)
        return output


class RotaryAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_head == 0
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.head_dim = self.hidden_dim // self.n_head
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def _sinusoidal_position_embedding(self, batch_size: int, n_head: int, seq_len: int, head_dim: int) -> Tensor:
        """Generates sinusoidal position embeddings for a given sequence length and number of heads.

        Args:
            batch_size: Batch size.
            n_head: Number of attention heads.
            seq_len: Sequence length.
            head_dim: Dimension per head.

        Returns:
            A Tensor of shape [batch_size, n_head, seq_len, head_dim] containing sinusoidal position embeddings.
        """
        position = torch.arange(seq_len).unsqueeze(-1).float()  # [seq_len, 1]
        ids = torch.arange(head_dim // 2).float()
        theta = 10000 ** (2 * ids / head_dim)  # [head_dim // 2]
        embeddings = position / theta  # [seq_len, head_dim // 2]

        # Apply sin for even position, cos for odd position
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1).reshape(seq_len, head_dim)
        # Expand the embedding to accommodate batch size and number of heads
        embeddings = embeddings[None, None, :, :].expand(batch_size, n_head, seq_len, head_dim)
        return embeddings

    def _rotary_position_embedding(self, query: Tensor, key: Tensor):
        """Applies Rotary Positional Encoding (RoPE) to query and key tensors.

        This function implements the rotation operation for queries and keys
        as described in the RoPE mechanism. It uses sinusoidal embeddings to
        encode positional information and applies the rotary transformation.

        Args:
            query: Query tensor of shape [batch_size, n_head, seq_len, head_dim].
            key: Key tensor of shape [batch_size, n_head, seq_len, head_dim].

        Returns:
            Tuple containing transformed query and key tensors.
        """
        position_embedding = self._sinusoidal_position_embedding(*query.shape).to(query.device)

        sin_position = position_embedding[..., 0::2].repeat_interleave(2, dim=-1)
        cos_position = position_embedding[..., 1::2].repeat_interleave(2, dim=-1)
        # Prepare query and key with their sine and cosine parts
        query_sin = torch.stack([-query[..., 1::2], query[..., 0::2]], dim=-1).reshape(query.shape)
        key_sin = torch.stack([-key[..., 1::2], key[..., 0::2]], dim=-1).reshape(key.shape)

        # Combine sine and cosine parts with appropriate rotations
        query_cos_position = query * cos_position
        query_sin_position = query_sin * sin_position
        transformed_query = query_cos_position + query_sin_position

        key_cos_position = key * cos_position
        key_sin_position = key_sin * sin_position
        transformed_key = key_cos_position + key_sin_position

        return transformed_query, transformed_key

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        batch_size, seq_len = x.size()[:2]

        qkv = self.qkv_proj(x).view(batch_size, seq_len, self.n_head, 3 * self.head_dim).transpose(1, 2)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query, key = self._rotary_position_embedding(query, key)

        attn_score = torch.matmul(query, key.transpose(2, 3))
        if attention_mask is not None and self.training:
            attention_mask = attention_mask[:, None, :, None].expand(batch_size, self.n_head, seq_len, seq_len).transpose(-1, -2)
            attn_score[attention_mask == 0] = -float('inf')
        attn_probs = F.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_probs, value)  # batch_size, self.config.n_head, seq_len, self.head_dim
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        return out


class RotaryEncoderLayer(nn.Module):
    def __init__(self, n_head: int, hidden_dim: int, intermediate_dim: int, dropout: int = 0.1):
        super().__init__()
        self.ln_1 = nn.RMSNorm(hidden_dim)
        self.res_dropout_1 = nn.Dropout(dropout)
        self.attn = RotaryAttention(hidden_dim, n_head, dropout)

        self.ln_2 = nn.RMSNorm(hidden_dim)
        self.res_dropout_2 = nn.Dropout(dropout)
        self.mlp = SwiGLU(hidden_dim, intermediate_dim)

    def forward(self, x: Tensor, attention_mask: Tensor = None) -> Tensor:
        """Apply Transformer Block to input data.

        Args:
            x: input tensor, shape [batch_size, seq len, hidden dim]
            attention_mask: mask with zeros for pad tokens, shape [batch_size, seq len, hidden dim]
        Returns:
            result tensor, shape [batch_size, seq len, hidden dim]
        """
        # Attention Block
        norm_x = self.ln_1(x)
        attn_output = self.attn(norm_x, attention_mask)
        x = x + self.res_dropout_1(attn_output)

        # Full Connection Block
        norm_x = self.ln_2(x)
        mlp_output = self.mlp(norm_x)
        x = x + self.res_dropout_2(mlp_output)

        return x


class SpectrogramEncoder(nn.Module):
    def __init__(self, n_layer: int, n_head: int, hidden_dim: int, dropout: float = 0.1, mask_prob: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.mask_prob = mask_prob

        self.layers = nn.ModuleList([RotaryEncoderLayer(n_head, self.hidden_dim, self.hidden_dim * 8 // 3, dropout) for _ in range(n_layer)])
        self.ln_final = nn.RMSNorm(self.hidden_dim)

    def _generate_attention_mask(self, batch_size: int, seq_len: int):
        mask_per_seq = int(seq_len * self.mask_prob)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.float32)
        mask_position = torch.rand(batch_size, seq_len).argsort(dim=1)
        mask = torch.arange(seq_len).expand(batch_size, seq_len) < mask_per_seq
        mask = mask.scatter(1, mask_position, mask)
        return attention_mask

    def forward(self, x: torch.Tensor) -> Tensor:
        attention_mask = self._generate_attention_mask(x.shape[0], x.shape[1])
        for layer in self.layers:
            x = layer(x, attention_mask)
        logits = self.ln_final(x)
        return logits
