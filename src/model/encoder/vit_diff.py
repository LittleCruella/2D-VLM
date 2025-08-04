import os
import sys
# Đường dẫn tương đối cho các module
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
parent_dir = os.path.abspath(os.path.join(src_dir, '..'))
parent_dir = os.path.abspath(os.path.join(parent_dir, '..'))
sys.path.append(parent_dir)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from einops.layers.torch import Rearrange
from src.kernel.rotary import apply_rotary_emb
# from flash_attn import flash_attn_func

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None) 

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

# Define the MultiheadDiffAttn class
class MultiheadDiffAttn(nn.Module):
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)
        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")  # Debugging line
        # print(rel_pos)
        # rel_pos: tuple (sin, cos) for rotary embedding, if None, generate default
        # if rel_pos[0] is None or rel_pos[1] is None:
        #     seq_len = max(q.shape[1], k.shape[1])
        #     dim = q.shape[-1]
        #     # Rotary embedding: create sin, cos for rotary position encoding
        #     pos = torch.arange(seq_len, device=x.device, dtype=q.dtype)
        #     inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=x.device, dtype=q.dtype) / dim))
        #     sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
        #     sin = torch.sin(sinusoid_inp)
        #     cos = torch.cos(sinusoid_inp)
        #     # print(f"sin shape: {sin.shape}, cos shape: {cos.shape}")  # Debugging line
        #     # # Expand to match batch and head dims
        #     # sin = sin[None, None, :, :].expand(q.shape[0], q.shape[2], -1, -1)
        #     # cos = cos[None, None, :, :].expand(q.shape[0], q.shape[2], -1, -1)
        #     # # Ensure cos and sin have the correct dimensions (seqlen_ro, rotary_dim)
        #     # sin = sin.squeeze()[:seq_len, :dim]
        #     # cos = cos.squeeze()[:seq_len, :dim]
        #     rel_pos = (sin, cos)
        # # print(f"rel_pos shape: {rel_pos[0].shape}, {rel_pos[1].shape}")  # Debugging line
        # q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        # k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += attn_mask   
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList([
                    MultiheadDiffAttn(dim, i, heads),
                    FeedForward(dim, mlp_dim, dropout=dropout),
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x, rel_pos=(None, None)) + x
            x = ff(x) + x
        return x

# Define the ViTEncoder class
class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size=[256, 256],
        patch_size=16,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim=2048,
        channels=1,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        h, w = image_size[0], image_size[1] 
        assert (h % patch_size == 0) and (w % patch_size == 0), "Image dimensions must be divisible by the patch size."

        self.vit_img_dim = [i // patch_size for i in image_size]
        num_patches = (h // patch_size) * (w // patch_size)

        patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Debugging line
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 1:, :]

        x = rearrange(
            x,
            "b (x y) c -> b c x y",
            x=self.vit_img_dim[0],
            y=self.vit_img_dim[1],
        )
        return x

# Define the Vit2D class
class Vit2Ddiff(nn.Module):
    def __init__(self, input_size=[256, 256], patch_size=16, dim=512, depth=8):
        super().__init__()
        self.encoder = ViTEncoder(input_size, patch_size, dim, depth)

    def forward(self, image, mask=None, device="cuda"):
        tokens = self.encoder(image)
        tokens = rearrange(tokens, "b c h w -> b h w c")  # Đảm bảo tokens có dạng (batch_size, height, width, channels)
        tokens, _ = pack([tokens], "h * w")
        return tokens

if __name__ == "__main__":
    # Mock test for Vit2D
    import torch

    # Create a dummy batch of images (batch size = 2, 3 color channels, 256x256 resolution)
    dummy_images = torch.randn(1, 1, 256, 256)

    # Initialize the Vit2D model
    model = Vit2Ddiff(input_size=[256, 256], patch_size=16, dim=512, depth=8)

    # Move model and data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_images = dummy_images.to(device)

    # Perform a forward pass
    output = model(dummy_images)

    # Print the output shape
    print("Output shape:", output.shape)
