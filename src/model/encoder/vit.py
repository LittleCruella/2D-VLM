import torch
import torch.nn as nn
from einops import pack, rearrange, repeat
from einops.layers.torch import Rearrange

# Định nghĩa lại các lớp FeedForward, Attention, Transformer
# Bạn cần đảm bảo các lớp này được định nghĩa trước khi sử dụng ViTEncoder và Vit2D
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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# Class ViTEncoder đã được chỉnh sửa cho 2D
class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size=[256, 256],  # Kích thước 2D
        patch_size=16,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim=4,
        channels=3,  # Số kênh cho ảnh 2D (ví dụ: 3 cho RGB)
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        h, w = image_size[0], image_size[1] 

        assert (h % patch_size == 0) and (w % patch_size == 0), \
            "Image dimensions must be divisible by the patch size."

        self.vit_img_dim = [i // patch_size for i in image_size]
        num_patches = (h // patch_size) * (w // patch_size) 

        patch_dim = channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", 
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

# Class Vit2D đã được chỉnh sửa
class Vit2D(nn.Module): 
    def __init__(self, input_size=[256, 256], patch_size=16, dim=512, depth=8): 
        super().__init__()

        self.encoder = ViTEncoder(input_size, patch_size, dim, depth)

    def forward(self, image, mask=None, device="cuda"): 
        tokens = self.encoder(image)
        shape = tokens.shape
        *_, h, w = shape

        # Nếu bạn có sử dụng VectorQuantize, hãy bỏ comment phần này
        # tokens, _ = pack([tokens], "b * d")
        # vq_mask = None
        # tokens, _, _ = self.vq(tokens, mask = vq_mask)
        # tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        return tokens


# --- Khởi tạo và kiểm tra ---
if __name__ == "__main__":
    # Tham số cho mô hình ViT2D
    image_size = [256, 256]
    patch_size = 16
    dim = 512
    depth = 6
    heads = 8
    mlp_dim = 2048 # Thường mlp_dim = 4 * dim
    channels = 3 # Ảnh màu RGB

    # Tạo một ảnh giả (batch_size, channels, height, width)
    batch_size = 2
    dummy_image = torch.randn(batch_size, channels, image_size[0], image_size[1])

    # Khởi tạo mô hình ViT2D
    model_2d = Vit2D(
        input_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
    )

    print(f"Kích thước ảnh đầu vào giả: {dummy_image.shape}")

    # Truyền ảnh qua mô hình
    output_tokens = model_2d(dummy_image)

    print(f"Kích thước đầu ra của ViT2D: {output_tokens.shape}")

    # Kiểm tra kích thước đầu ra
    # Output should be (batch_size, channels, new_height, new_width)
    # new_height = image_size[0] // patch_size
    # new_width = image_size[1] // patch_size
    expected_output_channels = dim # Output của ViTEncoder là dim
    expected_output_height = image_size[0] // patch_size
    expected_output_width = image_size[1] // patch_size


    assert output_tokens.shape == (batch_size, expected_output_channels, expected_output_height, expected_output_width), \
        f"Kích thước đầu ra không khớp! Expected: {(batch_size, expected_output_channels, expected_output_height, expected_output_width)}, Got: {output_tokens.shape}"

    print("\nKiểm tra thành công! Mô hình ViT2D hoạt động với ảnh 2D.")

    # Bạn có thể thử với ảnh grayscale (1 kênh)
    print("\n--- Kiểm tra với ảnh grayscale (1 kênh) ---")
    channels_gray = 1
    dummy_image_gray = torch.randn(batch_size, channels_gray, image_size[0], image_size[1])

    # Khởi tạo lại ViTEncoder và Vit2D với channels=1
    model_2d_gray = Vit2D(
        input_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
    )
    # Cần tạo lại ViTEncoder với số kênh mới nếu muốn kiểm tra riêng
    encoder_gray = ViTEncoder(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        channels=channels_gray # Quan trọng: thay đổi số kênh
    )
    # Gán encoder mới cho model_2d_gray
    model_2d_gray.encoder = encoder_gray

    print(f"Kích thước ảnh grayscale đầu vào giả: {dummy_image_gray.shape}")
    output_tokens_gray = model_2d_gray(dummy_image_gray)
    print(f"Kích thước đầu ra của ViT2D (grayscale): {output_tokens_gray.shape}")

    assert output_tokens_gray.shape == (batch_size, expected_output_channels, expected_output_height, expected_output_width), \
        f"Kích thước đầu ra (grayscale) không khớp! Expected: {(batch_size, expected_output_channels, expected_output_height, expected_output_width)}, Got: {output_tokens_gray.shape}"
    
    print("Kiểm tra ảnh grayscale thành công!")