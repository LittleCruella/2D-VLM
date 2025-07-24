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
        patch_size=2,
        dim=512,
        depth=8,
        heads=8,
        mlp_dim=4,
        channels=1,  # Số kênh cho ảnh 2D (ví dụ: 3 cho RGB)
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
        # print(f"Output shape after rearranging: {x.shape}")  # Debugging line
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
        # Đảm bảo tokens có kích thước (batch_size, channels, height, width)
        # Nếu bạn có sử dụng VectorQuantize, hãy bỏ comment phần này
        # tokens, _ = pack([tokens], "b * d")
        # vq_mask = None
        # tokens, _, _ = self.vq(tokens, mask = vq_mask)
        # tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)
        # print(f"Output shape of Vit2D: {tokens.shape}")  # Debugging line
        tokens = rearrange(tokens, "b c h w -> b h w c")  # Đảm bảo tokens có dạng (batch_size, height, width, channels)
        tokens, _ = pack([tokens], "h * w")
        return tokens


# --- Khởi tạo và kiểm tra ---
if __name__ == "__main__":
    # Lưu trữ shapes
    activation_shapes = {}

    # Hàm hook
    def get_output_shape_hook(module, input, output):
        # Lưu shape của output. Module được gắn hook là key.
        # output có thể là tensor hoặc tuple/list of tensors. Lấy shape của tensor chính.
        if isinstance(output, torch.Tensor):
            activation_shapes[module.__class__.__name__] = output.shape
        elif isinstance(output, (tuple, list)):
            # Nếu output là tuple/list (ví dụ từ BatchNorm), lấy shape của tensor đầu tiên
            activation_shapes[module.__class__.__name__] = output[-2].shape

    # Tham số cho mô hình ViT2D
    image_size = [256, 256]
    patch_size = 16
    dim = 512
    depth = 8 # Sử dụng depth=6 như trong phần test của bạn
    heads = 8
    mlp_dim = 4 # Đảm bảo mlp_dim = 4 * dim
    channels = 1 # Ảnh màu RGB

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
    
    # Đăng ký hooks
    # Lớp áp chót: Output của self.transformer trong ViTEncoder
    # self.transformer là một module con của self.encoder
    model_2d.encoder.transformer.register_forward_hook(get_output_shape_hook)
    
    # Lớp cuối cùng: Output của self.encoder sau khi rearrange
    # Chúng ta có thể gắn hook vào chính encoder
    model_2d.encoder.register_forward_hook(get_output_shape_hook)
    # HOẶC, nếu bạn muốn lớp cuối cùng chính là output của Vit2D, hãy gắn hook vào model_2d
    # model_2d.register_forward_hook(get_output_shape_hook)


    print(f"Kích thước ảnh đầu vào giả: {dummy_image.shape}")

    # Truyền ảnh qua mô hình
    # Output của Vit2D là tokens_rearranged_for_output (dạng b h w c)
    output_tokens = model_2d(dummy_image)

    print(f"Kích thước đầu ra của ViT2D: {output_tokens.shape}")
    print("\n--- Kích thước các lớp trung gian (sử dụng Hooks) ---")
    
    # Lớp áp chót (penultimate layer): Là đầu ra của Transformer.
    # Hook đã lưu output của Transformer là 'Transformer'.
    # Lưu ý: output của Transformer là (batch_size, num_patches + 1, dim)
    if 'Transformer' in activation_shapes:
        print(f"Shape của lớp áp chót (output của Transformer, bao gồm CLS token): {activation_shapes['Transformer']}")
        # Nếu bạn muốn chỉ các patch tokens (loại bỏ CLS token)
        # Thì số lượng patch tokens là num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        expected_penultimate_without_cls_shape = (batch_size, num_patches, dim)
        print(f"Shape của lớp áp chót (chỉ patch tokens, ước tính): {expected_penultimate_without_cls_shape}")
    else:
        print("Không tìm thấy shape của Transformer. Đảm bảo hook đã được đăng ký đúng.")

    # Lớp cuối cùng (final layer): Là đầu ra của ViTEncoder sau khi rearrange
    # Hook đã lưu output của ViTEncoder là 'ViTEncoder'.
    if 'ViTEncoder' in activation_shapes:
        print(f"Shape của lớp cuối cùng (output của ViTEncoder): {activation_shapes['ViTEncoder']}")
    elif 'Vit2D' in activation_shapes: # Nếu bạn đã gắn hook vào Vit2D
        print(f"Shape của lớp cuối cùng (output của Vit2D): {activation_shapes['Vit2D']}")
    else:
        print("Không tìm thấy shape của ViTEncoder hoặc Vit2D. Đảm bảo hook đã được đăng ký đúng.")


    # Kiểm tra kích thước đầu ra của Vit2D (lưu ý: bạn đã thay đổi định dạng output)
    expected_output_height = image_size[0] // patch_size
    expected_output_width = image_size[1] // patch_size
    expected_output_channels = dim # Output của ViTEncoder là dim, sau rearrange là chiều cuối

    assert output_tokens.shape == (batch_size, expected_output_height, expected_output_width, expected_output_channels), \
        f"Kích thước đầu ra không khớp! Expected: {(batch_size, expected_output_height, expected_output_width, expected_output_channels)}, Got: {output_tokens.shape}"

    print("\nKiểm tra thành công! Mô hình ViT2D hoạt động với ảnh 2D.")

    # Bạn có thể thử với ảnh grayscale (1 kênh)
    print("\n--- Kiểm tra với ảnh grayscale (1 kênh) ---")
    channels_gray = 1
    dummy_image_gray = torch.randn(batch_size, channels_gray, image_size[0], image_size[1])

    # (Lưu ý: Để kiểm tra với grayscale, bạn cần tạo một instance ViTEncoder mới với channels=1,
    # sau đó gán nó vào model_2d_gray.encoder và đăng ký lại hooks.)

    # Khởi tạo lại ViTEncoder và Vit2D với channels=1
    model_2d_gray = Vit2D(
        input_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
    )
    encoder_gray = ViTEncoder( # Tạo một encoder mới với kênh xám
        image_size=image_size,
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        channels=channels_gray
    )
    model_2d_gray.encoder = encoder_gray # Gán encoder mới vào model_2d_gray

    # Xóa shapes cũ và đăng ký lại hooks cho model_2d_gray
    activation_shapes = {}
    model_2d_gray.encoder.transformer.register_forward_hook(get_output_shape_hook)
    model_2d_gray.encoder.register_forward_hook(get_output_shape_hook)


    print(f"Kích thước ảnh grayscale đầu vào giả: {dummy_image_gray.shape}")
    output_tokens_gray = model_2d_gray(dummy_image_gray)
    print(f"Kích thước đầu ra của ViT2D (grayscale): {output_tokens_gray.shape}")

    print("\n--- Kích thước các lớp trung gian (grayscale, sử dụng Hooks) ---")
    if 'Transformer' in activation_shapes:
        print(f"Shape của lớp áp chót (output của Transformer, bao gồm CLS token): {activation_shapes['Transformer']}")
        num_patches_gray = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        expected_penultimate_without_cls_shape_gray = (batch_size, num_patches_gray, dim)
        print(f"Shape của lớp áp chót (chỉ patch tokens, ước tính): {expected_penultimate_without_cls_shape_gray}")
    if 'ViTEncoder' in activation_shapes:
        print(f"Shape của lớp cuối cùng (output của ViTEncoder): {activation_shapes['ViTEncoder']}")


    assert output_tokens_gray.shape == (batch_size, expected_output_height, expected_output_width, expected_output_channels), \
        f"Kích thước đầu ra (grayscale) không khớp! Expected: {(batch_size, expected_output_height, expected_output_width, expected_output_channels)}, Got: {output_tokens_gray.shape}"
    
    print("Kiểm tra ảnh grayscale thành công!")