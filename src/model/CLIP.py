import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

# from src.model.encoder.dcformer import (
#     decomp_base,
#     decomp_naive,
#     decomp_nano,
#     decomp_small,
#     decomp_tiny,
# )
from src.model.encoder.vit import Vit2D
from src.model.encoder.vit_diff import Vit2Ddiff
# from src.model.projector.mlp import MultiLayerPerceptron

# try:
#     import torch.distributed.nn
#     from torch import distributed as dist

#     has_distributed = True
# except ImportError:
#     has_distributed = False

has_distributed = False

class DEC_CLIPConfig(PretrainedConfig):
    model_type = "dec_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        input_size: tuple = (256, 256),
        dim: int = 512,
        depth: int = 12,
        hidden_size: int = 512,
        mlp_depth: int = 2,
        loss_type: str = "nce",
        t_prime: float = np.log(1 / 0.07),
        bias: float = 0.0,
        efficient_loss: bool = False,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.input_size = input_size
        self.dim = dim
        self.depth = depth
        self.hidden_size = hidden_size
        self.mlp_depth = mlp_depth
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.loss_type = loss_type
        self.t_prime = t_prime
        self.bias = bias
        self.efficient_loss = efficient_loss
        super().__init__(**kwargs)


class DEC_CLIP(PreTrainedModel):
    config_class = DEC_CLIPConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if config.vision_encoder == "vit2d":
            self.vision_encoder = Vit2D(
                input_size=config.input_size,
                dim=config.dim,
                depth=config.depth,
            )
        elif config.vision_encoder == "vit2d_diff":
            self.vision_encoder = Vit2Ddiff(
                input_size=config.input_size,
                dim=config.dim,
                depth=config.depth,
            )
        # elif config.vision_encoder == "dcformer":
        #    self.vision_encoder = decomp_small(input_size=config.input_size)
        else:
            raise ValueError(f"Unexpected vision encoder: {config.vision_encoder}")

        self.language_encoder = AutoModel.from_pretrained(
            config.language_model_name_or_path
        )
 
        self.mm_vision_proj = nn.Linear(
            512, config.hidden_size # Lưu ý: Đây là 512, không phải self.vision_encoder.channels[-1]
        )

        self.mm_language_proj = nn.Linear(
            768, config.hidden_size # Lưu ý: Đây là 768, không phải self.language_encoder.config.dim
        )
        self.efficient_loss = config.efficient_loss
        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss # Sẽ bị bỏ qua nếu không có distributed
        self.loss_type = config.loss_type

        if self.loss_type == "sigmoid":
            self.t_prime = nn.Parameter(torch.tensor(config.t_prime))
            self.bias = nn.Parameter(torch.tensor(config.bias))
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * config.t_prime)

    def encode_image(self, image):
        image_feats = self.vision_encoder(image)
        if isinstance(image_feats, list):
            image_feats = image_feats[-1]
        # print("Shape trước mean:", image_feats.shape) # Ví dụ: [B, num_tokens, dim]
        image_feats = image_feats.mean(dim=1) # Lấy average pooling trên các token/patch.
        # print("Shape sau mean:", image_feats.shape) # Ví dụ: [B, dim]
        # Sau Vit2D, dim = config.dim = 768. Do đó mm_vision_proj cần input là 768, không phải 512.
        # self.mm_vision_proj = nn.Linear(self.vision_encoder.channels[-1], config.hidden_size)
        # Nếu Vit2D trả về dim=768, thì mm_vision_proj nên là Linear(768, config.hidden_size)
        image_feats = self.mm_vision_proj(image_feats)
        # print("Shape sau mm_vision_proj:", image_feats.shape)
        image_feats = F.normalize(image_feats, dim=-1)
        # print("Shape sau normalize:", image_feats.shape)
        return image_feats

    def encode_text(self, input_id, attention_mask):
        text_feats = self.language_encoder(input_id, attention_mask=attention_mask)[
            "last_hidden_state"
        ]
        text_feats = text_feats[:, 0] # Lấy CLS token của BERT
        text_feats = self.mm_language_proj(text_feats)
        text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        image_features = self.encode_image(images)
        # print(f"image_features.shape: {image_features.shape}")
        text_features = self.encode_text(input_ids, attention_mask)
        # print(f"text_features.shape: {text_features.shape}")
        # Removed distributed and dist related code
        # rank = 0
        # world_size = 1

        batch_size = image_features.size(0)
        device = image_features.device

        if self.loss_type == "sigmoid":
            # Không có has_distributed, nên chỉ chạy phần else
            t = torch.exp(self.t_prime)

            # gather_features sẽ không còn được sử dụng
            # all_image_features, all_text_features = gather_features(...)
            # Giờ đây, all_image_features = image_features và all_text_features = text_features
            
            # Tính logits
            logits_per_image = (
                image_features @ text_features.T
            ) * t + self.bias
            logits_per_text = logits_per_image.T
            
            # batch_size ở đây là batch_size của local
            labels_tensor = 2 * torch.eye(
                batch_size, device=image_features.device
            ) - torch.ones(batch_size, batch_size, device=image_features.device)

            logits = (logits_per_image + logits_per_text) / 2.0
            # logits = (logits_per_image + logits_per_text).mean(dim=1)    
            # print("logits.shape:", logits.shape)
            # print("labels_tensor.shape:", labels_tensor.shape)
            loss = -torch.sum(F.logsigmoid(labels_tensor * logits)) / batch_size # Chia cho batch_size nếu muốn mean loss

        else: # self.loss_type != "sigmoid" (e.g., "cross_entropy" as in CLIP)
            # gather_features sẽ không còn được sử dụng
            # all_image_features, all_text_features = gather_features(...)
            # Giờ đây, all_image_features = image_features và all_text_features = text_features

            # gather_loss, local_loss không còn tác dụng khi không có distributed
            # nên sẽ chỉ thực hiện phép nhân ma trận cục bộ
            
            logits_per_image = self.logit_scale * image_features @ text_features.T
            # print("logits_per_image.shape:", logits_per_image.shape)
            # print("text_features.shape (transposed for dot product):", text_features.T.shape)
            # print("image_features.shape:", image_features.shape)
            
            logits_per_text = self.logit_scale * text_features @ image_features.T

            # Labels cho Cross Entropy Loss (thường là one-hot hoặc chỉ số)
            # Trong CLIP, labels là ma trận đồng nhất (identity matrix)
            labels_ce = torch.arange(batch_size, device=device) # [0, 1, ..., batch_size-1]

            image_loss = F.cross_entropy(logits_per_image, labels_ce)
            text_loss = F.cross_entropy(logits_per_text, labels_ce)

            loss = (image_loss + text_loss) / 2.0
            logits = (logits_per_image + logits_per_text) / 2.0 # Logits để đánh giá

        ret = {
            "loss": loss,
            "logits": logits,
        }

        return ret


def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=True,
    rank=0,
    world_size=1,
):
    # assert (
    #     has_distributed
    # ), "torch.distributed did not import correctly, please use a PyTorch version with support."

    if not (has_distributed and dist.is_initialized()):
        return image_features, text_features

    if gather_with_grad:
        all_image_features = torch.cat(
            torch.distributed.nn.all_gather(image_features), dim=0
        )
        all_text_features = torch.cat(
            torch.distributed.nn.all_gather(text_features), dim=0
        )
    else:
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


AutoConfig.register("dec_clip", DEC_CLIPConfig)
AutoModel.register(DEC_CLIPConfig, DEC_CLIP)