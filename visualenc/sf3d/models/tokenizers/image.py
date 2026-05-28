from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from visualenc.sf3d.models.tokenizers.dinov2 import Dinov2Model, set_adapter_dino
# from visualenc.sf3d.models.tokenizers.sapiens import VisionTransformer, set_adapter
from visualenc.sf3d.models.transformers.attention import Modulation
from visualenc.sf3d.models.utils import BaseModule
from torch.nn import functional as F




class DINOV2SingleImageTokenizer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = "facebook/dinov2-large"
        width: int = 512
        height: int = 512
        modulation_cond_dim: int = 768
        lora_on: bool = False

    cfg: Config

    def configure(self) -> None:
        self.model = Dinov2Model.from_pretrained(self.cfg.pretrained_model_name_or_path)

        # lora_on = False
        if self.cfg.lora_on:
            set_adapter_dino(self.model, in_dim=1024, out_dim=1024, r=32)
            for n, p in self.model.named_parameters():
                if "lora" not in n:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(True)
        else:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        self.model.set_gradient_checkpointing(False)

        # add modulation
        modulations = []
        for layer in self.model.encoder.layer:
            norm1_modulation = Modulation(
                self.model.config.hidden_size,
                self.cfg.modulation_cond_dim,
                zero_init=True,
                single_layer=True,
            )
            norm2_modulation = Modulation(
                self.model.config.hidden_size,
                self.cfg.modulation_cond_dim,
                zero_init=True,
                single_layer=True,
            )
            layer.register_ada_norm_modulation(norm1_modulation, norm2_modulation)
            modulations += [norm1_modulation, norm2_modulation]
        self.modulations = nn.ModuleList(modulations)

        self.register_buffer(
            "image_mean",
            torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        images: Float[Tensor, "B *N C H W"],
        modulation_cond: Optional[Float[Tensor, "B *N Cc"]],
        **kwargs,
    ) -> Float[Tensor, "B *N Ct Nt"]:
        model = self.model

        packed = False
        if images.ndim == 4:
            packed = True
            images = images.unsqueeze(1)
            if modulation_cond is not None:
                assert modulation_cond.ndim == 2
                modulation_cond = modulation_cond.unsqueeze(1)

        batch_size, n_input_views = images.shape[:2]
        images = (images - self.image_mean) / self.image_std
        out = model(
            rearrange(images, "B N C H W -> (B N) C H W"),
            modulation_cond=rearrange(modulation_cond, "B N Cc -> (B N) Cc")
            if modulation_cond is not None
            else None,
        )
        local_features = out.last_hidden_state
        local_features = local_features.permute(0, 2, 1)
        local_features = rearrange(
            local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
        )
        if packed:
            local_features = local_features.squeeze(1)

        return local_features

    def detokenize(self, *args, **kwargs):
        raise NotImplementedError




# class SapiensSingleImageTokenizer(BaseModule):
#
#     @dataclass
#     class Config(BaseModule.Config):
#         arch: str = 'sapiens_0.3b'
#         patch_size: int = 16
#         img_size: int = 1024
#         final_norm: bool = True
#         modulation_cond_dim: int = 768
#         out_type: str = 'raw'
#
#     cfg: Config
#
#
#     def configure(self) -> None:
#         self.model = VisionTransformer(**self.cfg)
#         set_adapter(self.model, in_dim=1024, out_dim=1024, r=32)
#
#
#         for n, p in self.model.named_parameters():
#             if "lora" not in n:
#                 p.requires_grad_(False)
#             else:
#                 p.requires_grad_(True)
#
#
#
#
#         # self.model.set_gradient_checkpointing(False)
#
#         # add modulation
#         # modulations = []
#         # for layer in self.model.layers:
#         #     norm1_modulation = Modulation(
#         #         self.model.embed_dims,
#         #         self.cfg.modulation_cond_dim,
#         #         zero_init=True,
#         #         single_layer=True,
#         #     )
#         #     norm2_modulation = Modulation(
#         #         self.model.embed_dims,
#         #         self.cfg.modulation_cond_dim,
#         #         zero_init=True,
#         #         single_layer=True,
#         #     )
#         #     layer.register_ada_norm_modulation(norm1_modulation, norm2_modulation)
#         #     modulations += [norm1_modulation, norm2_modulation]
#         # self.modulations = nn.ModuleList(modulations)
#
#         self.register_buffer( # DINO2和Sapiens使用相同的mean和std
#             "image_mean",
#             torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
#             persistent=False,
#         )
#         self.register_buffer(
#             "image_std",
#             torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
#             persistent=False,
#         )
#
#     def forward(
#         self,
#         images: Float[Tensor, "B *N C H W"],
#         modulation_cond: Optional[Float[Tensor, "B *N Cc"]],
#         **kwargs,
#     ) -> Float[Tensor, "B *N Ct Nt"]:
#         model = self.model
#
#         packed = False
#         if images.ndim == 4:
#             packed = True
#             images = images.unsqueeze(1)
#             if modulation_cond is not None:
#                 assert modulation_cond.ndim == 2
#                 modulation_cond = modulation_cond.unsqueeze(1)
#
#         batch_size, n_input_views = images.shape[:2]
#
#         images = (images - self.image_mean) / self.image_std
#         out = model(
#             F.interpolate(rearrange(images, "B N C H W -> (B N) C H W"), scale_factor=2, mode='bilinear'),
#             modulation_cond=rearrange(modulation_cond, "B N Cc -> (B N) Cc")
#             if modulation_cond is not None
#             else None,
#         )
#         cls_token_features = out[0][:, :1, :]
#         image_token_features = F.interpolate(rearrange(out[0][:, 1:, :], "B (h w) C -> B C h w", h=self.cfg.img_size // self.cfg.patch_size, w=self.cfg.img_size // self.cfg.patch_size), size=(36, 36), mode='bilinear')
#         image_token_features = rearrange(image_token_features, "B C h w -> B (h w) C")
#
#         local_features = torch.concatenate([cls_token_features, image_token_features], dim=1)
#         local_features = local_features.permute(0, 2, 1)
#         local_features = rearrange(
#             local_features, "(B N) Ct Nt -> B N Ct Nt", B=batch_size
#         )
#         if packed:
#             local_features = local_features.squeeze(1)
#
#         return local_features
