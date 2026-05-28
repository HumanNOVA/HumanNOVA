import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from collections import OrderedDict
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
    get_rays,
    get_ray_directions
)
from visualenc.sf3d.utils import create_intrinsic_from_fov_deg, default_cond_c2w
from torch.amp import autocast


def save_model_weights_to_dict(model, n_trunk_actual):
    weights_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights_dict[name] = param.detach()

    # last_layer -> color, opacity
    last_weight_name, last_bias_name = list(weights_dict.keys())[-2:]
    last_weights = weights_dict[last_weight_name]
    last_bias = weights_dict[last_bias_name]
    weights_color = [last_weights.T[:, 1:4]]
    biases_color = [last_bias.T[1:4]]
    weights_opacity = [last_weights.T[:, 0:1]]
    biases_opacity = [last_bias.T[0:1]]

    trunk_name_list = list(weights_dict.keys())[:-2]

    weights_trunk, biases_trunk = [], []
    idx = 0
    idx_list = []
    for cur_name in trunk_name_list:
        if idx not in idx_list:
            if 'weight' in cur_name:
                weights_trunk.append(weights_dict[cur_name].T)
            if 'bias' in cur_name:
                biases_trunk.append(weights_dict[cur_name].T)
        idx += 1
    print('Extraction complete')

    first_weight = weights_trunk[0]
    weights_trunk[0] = torch.eye(first_weight.shape[1])

    return first_weight, weights_trunk, biases_trunk, weights_color, biases_color, weights_opacity, biases_opacity


class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str, reload_mlp: bool
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
            use_saved_ckpt = True
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )
            use_saved_ckpt = False

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        if use_saved_ckpt:
            if "module" in list(ckpt["state_dict"].keys())[0]:
                ckpt = {key.replace('module.',''): item for key, item in ckpt["state_dict"].items()}
            else:
                ckpt = ckpt["state_dict"]
        if ckpt['tokenizer.embeddings'].shape[-1] != cfg.tokenizer.plane_size:
            ckpt['tokenizer.embeddings'] = F.interpolate(ckpt['tokenizer.embeddings'],
                                                         size=(cfg.tokenizer.plane_size, cfg.tokenizer.plane_size),
                                                         mode='bilinear', align_corners=False)
        model.load_state_dict(ckpt, strict=False)

        if reload_mlp:
            first_weight, weights_trunk, biases_trunk, weights_color, biases_color, weights_opacity, biases_opacity = \
                save_model_weights_to_dict(model.decoder, len(model.renderer.lightplane_renderer.n_hidden_trunk)-1)
            pad_color_channels_to_min_block_size = True
            (
                mlp_params,
                n_hidden_trunk,
                n_hidden_opacity,
                n_hidden_color,
            ) = flatten_decoder_params(
                weights_trunk,
                biases_trunk,
                weights_opacity,
                biases_opacity,
                weights_color,
                biases_color,
                pad_color_channels_to_min_block_size,
            )
            pretrained_dict_lp = {
                'renderer.lightplane_renderer.mlp_params': mlp_params,
                'renderer.first_weight': first_weight,
            }
            model_dict = model.state_dict()
            for k, v in pretrained_dict_lp.items():
                if k in model_dict:
                    model_dict[k] = v
            model.load_state_dict(model_dict)
            print('Reloaded MLP')

        return model

    @classmethod
    def from_pretrained_sfmix(
            cls, pretrained_model_name_or_path: str, config_name: str, weight_name1: str, weight_name2: str, reload_mlp: bool
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path1 = os.path.join(weight_name1)
            weight_path2 = os.path.join(pretrained_model_name_or_path, weight_name2)
            use_saved_ckpt = True
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )
            use_saved_ckpt = False

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt1 = load_file(weight_path1, "cpu")
        ckpt2 = torch.load(weight_path2, map_location="cpu")

        if use_saved_ckpt:
            if "module" in list(ckpt1.keys())[0]:
                ckpt1 = {key.replace('module.', ''): item for key, item in ckpt1.items()}
            else:
                ckpt1 = ckpt1
            if "module" in list(ckpt2["state_dict"].keys())[0]:
                ckpt2 = {key.replace('module.', ''): item for key, item in ckpt2["state_dict"].items()}
            else:
                ckpt2 = ckpt2["state_dict"]

        backbone_dict = OrderedDict((k, v) for k, v in ckpt1.items()
                                    if k.startswith('backbone') or k.startswith('image_tokenizer') or
                                    k.startswith('tokenizer') or k.startswith('post_processor'))
        renderer_dict = OrderedDict((k, v) for k, v in ckpt2.items()
                                    if k.startswith('decoder') or k.startswith('renderer'))
        merged_dict = OrderedDict()
        merged_dict.update(backbone_dict)
        merged_dict.update(renderer_dict)

        missing, unexpected = model.load_state_dict(merged_dict, strict=False)
        print('missing:', missing)
        print('unexpected:', unexpected)

        if reload_mlp:
            first_weight, weights_trunk, biases_trunk, weights_color, biases_color, weights_opacity, biases_opacity = \
                save_model_weights_to_dict(model.decoder, len(model.renderer.lightplane_renderer.n_hidden_trunk) - 1)
            pad_color_channels_to_min_block_size = True
            (
                mlp_params,
                n_hidden_trunk,
                n_hidden_opacity,
                n_hidden_color,
            ) = flatten_decoder_params(
                weights_trunk,
                biases_trunk,
                weights_opacity,
                biases_opacity,
                weights_color,
                biases_color,
                pad_color_channels_to_min_block_size,
            )
            pretrained_dict_lp = {
                'renderer.lightplane_renderer.mlp_params': mlp_params,
                'renderer.first_weight': first_weight,
            }
            model_dict = model.state_dict()
            for k, v in pretrained_dict_lp.items():
                if k in model_dict:
                    model_dict[k] = v
            model.load_state_dict(model_dict)
            print('Reloaded MLP')

        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None


    def forward(self, 
                inputs: torch.FloatTensor, 
                rays_o: torch.FloatTensor,
                rays_d: torch.FloatTensor,
                ):
        batch_size, n_views = rays_o.shape[:2]

        with torch.no_grad():
            input_image_tokens: torch.Tensor = self.image_tokenizer(inputs)         # [b,1,c,n]
            input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C').contiguous()
        tokens: torch.Tensor = self.tokenizer(batch_size)                       # [b,ct,Np*Hp*Wp]
        tokens = self.backbone(tokens, encoder_hidden_states=input_image_tokens)# triplanes in [b,Np,Ct,Hp,Wp]
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))    # triplanes in [b,Np,Ct',Hp',Wp']

        scene_codes = rearrange(scene_codes.unsqueeze(1).repeat(1,n_views,1,1,1,1),
                                'b Nv Np Ct Hp Wp -> (b Nv) Np Ct Hp Wp')

        rays_o = rearrange(rays_o, 'b Nv h w c -> (b Nv) h w c')
        rays_d = rearrange(rays_d, 'b Nv h w c -> (b Nv) h w c')
        render_images, render_masks = self.renderer(self.decoder, 
                                                    scene_codes, 
                                                    rays_o, rays_d, 
                                                    return_mask=True)  # [b*Nv,h,w,3], [b*Nv,h,w]


        render_images = rearrange(render_images, '(b Nv) h w c -> b Nv c h w', Nv=n_views)
        render_masks = rearrange(render_masks, '(b Nv) h w c -> b Nv c h w', Nv=n_views)
        
        return {'images_rgb': render_images, 
                'images_weight': render_masks}


    def get_latent_from_img(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(
            device
        )
        batch_size = rgb_cond.shape[0]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens: torch.Tensor = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    def render_360(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)
        return images

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, scene_codes, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            with torch.no_grad():
                color = self.renderer.query_triplane(
                    self.decoder,
                    v_pos,
                    scene_code,
                )["color"]
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy(),
            )
            meshes.append(mesh)
        return meshes


class HumanNOVA(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        default_fovy_deg: float
        default_distance: float

        camera_embedder_cls: str
        camera_embedder: dict

        image_tokenizer_cls: str
        image_tokenizer: dict

        pointformer_cls: str
        pointformer: dict


        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: str, config_name: str, weight_name1: str, weight_name2: str, weight_name3: str,
            reload_mlp: bool
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path1 = os.path.join(weight_name1)
            weight_path2 = os.path.join(pretrained_model_name_or_path, weight_name2)
            weight_path3 = os.path.join(pretrained_model_name_or_path, weight_name3)
            use_saved_ckpt = True
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )
            use_saved_ckpt = False

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt1 = load_file(weight_path1, "cpu")
        ckpt2 = torch.load(weight_path2, map_location="cpu")
        ckpt3 = torch.load(weight_path3, map_location="cpu")

        if use_saved_ckpt:
            if "module" in list(ckpt1.keys())[0]:
                ckpt1 = {key.replace('module.', ''): item for key, item in ckpt1.items()}
            else:
                ckpt1 = ckpt1
            if "module" in list(ckpt2["state_dict"].keys())[0]:
                ckpt2 = {key.replace('module.', ''): item for key, item in ckpt2["state_dict"].items()}
            else:
                ckpt2 = ckpt2["state_dict"]

        if ckpt1['tokenizer.embeddings'].shape[-1] != cfg.tokenizer.plane_size:
            print('Rescale token spatial size to:', cfg.tokenizer.plane_size)
            ckpt1['tokenizer.embeddings'] = F.interpolate(ckpt1['tokenizer.embeddings'],
                                                         size=(cfg.tokenizer.plane_size, cfg.tokenizer.plane_size),
                                                         mode='bilinear', align_corners=False)

        backbone_dict = OrderedDict((k, v) for k, v in ckpt1.items()
                                    if k.startswith('backbone') or k.startswith('image_tokenizer') or
                                    k.startswith('tokenizer') or k.startswith('post_processor') or
                                    k.startswith('camera')
                                    )
        renderer_dict = OrderedDict((k, v) for k, v in ckpt2.items()
                                    if k.startswith('decoder') or k.startswith('renderer'))


        pointformer_dict = OrderedDict((k.replace("module.backbone", "pointformer"), v) for k, v in ckpt3['state_dict'].items() if "dec" not in k)
        del pointformer_dict['module.seg_head.weight']
        del pointformer_dict['module.seg_head.bias']


        merged_dict = OrderedDict()
        merged_dict.update(backbone_dict)
        merged_dict.update(renderer_dict)
        merged_dict.update(pointformer_dict)

        missing, unexpected = model.load_state_dict(merged_dict, strict=False)
        print('missing:', missing)
        print('unexpected:', unexpected)

        if reload_mlp:
            first_weight, weights_trunk, biases_trunk, weights_color, biases_color, weights_opacity, biases_opacity = \
                save_model_weights_to_dict(model.decoder, len(model.renderer.lightplane_renderer.n_hidden_trunk) - 1)
            pad_color_channels_to_min_block_size = True
            (
                mlp_params,
                n_hidden_trunk,
                n_hidden_opacity,
                n_hidden_color,
            ) = flatten_decoder_params(
                weights_trunk,
                biases_trunk,
                weights_opacity,
                biases_opacity,
                weights_color,
                biases_color,
                pad_color_channels_to_min_block_size,
            )
            pretrained_dict_lp = {
                'renderer.lightplane_renderer.mlp_params': mlp_params,
                'renderer.first_weight': first_weight,
            }
            model_dict = model.state_dict()
            for k, v in pretrained_dict_lp.items():
                if k in model_dict:
                    model_dict[k] = v
            model.load_state_dict(model_dict)
            print('Reloaded MLP')

        return model

    @classmethod
    def from_pretrained_object(
            cls, pretrained_model_name_or_path: str, config_name: str, weight_name1: str, weight_name2: str, weight_name3: str,
            reload_mlp: bool
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path1 = os.path.join(weight_name1)
            weight_path2 = os.path.join(pretrained_model_name_or_path, weight_name2)
            weight_path3 = os.path.join(pretrained_model_name_or_path, weight_name3)
            use_saved_ckpt = True
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )
            use_saved_ckpt = False

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt1 = load_file(weight_path1, "cpu")
        ckpt2 = torch.load(weight_path2, map_location="cpu")
        ckpt3 = torch.load(weight_path3, map_location="cpu")

        if use_saved_ckpt:
            if "module" in list(ckpt1.keys())[0]:
                ckpt1 = {key.replace('module.', ''): item for key, item in ckpt1.items()}
            else:
                ckpt1 = ckpt1
            if "module" in list(ckpt2["state_dict"].keys())[0]:
                ckpt2 = {key.replace('module.', ''): item for key, item in ckpt2["state_dict"].items()}
            else:
                ckpt2 = ckpt2["state_dict"]

        if ckpt1['tokenizer.embeddings'].shape[-1] != cfg.tokenizer.plane_size:
            print('Rescale token spatial size to:', cfg.tokenizer.plane_size)
            ckpt1['tokenizer.embeddings'] = F.interpolate(ckpt1['tokenizer.embeddings'],
                                                         size=(cfg.tokenizer.plane_size, cfg.tokenizer.plane_size),
                                                         mode='bilinear', align_corners=False)

        backbone_dict = OrderedDict((k, v) for k, v in ckpt1.items()
                                    if k.startswith('backbone') or k.startswith('image_tokenizer') or
                                    k.startswith('tokenizer') or k.startswith('post_processor') or
                                    k.startswith('camera')
                                    )
        renderer_dict = OrderedDict((k, v) for k, v in ckpt2.items()
                                    if k.startswith('decoder') or k.startswith('renderer'))

        pointformer_dict = OrderedDict((k.replace("module.", "pointformer."), v) for k, v in ckpt3['state_dict'].items() if "dec" not in k)
        merged_dict = OrderedDict()
        merged_dict.update(backbone_dict)
        merged_dict.update(renderer_dict)
        merged_dict.update(pointformer_dict)

        missing, unexpected = model.load_state_dict(merged_dict, strict=False)
        print('missing:', missing)
        print('unexpected:', unexpected)

        if reload_mlp:
            first_weight, weights_trunk, biases_trunk, weights_color, biases_color, weights_opacity, biases_opacity = \
                save_model_weights_to_dict(model.decoder, len(model.renderer.lightplane_renderer.n_hidden_trunk) - 1)
            pad_color_channels_to_min_block_size = True
            (
                mlp_params,
                n_hidden_trunk,
                n_hidden_opacity,
                n_hidden_color,
            ) = flatten_decoder_params(
                weights_trunk,
                biases_trunk,
                weights_opacity,
                biases_opacity,
                weights_color,
                biases_color,
                pad_color_channels_to_min_block_size,
            )
            pretrained_dict_lp = {
                'renderer.lightplane_renderer.mlp_params': mlp_params,
                'renderer.first_weight': first_weight,
            }
            model_dict = model.state_dict()
            for k, v in pretrained_dict_lp.items():
                if k in model_dict:
                    model_dict[k] = v
            model.load_state_dict(model_dict)
            print('Reloaded MLP')

        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.pointformer = find_class(self.cfg.pointformer_cls)(**self.cfg.pointformer)
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.camera_embedder = find_class(self.cfg.camera_embedder_cls)(self.cfg.camera_embedder)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None


        self.c2w_cond = default_cond_c2w(self.cfg.default_distance)
        self.intrinsic, self.intrinsic_normed_cond = create_intrinsic_from_fov_deg(
            self.cfg.default_fovy_deg,
            self.cfg.cond_image_size,
            self.cfg.cond_image_size,
        )

    def forward(self,
                inputs: torch.FloatTensor,
                rays_o: torch.FloatTensor,
                rays_d: torch.FloatTensor,
                smplx_v_align: torch.FloatTensor,
                ):
        batch_size, n_views = rays_o.shape[:2]

        batch = {
            "c2w_cond": self.c2w_cond.view(1, 1, 4, 4).to('cuda').repeat(batch_size, 1, 1, 1),
            "intrinsic_cond": self.intrinsic.to('cuda').view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
            "intrinsic_normed_cond": self.intrinsic_normed_cond.to('cuda').view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
        }
        with torch.no_grad():
            camera_embeds = self.camera_embedder(**batch)
            input_image_tokens: torch.Tensor = self.image_tokenizer(inputs, modulation_cond=camera_embeds)  # [b,1,c,n]
            input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C').contiguous()
        if smplx_v_align.shape[1] == 1:
            smplx_v_align_tmp = torch.from_numpy(0.5*np.random.rand(8000,6).astype(np.float32)).cuda()[None]
            input_point_tokens = self.pointformer(smplx_v_align_tmp.squeeze(0))
        else:
            input_point_tokens = self.pointformer(smplx_v_align.squeeze(0))

        if smplx_v_align.shape[1] == 1:
            encoder_hidden_states = torch.cat((input_image_tokens,
                                               torch.zeros_like(input_point_tokens).cuda() * input_point_tokens), dim=1)
        else:
            encoder_hidden_states = torch.cat((input_image_tokens, input_point_tokens), dim=1)
        tokens: torch.Tensor = self.tokenizer(batch_size)  # [b,ct,Np*Hp*Wp]
        tokens = self.backbone(tokens, encoder_hidden_states=encoder_hidden_states, modulation_cond=None) # triplanes in [b,Np,Ct,Hp,Wp]
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))  # triplanes in [b,Np,Ct',Hp',Wp']
        scene_codes = rearrange(scene_codes.unsqueeze(1).repeat(1, n_views, 1, 1, 1, 1),
                                'b Nv Np Ct Hp Wp -> (b Nv) Np Ct Hp Wp')

        rays_o = rearrange(rays_o, 'b Nv h w c -> (b Nv) h w c')
        rays_d = rearrange(rays_d, 'b Nv h w c -> (b Nv) h w c')
        render_images, render_masks = self.renderer(self.decoder,
                                                    scene_codes,
                                                    rays_o, rays_d,
                                                    return_mask=True)  # [b*Nv,h,w,3], [b*Nv,h,w]

        render_images = rearrange(render_images, '(b Nv) h w c -> b Nv c h w', Nv=n_views)
        render_masks = rearrange(render_masks, '(b Nv) h w c -> b Nv c h w', Nv=n_views)

        return {'images_rgb': render_images,
                'images_weight': render_masks}

    def get_latent_from_img(
            self,
            image: Union[
                PIL.Image.Image,
                np.ndarray,
                torch.FloatTensor,
                List[PIL.Image.Image],
                List[np.ndarray],
                List[torch.FloatTensor],
            ],
            smplx_v_align: torch.FloatTensor,
            device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(device)
        batch_size = rgb_cond.shape[0]

        batch = {
            "c2w_cond": self.c2w_cond.view(1, 1, 4, 4).to('cuda').repeat(batch_size, 1, 1, 1),
            "intrinsic_cond": self.intrinsic.to('cuda').view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
            "intrinsic_normed_cond": self.intrinsic_normed_cond.to('cuda').view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1),
        }
        if smplx_v_align.shape[1] != 1:
            with torch.no_grad():
                input_point_tokens = self.pointformer(smplx_v_align.squeeze(0))
        with autocast('cuda', enabled=True, dtype=torch.float16):
            with torch.no_grad():
                camera_embeds = self.camera_embedder(**batch)
                input_image_tokens: torch.Tensor = self.image_tokenizer(rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1), modulation_cond=camera_embeds)  # [b,1,c,n]
                input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C').contiguous()
            if smplx_v_align.shape[1] != 1:
                encoder_hidden_states = torch.cat((input_image_tokens, input_point_tokens), dim=1)
            else:
                encoder_hidden_states = input_image_tokens

            tokens: torch.Tensor = self.tokenizer(batch_size)  # [b,ct,Np*Hp*Wp]
            tokens = self.backbone(tokens, encoder_hidden_states=encoder_hidden_states, modulation_cond=None) # triplanes in [b,Np,Ct,Hp,Wp]
            scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))  # triplanes in [b,Np,Ct',Hp',Wp']

        return scene_codes

    def render_360(
            self,
            scene_codes,
            n_views: int,
            elevation_deg: float = 0.0,
            camera_distance: float = 1.9,
            fovy_deg: float = 40.0,
            height: int = 256,
            width: int = 256,
            return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)
        return images

    def set_marching_cubes_resolution(self, resolution: int):
        if (
                self.isosurface_helper is not None
                and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(self, scene_codes, resolution: int = 256, threshold: float = 25.0):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            with torch.no_grad():
                color = self.renderer.query_triplane(
                    self.decoder,
                    v_pos,
                    scene_code,
                )["color"]
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy(),
            )
            meshes.append(mesh)
        return meshes


# Backward-compatible alias for older code paths.
SF_TSR = HumanNOVA
