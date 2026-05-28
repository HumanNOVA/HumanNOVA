import argparse
import os
import pickle
import warnings
import numpy as np
import rembg
import torch
from PIL import Image
import trimesh
from visualenc.base.utils import remove_background, resize_foreground, save_video
from visualenc.base.system import HumanNOVA
from torch.amp import autocast
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SMPL_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoint", "SMPL_NEUTRAL.pkl")
warnings.filterwarnings("ignore")


def print_stage(title: str) -> None:
    print(f"\n[{title}]")


def print_status(message: str) -> None:
    print(f"  - {message}")


parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, help="Path to input image(s).")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="./checkpoint",
    type=str,
    help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'",
)
parser.add_argument(
    "--chunk-size",
    default=291520,
    type=int,
    help="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 8192",
)
parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.8,
    type=float,
    help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="output_demo/",
    type=str,
    help="Output directory to save the results. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)
parser.add_argument(
    "--render",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)
parser.add_argument(
    "--render-num-views",
    default=30,
    type=int,
    help="Number of views to render. Default: 30",
)
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

print_stage("HumanNOVA")
print_status(f"Input folder: {args.image}")
print_status(f"Output folder: {output_dir}")
print_status(f"Device: {device}")

print_stage("Initializing Model")
if os.path.isdir(args.pretrained_model_name_or_path):
    config_path = os.path.join(
        args.pretrained_model_name_or_path, "config_humannova.yaml"
    )
else:
    config_path = hf_hub_download(
        repo_id=args.pretrained_model_name_or_path,
        filename="config_humannova.yaml",
    )

cfg = OmegaConf.load(config_path)
OmegaConf.resolve(cfg)
model = HumanNOVA(cfg).to(device)

pretrain_dir = os.path.join(PROJECT_ROOT, "checkpoint", "ckpt_best.pth.tar")
print_status(f"Loading checkpoint: {pretrain_dir}")
if device is not None:
    checkpoint = torch.load(pretrain_dir, map_location=device)
else:
    checkpoint = torch.load(pretrain_dir, map_location=torch.device('cpu'))
if "module" in list(checkpoint["state_dict"].keys())[0]:
    state_dict = {key.replace('module.', ''): item for key, item in checkpoint["state_dict"].items()}
else:
    state_dict = checkpoint["state_dict"]
model.load_state_dict(state_dict, strict=True)
print_status("Model ready")

model.renderer.set_chunk_size(args.chunk_size)
model.to(device)

images = []
image_list = sorted(os.listdir(args.image))
total_images = len(image_list)
print_stage("Processing Images")
print_status(f"Found {total_images} image(s)")

if args.no_remove_bg:
    rembg_session = None
    print_status("Background removal: disabled")
else:
    rembg_session = rembg.new_session()
    print_status("Background removal: enabled")

for image_index, image_name in enumerate(image_list, start=1):
    print(f"\n[{image_index}/{total_images}] {image_name}")
    image_path = os.path.join(args.image, image_name)
    sample_output_dir = os.path.join(output_dir, image_name.split('.')[0])
    if args.no_remove_bg:
        image = Image.open(image_path)
        image = image.resize((512, 512))
        if not os.path.exists(sample_output_dir):
            os.makedirs(sample_output_dir)
        input_path = os.path.join(sample_output_dir, "input.png")
        image.save(input_path)
    else:
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        if not os.path.exists(sample_output_dir):
            os.makedirs(sample_output_dir)
        input_path = os.path.join(sample_output_dir, "input.png")
        image.save(input_path)

    from hmr.models.smpl_wrapper import SMPL
    body_model_smpl = SMPL(model_path=SMPL_MODEL_PATH)
    with open(os.path.join(os.path.dirname(args.image), 'smpl_est.pkl'), 'rb') as f:
        es_smpl_dict = pickle.load(f)
    smplx_v_align = es_smpl_dict[image_name]['out']['pred_vertices'][0]
    all_cam_t = es_smpl_dict[image_name]['final']['all_cam_t'][0].copy()
    all_cam_t[2] = 0
    smplx_v_align = smplx_v_align.cpu().numpy() + all_cam_t
    smplx_normal_align = trimesh.Trimesh(vertices=smplx_v_align, faces=body_model_smpl.faces).vertex_normals
    smplx_final = np.concatenate([smplx_v_align, smplx_normal_align.astype(smplx_v_align.dtype)], axis=-1)

    with torch.no_grad():
        scene_codes = model.get_latent_from_img([image], torch.from_numpy(smplx_final[None]).cuda(), device=device)

    with autocast('cuda', enabled=True, dtype=torch.float16):
        if args.render:
            render_images = model.render_360(scene_codes, n_views=args.render_num_views, camera_distance=2.8,
                                             return_type="pil", height=512, width=512)
            for ri, render_image in enumerate(render_images[0]):
                render_path = os.path.join(sample_output_dir, f"render_{ri:03d}.png")
                render_image.save(render_path)
            render_video_path = os.path.join(sample_output_dir, "render.mp4")
            save_video(
                render_images[0], render_video_path, fps=15
            )
    print_status(f"Saved results to {sample_output_dir}")
