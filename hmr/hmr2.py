import pickle
from pathlib import Path
import torch
import argparse
import os
import cv2
import math
import numpy as np
from tqdm import tqdm

from hmr.configs import CACHE_DIR_4DHUMANS
from hmr.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr.utils import recursive_to
from hmr.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def main():
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default=None, help='Output folder for optional rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--cur_num', type=int, default=0, help='')
    parser.add_argument('--split_num', type=int, default=1, help='')

    args = parser.parse_args()

    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    from hmr.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hmr
        cfg_path = Path(hmr.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    needs_output_dir = args.save_mesh or args.full_frame
    if needs_output_dir and args.out_folder is None:
        raise ValueError("--out_folder is required when using --save_mesh or --full_frame")
    if args.out_folder is not None:
        args.out_folder = os.path.abspath(args.out_folder)
    if needs_output_dir:
        os.makedirs(args.out_folder, exist_ok=True)

    img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])

    out_dict = {}

    total_number = len(img_paths)
    total_split_num = args.split_num
    cur_num = args.cur_num
    chunk_num = math.ceil(total_number/total_split_num)

    idx = -1
    for img_path in tqdm(img_paths):
        idx += 1
        if idx >= chunk_num * cur_num and idx < chunk_num * (cur_num+1):
            pass
        else:
            continue

        img_cv2 = cv2.imread(str(img_path))

        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        if boxes.shape[0] == 0:
            print(img_path)
            boxes = np.array([[0., 0., 1024., 1024.]])

        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)

            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

        out_dict[str(img_path).split('/')[-1]] = {
            'out': out,
            'final': {
                'scaled_focal_length': scaled_focal_length,
                'all_cam_t': all_cam_t,
            }
        }

    output_pkl = os.path.join(os.path.dirname(args.img_folder), 'smpl_est.pkl')
    with open(output_pkl, 'wb') as f:
        pickle.dump(out_dict, f)
    print(output_pkl)

if __name__ == '__main__':
    main()
