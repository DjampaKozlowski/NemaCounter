import torch
import cv2
import os
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry, SamPredictor

import nemacounter.utils as utils
import nemacounter.common as common


class NemaCounterSegmentation:

    def __init__(self, checkpoint_path, model_type='vit_h', device='cpu'):
        ## Load the SAM model
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def objects_segmentation(self, image, boxes):
        img = image.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(img)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            torch.tensor(boxes, device=self.predictor.device),
            img.shape[:2])
        masks, _, _ = self.predictor.predict_torch(point_coords=None,
                                                   point_labels=None,
                                                   boxes=transformed_boxes,
                                                   multimask_output=False)
        masks = masks.cpu().numpy().astype('uint8').squeeze()
        return masks


def add_masks_on_image(masks, img):
    mask = np.sum(masks, axis=0)
    img[mask == 1] = [0, 0, 255]  # Change green to red


def create_multicolored_masks_image(masks):
    if masks.ndim != 3:
        raise ValueError("Expected a 3D array of masks")

    # Create a black background image
    black_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)

    # Loop over each mask layer
    for i in range(masks.shape[0]):
        color = np.random.randint(100, 256, size=3)  # Generate a bright color
        mask_layer = masks[i, :, :]  # Select the i-th mask layer
        # Apply color to pixels where mask_layer is 1
        for c in range(3):  # Apply to each color channel
            black_image[:, :, c][mask_layer == 1] = color[c]

    return black_image



def segmentation_workflow(dct_args):
    dct_args['project_id'] = os.path.basename(dct_args['input_file']).replace('_globinfo.csv', '')
    dct_args['input_dir'] = os.path.dirname(dct_args['input_file'])

    gpu_if_avail = utils.get_bool(dct_args['gpu'])
    add_overlay = utils.get_bool(dct_args['add_overlay'])
    utils.set_cpu_usage(dct_args['cpu'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and gpu_if_avail else 'cpu')

    if add_overlay:
        dpath_overlay = os.path.join(dct_args['input_dir'], dct_args['project_id'], 'img', 'segmentation')
        os.makedirs(dpath_overlay, exist_ok=True)

    if utils.check_file_existence(dct_args['input_file']):
        df = pd.read_csv(dct_args['input_file'])
        lst_img_paths = df['img_id'].unique()
        segmentation_model = NemaCounterSegmentation(dct_args['segany'], device=device)
        df['surface'] = np.nan

        for img_path in lst_img_paths:
            img = common.read_image(img_path)
            boxes = common.create_boxes(df[df['img_id'] == img_path])
            masks = segmentation_model.objects_segmentation(img, boxes)
            df.loc[df['img_id'] == img_path, 'surface'] = np.sum(np.sum(masks, axis=1), axis=1)

            if add_overlay:
                add_masks_on_image(masks, img)
                fpath_out_img = os.path.join(dpath_overlay, f"{dct_args['project_id']}_{os.path.basename(img_path)}")
                cv2.imwrite(fpath_out_img, img)

                multicolored_img = create_multicolored_masks_image(masks)
                fpath_out_multi = os.path.join(dpath_overlay,
                                               f"{dct_args['project_id']}_{os.path.basename(img_path)}_colored.png")
                cv2.imwrite(fpath_out_multi, multicolored_img)

        df.to_csv(dct_args['input_file'], index=False)

        summary_fpath = os.path.join(dct_args['input_dir'], f"{dct_args['project_id']}_summary.csv")
        df_summary_original = pd.read_csv(summary_fpath)
        df_summary_new = common.create_summary_table(df, dct_args['project_id'])
        df_summary = pd.merge(df_summary_original, df_summary_new[['img_id', 'surface_mean', 'surface_std']],
                              on='img_id')
        df_summary.to_csv(summary_fpath, index=False)
