import numpy as np
from pycocotools import mask as mask_util
import torch


def segment_overlap(mask, segments, config):
    segment_rles = [s['segmentation'] for s in segments]
    segment_areas = [s['area'] for s in segments]
    #segment_bbox = [s['bbox'] for s in segments]
    #orin_mask = mask.detach().cpu()

    mask = np.uint8(1) - np.copy(mask.detach().cpu(), order='F').astype(np.uint8)
    #orin_mask = mask.copy()

    mask_rle = mask_util.encode(mask)
    intersections = [mask_util.merge([mask_rle, segment_rle], intersect=1) for segment_rle in segment_rles]
    areas_overlaps = [mask_util.area(rle) / seg_area for (rle, seg_area) in zip(intersections, segment_areas)]

    #selected_seg = [seg for (seg, overlap) in zip(segment_rles, areas_overlaps) if overlap >= config['seg_overlap']]
    selected_seg = []
    for (seg, inter, overlap) in zip(segment_rles, intersections, areas_overlaps):
        if overlap >= config['seg_overlap']:
            selected_seg.append(seg)
        else:
            # x_min, y_min, width, height = inter
            # mask_pad = np.pad(orin_mask[int(x_min):int(x_min+width), int(y_min):int(y_min+height)], ((1, 1), (1, 1)), mode='constant', constant_values=0)
            # #mask_sliced = np.uint8(1) - np.copy(mask_pad.detach().cpu(), order='F').astype(np.uint8)
            # mask_sliced_rle = mask_util.encode(np.asfortranarray(mask_pad))
            # selected_seg.append(mask_sliced_rle)
            selected_seg.append(inter)

    if len(selected_seg) > 0:
        return 1. - torch.tensor(mask_util.decode(mask_util.merge(selected_seg)))
    else:
        return torch.ones(mask.shape)
