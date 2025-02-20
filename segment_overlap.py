import numpy as np
from pycocotools import mask as mask_util
import torch


def segment_overlap(mask, segments, config):
    segment_rles = [s['segmentation'] for s in segments]
    segment_areas = [s['area'] for s in segments]

    mask = np.uint8(1) - np.copy(mask.detach().cpu(), order='F').astype(np.uint8)

    mask_rle = mask_util.encode(mask)
    intersections = [mask_util.merge([mask_rle, segment_rle], intersect=1) for segment_rle in segment_rles]
    areas_overlaps = [mask_util.area(rle) / seg_area for (rle, seg_area) in zip(intersections, segment_areas)]

    selected_seg = [seg for (seg, overlap) in zip(segment_rles, areas_overlaps) if overlap >= config['seg_overlap']]
    # 修改覆盖率 看效果

    if len(selected_seg) > 0:
        return 1. - torch.tensor(mask_util.decode(mask_util.merge(selected_seg)))
    else:
        return torch.ones(mask.shape)
