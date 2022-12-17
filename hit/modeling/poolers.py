import torch
from torch import nn

from hit.layers import ROIAlign3d, ROIPool3d


class Pooler3d(nn.Module):
    def __init__(self, output_size, scale, sampling_ratio=None, pooler_type='align3d'):
        super(Pooler3d, self).__init__()
        if pooler_type == 'align3d':
            assert sampling_ratio is not None, 'Sampling ratio should be specified for 3d roi align.'
            self.pooler = ROIAlign3d(
                output_size, spatial_scale=scale, sampling_ratio=sampling_ratio
            )
        elif pooler_type == 'pooling3d':
            self.pooler = ROIPool3d(
                output_size, spatial_scale=scale
            )
        self.output_size = output_size

    def convert_to_roi_format(self, boxes, dtype, device):
        bbox_list = list()
        ids_list = list()
        for i, b in enumerate(boxes):
            # if "keypoints" not in b.extra_fields:
            if not b:
                bbox_list.append(torch.zeros((0, 4), dtype=dtype, device=device))
                ids_list.append(torch.zeros((0, 1), dtype=dtype, device=device))
            else:
                b.bbox = b.bbox.to('cuda')
                bbox_list.append(b.bbox)
                ids_list.append(torch.full((len(b), 1), i, dtype=dtype, device=device))
            # if "keypoints" in b.extra_fields:
            #     kpts_flatten = torch.flatten(b.extra_fields["keypoints"], start_dim=1)
            #     # kpts_flatten = kpts_flatten.to('cuda')
            #     kpts_hands_bbox = kpts_flatten
            #     bbox_list.append(torch.flatten(b.extra_fields["keypoints"], start_dim=1))
            #     ids_list.append(torch.full((len(b), 1), i, dtype=dtype, device=device))
        concat_boxes = torch.cat(bbox_list, dim=0)
        ids = torch.cat(ids_list, dim=0)
        rois = torch.cat([ids, concat_boxes], dim=1)

        return rois

    def forward(self, x, boxes):
        rois = self.convert_to_roi_format(boxes, x.dtype, x.device)
        return self.pooler(x, rois)


def make_3d_pooler(head_cfg):
    resolution = head_cfg.POOLER_RESOLUTION
    scale = head_cfg.POOLER_SCALE
    sampling_ratio = head_cfg.POOLER_SAMPLING_RATIO
    pooler_type = head_cfg.POOLER_TYPE
    pooler = Pooler3d(
        output_size=(resolution, resolution),
        scale=scale,
        sampling_ratio=sampling_ratio,
        pooler_type=pooler_type,
    )
    return pooler
