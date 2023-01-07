import torch
from torch import nn
from torch.nn import functional as F

from hit.modeling import registry
from hit.modeling.poolers import make_3d_pooler
from hit.modeling.roi_heads.action_head.hit_structure import make_hit_structure
from hit.modeling.utils import cat, pad_sequence, prepare_pooled_feature
from hit.utils.IA_helper import has_object, has_hand
from hit.structures.bounding_box import BoxList

from hit.modeling.roi_heads.action_head.pose_transformer import PoseTransformer



@registry.ROI_ACTION_FEATURE_EXTRACTORS.register("2MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    def __init__(self, config, dim_in):
        super(MLPFeatureExtractor, self).__init__()
        self.config = config
        head_cfg = config.MODEL.ROI_ACTION_HEAD

        self.pooler = make_3d_pooler(head_cfg)

        resolution = head_cfg.POOLER_RESOLUTION

        self.max_pooler = nn.MaxPool3d((1, resolution, resolution))
        self.pose_out = config.MODEL.HIT_STRUCTURE.DIM_INNER
        if config.MODEL.HIT_STRUCTURE.ACTIVE:
            self.max_feature_len_per_sec = config.MODEL.HIT_STRUCTURE.MAX_PER_SEC
            self.hit_structure = make_hit_structure(config, dim_in)
            self.hit_structure_pose = make_hit_structure(config, self.pose_out)

        representation_size = head_cfg.MLP_HEAD_DIM

        fc1_dim_in = dim_in
        if config.MODEL.HIT_STRUCTURE.ACTIVE and (config.MODEL.HIT_STRUCTURE.FUSION == "concat"):
            fc1_dim_in += config.MODEL.HIT_STRUCTURE.DIM_OUT

        self.fc1 = nn.Linear(fc1_dim_in, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        self.dim_out = representation_size
        self.pose_transformer = PoseTransformer()

    def roi_pooling(self, slow_features, fast_features, proposals):
        if slow_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_features = slow_features.mean(dim=2, keepdim=True)
            slow_x = self.pooler(slow_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_x = slow_x.mean(dim=2, keepdim=True)
            x = slow_x
        if fast_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_features = fast_features.mean(dim=2, keepdim=True)
            fast_x = self.pooler(fast_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_x = fast_x.mean(dim=2, keepdim=True)
            x = fast_x

        if slow_features is not None and fast_features is not None:
            x = torch.cat([slow_x, fast_x], dim=1)
        return x

    def max_pooling_zero_safe(self, x):
        if x.size(0) == 0:
            _, c, t, h, w = x.size()
            res = self.config.MODEL.ROI_ACTION_HEAD.POOLER_RESOLUTION
            x = torch.zeros((0, c, 1, h - res + 1, w - res + 1), device=x.device)
        else:
            x = self.max_pooler(x)
        return x

    def forward(self, slow_features, fast_features, proposals, objects=None, keypoints=None, extras={}, part_forward=-1):
        ia_active = hasattr(self, "hit_structure")
        if part_forward == 1:
            person_pooled = cat([box.get_field("pooled_feature") for box in proposals])
            if objects is None:
                object_pooled = None
            else:
                object_pooled = cat([box.get_field("pooled_feature") for box in objects])
                
            if keypoints is None:
                hands_pooled = None
            else:
                hands_pooled = cat([box.get_field("pooled_feature") for box in keypoints[0]])
                pose_out = cat([box.get_field("pooled_feature") for box in keypoints[1]])
                keypoints = keypoints[0]
        else:
            x = self.roi_pooling(slow_features, fast_features, proposals)
            person_pooled = self.max_pooler(x)

            if has_object(self.config.MODEL.HIT_STRUCTURE):
                object_pooled = self.roi_pooling(slow_features, fast_features, objects)
                object_pooled = self.max_pooling_zero_safe(object_pooled)
            else:
                object_pooled = None
            hand_boxlists = []

            if has_hand(self.config.MODEL.HIT_STRUCTURE):    
                for k in keypoints:
                    if 'keypoints' in k.extra_fields:
                        kk = torch.flatten(k.extra_fields['keypoints'], start_dim=1)[:, 18:22]
                        hand_boxlists.append(BoxList(kk, k.size, mode="xyxy"))
                    else:
                        hand_boxlists.append(BoxList(torch.zeros((0, 4), dtype=k.bbox.dtype, device=k.bbox.device), k.size, mode="xyxy"))


                    # extend width and height by 50%
                    proposals_hand = [box.extend((0.2, 0.8)) for box in hand_boxlists]
                    hands_pooled = self.roi_pooling(slow_features, fast_features, proposals_hand)
                    hands_pooled = self.max_pooling_zero_safe(hands_pooled)
            else:
                hands_pooled = None
            
            # Pose  start     
            pose_data = torch.cat([k.extra_fields['keypoints'].to(k.bbox.device) for k in keypoints if 'keypoints' in k.extra_fields], dim=0).unsqueeze(1)
           
            if pose_data.shape[0] == person_pooled.shape[0]:
                self.pose_transformer = self.pose_transformer.to(keypoints[0].bbox.device)
                pose_out = self.pose_transformer(pose_data)
                pose_out = pose_out.view(-1, self.pose_out).unsqueeze(2).unsqueeze(2).unsqueeze(2)
                
            else:
                pose_out = None
            # Pose end
                
        if part_forward == 0:
            return None, person_pooled, object_pooled, hands_pooled, pose_out

        x_after = person_pooled
        if ia_active:
            tsfmr = self.hit_structure
            mem_len = self.config.MODEL.HIT_STRUCTURE.LENGTH
            mem_rate = self.config.MODEL.HIT_STRUCTURE.MEMORY_RATE
            use_penalty = self.config.MODEL.HIT_STRUCTURE.PENALTY
            memory_person, memory_person_boxes = self.get_memory_feature(extras["person_pool"], extras, mem_len, mem_rate,
                                                                       self.max_feature_len_per_sec, tsfmr.dim_others,
                                                                       person_pooled, proposals, use_penalty)
            # RGB stream
            ia_feature, res_person, res_object, res_keypoint = self.hit_structure(person_pooled, proposals, object_pooled, objects, hands_pooled, keypoints, memory_person, None, None, phase="rgb")
            # pose
            pose_ia_feature = self.hit_structure_pose(pose_out, proposals, res_object, objects, res_keypoint, keypoints, memory_person, res_person, ia_feature, phase="pose")
            x_after = self.fusion(x_after, pose_ia_feature, self.config.MODEL.HIT_STRUCTURE.FUSION)
        x_after = x_after.view(x_after.size(0), -1)
        
        x_after = F.relu(self.fc1(x_after))
        x_after = F.relu(self.fc2(x_after))

        return x_after, person_pooled, object_pooled, hands_pooled, pose_out

    def get_memory_feature(self, feature_pool, extras, mem_len, mem_rate, max_boxes, fixed_dim, current_x, current_box, use_penalty):
        before, after = mem_len
        mem_feature_list = []
        mem_pos_list = []
        device = current_x.device
        if use_penalty and self.training:
            cur_loss = extras["cur_loss"]
        else:
            cur_loss = 0.0
        current_feat = prepare_pooled_feature(current_x, current_box, detach=True)
        for movie_id, timestamp, new_feat in zip(extras["movie_ids"], extras["timestamps"], current_feat):
            before_inds = range(timestamp - before * mem_rate, timestamp, mem_rate)
            after_inds = range(timestamp + mem_rate, timestamp + (after + 1) * mem_rate, mem_rate)
            cache_cur_mov = feature_pool[movie_id]
            mem_box_list_before = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes, cur_loss, use_penalty)
                                   for mem_ind in before_inds]
            mem_box_list_after = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes, cur_loss, use_penalty)
                                  for mem_ind in after_inds]
            mem_box_current = [self.sample_mem_feature(new_feat, max_boxes), ]
            mem_box_list = mem_box_list_before + mem_box_current + mem_box_list_after
            mem_feature_list += [box_list.get_field("pooled_feature")
                                 if box_list is not None
                                 else torch.zeros(0, fixed_dim, 1, 1, 1, dtype=torch.float32, device="cuda")
                                 for box_list in mem_box_list]
            mem_pos_list += [box_list.bbox
                             if box_list is not None
                             else torch.zeros(0, 4, dtype=torch.float32, device="cuda")
                             for box_list in mem_box_list]

        seq_length = sum(mem_len) + 1
        person_per_seq = seq_length * max_boxes
        mem_feature = pad_sequence(mem_feature_list, max_boxes)
        mem_feature = mem_feature.view(-1, person_per_seq, fixed_dim, 1, 1, 1)
        mem_feature = mem_feature.to(device)
        mem_pos = pad_sequence(mem_pos_list, max_boxes)
        mem_pos = mem_pos.view(-1, person_per_seq, 4)
        mem_pos = mem_pos.to(device)

        return mem_feature, mem_pos

    def check_fetch_mem_feature(self, movie_cache, mem_ind, max_num, cur_loss, use_penalty):
        if mem_ind not in movie_cache:
            return None
        box_list = movie_cache[mem_ind]
        box_list = self.sample_mem_feature(box_list, max_num)
        if use_penalty and self.training:
            loss_tag = box_list.delete_field("loss_tag")
            penalty = loss_tag / cur_loss if loss_tag < cur_loss else cur_loss / loss_tag
            features = box_list.get_field("pooled_feature") * penalty
            box_list.add_field("pooled_feature", features)
        return box_list

    def sample_mem_feature(self, box_list, max_num):
        if len(box_list) > max_num:
            idx = torch.randperm(len(box_list))[:max_num]
            return box_list[idx].to("cuda")
        else:
            return box_list.to("cuda")

    def fusion(self, x, out, type="add"):
        if type == "add":
            return x + out
        elif type == "concat":
            return torch.cat([x, out], dim=1)
        else:
            raise NotImplementedError


def make_roi_action_feature_extractor(cfg, dim_in):
    func = registry.ROI_ACTION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, dim_in)
