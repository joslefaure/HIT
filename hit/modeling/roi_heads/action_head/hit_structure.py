from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import torch
import torch.nn as nn
from hit.modeling import registry
from hit.utils.IA_helper import has_memory, has_person, has_object, has_hand

# Non-local helper, used for Attentive Fusion
class NL(nn.Module):
    def __init__(self, hidden_dim=512, out_dim=1024, kernel_size=3):
        super(NL, self).__init__()

        self.hidden_dim = hidden_dim

        self.conv_q = nn.Conv3d(hidden_dim, hidden_dim, 1, padding=0, bias=False)
        self.conv_k = nn.Conv3d(hidden_dim, hidden_dim, 1, padding=0, bias=False)
        self.conv_v = nn.Conv3d(hidden_dim, hidden_dim, 1, padding=0, bias=False)
        
        self.conv = nn.Conv3d(hidden_dim, hidden_dim, 1, padding=0, bias=False)
        self.conv_out = nn.Conv3d(hidden_dim, out_dim, 1, padding=0, bias=False)
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)

        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)
        
        x = x + virt_feats
        x = self.conv_out(x)
        return x

class InteractionUnit(nn.Module):
    def __init__(self, dim_person, dim_other, dim_out, dim_inner, structure_config,
                 max_others=20,
                 temp_pos_len=-1,  # configs for memory feature
                 dropout=0.):
        super(InteractionUnit, self).__init__()
        self.dim_person = dim_person
        self.dim_other = dim_other
        self.dim_out = dim_out
        self.dim_inner = dim_inner
        self.max_others = max_others
        self.scale_value = dim_inner ** (-0.5)
        # config for temporal position, only used for temporal interaction,
        self.temp_pos_len = temp_pos_len

        bias = not structure_config.NO_BIAS
        init_std = structure_config.CONV_INIT_STD

        self.query = nn.Conv3d(dim_person, dim_inner, 1, bias)
        init_layer(self.query, init_std, bias)

        self.key = nn.Conv3d(dim_other, dim_inner, 1, bias)
        init_layer(self.key, init_std, bias)

        self.value = nn.Conv3d(dim_other, dim_inner, 1, bias)
        init_layer(self.value, init_std, bias)

        self.out = nn.Conv3d(dim_inner, dim_out, 1, bias)
        if structure_config.USE_ZERO_INIT_CONV:
            out_init = 0
        else:
            out_init = init_std
        init_layer(self.out, out_init, bias)

        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.use_ln = structure_config.LAYER_NORM

        if dim_person != dim_out:
            self.shortcut = nn.Conv3d(dim_person, dim_out, 1, bias)
            init_layer(self.shortcut, init_std, bias)
        else:
            self.shortcut = None

        if self.temp_pos_len > 0:
            self.temporal_position_k = nn.Parameter(torch.zeros(temp_pos_len, 1, self.dim_inner, 1, 1, 1))
            self.temporal_position_v = nn.Parameter(torch.zeros(temp_pos_len, 1, self.dim_inner, 1, 1, 1))

    def forward(self, person, others):
        """
        :param person: [n, channels, t, h, w]
        :param others: [n, num_other, channels, t, h, w]
        """
        device = person.device
        n, dim_person, t, h, w = person.size()
        _, max_others, dim_others, t_others, h_others, w_others = others.size()

        query_batch = person
        key = fuse_batch_num(others)  # [n*num_other, channels, t, h, w]

        query_batch = self.query(query_batch)
        key_batch = self.key(key).contiguous().view(n, self.max_others, self.dim_inner, t_others, h_others, w_others)
        value_batch = self.value(key).contiguous().view(n, self.max_others, self.dim_inner, t_others, h_others,
                                                        w_others)

        if self.temp_pos_len > 0:
            max_person_per_sec = max_others // self.temp_pos_len
            key_batch = key_batch.contiguous().view(n, self.temp_pos_len, max_person_per_sec, self.dim_inner, t_others,
                                                    h_others,
                                                    w_others)
            key_batch = key_batch + self.temporal_position_k
            key_batch = key_batch.contiguous().view(n, self.max_others, self.dim_inner, t_others,
                                                                    h_others, w_others)

            value_batch = value_batch.contiguous().view(n, self.temp_pos_len, max_person_per_sec, self.dim_inner,
                                                        t_others,
                                                        h_others, w_others)
            value_batch = value_batch + self.temporal_position_v
            value_batch = value_batch.contiguous().view(n, self.max_others, self.dim_inner, t_others,
                                                                        h_others, w_others)

        query_batch = query_batch.contiguous().view(n, self.dim_inner, -1).transpose(1, 2)  # [n, thw, dim_inner]
        key_batch = key_batch.contiguous().view(n, self.max_others, self.dim_inner, -1).transpose(2, 3)
        key_batch = key_batch.contiguous().view(n, self.max_others * t_others * h_others * w_others, -1).transpose(
            1, 2)

        qk = torch.bmm(query_batch, key_batch)  # n, thw, max_other * thw

        qk_sc = qk * self.scale_value

        weight = self.softmax(qk_sc)


        value_batch = value_batch.contiguous().view(n, self.max_others, self.dim_inner, -1).transpose(2, 3)
        value_batch = value_batch.contiguous().view(n, self.max_others * t_others * h_others * w_others, -1)
        out = torch.bmm(weight, value_batch)  # n, thw, dim_inner

        out = out.contiguous().view(n, t * h * w, -1)
        out = out.transpose(1, 2)
        out = out.contiguous().view(n, self.dim_inner, t, h, w)

        if self.use_ln:
            if not hasattr(self, "layer_norm"):
                self.layer_norm = nn.LayerNorm([self.dim_inner, t, h, w], elementwise_affine=False).to(
                    device)
            out = self.layer_norm(out)

        out = self.relu(out)

        out = self.out(out)
        out = self.dropout(out)

        if self.shortcut:
            person = self.shortcut(person)

        out = out + person
        return out


class HITStructure(nn.Module):
    def __init__(self, dim_person, dim_mem, dim_out, structure_cfg):
        super(HITStructure, self).__init__()
        self.dim_person = dim_person
        self.dim_others = dim_mem
        self.dim_inner = structure_cfg.DIM_INNER
        self.dim_out = dim_out

        self.max_person = structure_cfg.MAX_PERSON
        self.max_object = structure_cfg.MAX_OBJECT
        self.max_hand = structure_cfg.MAX_HAND
        self.mem_len = structure_cfg.LENGTH[0] + structure_cfg.LENGTH[1] + 1 # 61
        self.mem_feature_len = self.mem_len * structure_cfg.MAX_PER_SEC # 61 * 5 = 305

        self.I_block_list = structure_cfg.I_BLOCK_LIST

        bias = not structure_cfg.NO_BIAS
        conv_init_std = structure_cfg.CONV_INIT_STD

        self.has_P = has_person(structure_cfg)
        self.has_O = has_object(structure_cfg)
        self.has_M = has_memory(structure_cfg)
        self.has_H = has_hand(structure_cfg)

        self.person_dim_reduce = nn.Conv3d(dim_person, self.dim_inner, 1, bias)  # reduce person query
        init_layer(self.person_dim_reduce, conv_init_std, bias)
        self.reduce_dropout = nn.Dropout(structure_cfg.DROPOUT)

        # Init Temporal 
        self.mem_dim_reduce = nn.Conv3d(dim_mem, self.dim_inner, 1, bias)
        init_layer(self.mem_dim_reduce, conv_init_std, bias)
        
        # Init Person
        self.person_key_dim_reduce = nn.Conv3d(dim_person, self.dim_inner, 1, bias)  # reduce person key
        init_layer(self.person_key_dim_reduce, conv_init_std, bias)
        
        # Init Object
        self.object_dim_reduce = nn.Conv3d(dim_person, self.dim_inner, 1, bias)
        init_layer(self.object_dim_reduce, conv_init_std, bias)
        
        # Init Hand
        self.hand_dim_reduce = nn.Conv3d(dim_person, self.dim_inner, 1, bias)
        init_layer(self.hand_dim_reduce, conv_init_std, bias)
            
                        
    def forward(self, person, person_boxes, obj_feature, object_boxes, hand_feature, hand_boxes, mem_feature, person_pooled, ia_feature, phase):
        # RGB stream
        if phase == "rgb":
            query, person_key, object_key, hand_key, mem_key = self._reduce_dim(person, person_boxes, obj_feature, object_boxes, hand_feature, hand_boxes,
                                                                    mem_feature, phase)

            return self._aggregate(person_boxes, query, person_key, object_key, hand_key, mem_key)
        # Pose stream
        else:
            person_key = person_pooled
            
            query, _, object_key, hand_key, mem_key = self._reduce_dim(person, person_boxes, obj_feature, person_boxes, hand_feature, person_boxes,
                                                                    mem_feature, phase)
            pose_feats, _, _, _ = self._aggregate(person_boxes, query, person_key, object_key, hand_key, mem_key)
            
            # Attentive Fusion and Temporal Interaction
            return self._fuse(ia_feature, pose_feats, mem_key)
            
    

    def _reduce_dim(self, person, person_boxes, obj_feature, object_boxes, hand_feature, hand_boxes, mem_feature, phase):
        query = self.person_dim_reduce(person)
        query = self.reduce_dropout(query)
        n = query.size(0)

        if self.has_P:
            person_key = self.person_key_dim_reduce(person)
            person_key = self.reduce_dropout(person_key)
        else:
            person_key = None

        if self.has_O and obj_feature != None:
            object_key = separate_roi_per_person(person_boxes, obj_feature, object_boxes,
                                                 self.max_object)
            object_key = fuse_batch_num(object_key)
            object_key = self.object_dim_reduce(object_key)
            object_key = unfuse_batch_num(object_key, n, self.max_object)
            object_key = self.reduce_dropout(object_key)
        else:
            object_key = None


        if self.has_H and hand_feature != None:
            
            hand_key = separate_roi_per_person(person_boxes, hand_feature, hand_boxes,
                                                    self.max_hand)
            hand_key = fuse_batch_num(hand_key)
            hand_key = self.hand_dim_reduce(hand_key)
            hand_key = unfuse_batch_num(hand_key, n, self.max_hand)
            hand_key = self.reduce_dropout(hand_key)
        else:
            hand_key = None

        if self.has_M and mem_feature != None:
            mem_key = separate_batch_per_person(person_boxes, mem_feature)
            mem_key = fuse_batch_num(mem_key)
            mem_key = self.mem_dim_reduce(mem_key)
            mem_key = unfuse_batch_num(mem_key, n, self.mem_feature_len)
            mem_key = self.reduce_dropout(mem_key)
        else:
            mem_key = None

        return query, person_key, object_key, hand_key, mem_key

    def _aggregate(self, proposals, query, person_key, object_key, hand_key, mem_key):
        raise NotImplementedError

    def _make_interaction_block(self, block_type, block_name, dim_person, dim_other, dim_out, dim_inner, structure_cfg):
        dropout = structure_cfg.DROPOUT
        temp_pos_len = -1
        if block_type == "P":
            max_others = self.max_person
        elif block_type == "O":
            max_others = self.max_object
        elif block_type == "H":
            max_others = self.max_hand
        elif block_type == "M":
            max_others = self.mem_feature_len
            if structure_cfg.TEMPORAL_POSITION:
                temp_pos_len = self.mem_len
        else:
            raise KeyError("Unrecognized interaction block type '{}'!".format(block_type))

        I_block = InteractionUnit(dim_person, dim_other, dim_out, dim_inner,
                                   structure_cfg, max_others,
                                   temp_pos_len=temp_pos_len, dropout=dropout)

        self.add_module(block_name, I_block)


@registry.INTERACTION_AGGREGATION_STRUCTURES.register("serial")
class SerialHITStructure(HITStructure):
    def __init__(self, dim_person, dim_mem, dim_out, structure_cfg):
        super(SerialHITStructure, self).__init__(dim_person, dim_mem, dim_out, structure_cfg)

        block_count = dict()
        for idx, block_type in enumerate(self.I_block_list):
            block_count[block_type] = block_count.get(block_type, 0) + 1
            name = block_type + "_block_{}".format(block_count[block_type])
            dim_out_trans = dim_out if (idx == len(self.I_block_list) - 1) else self.dim_inner
            self._make_interaction_block(block_type, name, self.dim_inner, self.dim_inner,
                                         dim_out_trans, self.dim_inner, structure_cfg)

    def _aggregate(self, person_boxes, query, person_key, object_key, hand_key, mem_key):
        block_count = dict()
        for idx, block_type in enumerate(self.I_block_list):
            block_count[block_type] = block_count.get(block_type, 0) + 1
            name = block_type + "_block_{}".format(block_count[block_type])
            I_block = getattr(self, name)
            if block_type == "P":
                person_key = separate_roi_per_person(person_boxes, person_key, person_boxes, self.max_person, )
                query = I_block(query, person_key)
            elif block_type == "O":
                query = I_block(query, object_key)
            elif block_type == "H":
                query = I_block(query, hand_key)
            elif block_type == "M":
                query = I_block(query, mem_key)
            person_key = query

        return query

@registry.INTERACTION_AGGREGATION_STRUCTURES.register("hitnet")
class HITNet(SerialHITStructure):
    # This is the main class of our framework. Both RGB and Pose
    def __init__(self, dim_person, dim_mem, dim_out, structure_cfg):
        super(HITNet, self).__init__(dim_person, dim_mem, dim_out, structure_cfg)
        self.dense_params = nn.ParameterDict()
        for idx in range(2, len(self.I_block_list)):
            init_tensor = torch.full((idx, self.dim_inner, 1, 1, 1), 
                                     -5 * math.log(10), dtype=torch.float32)
            init_tensor[idx - 1, ...] = 0
            self.dense_params[str(idx)] = nn.Parameter(init_tensor)
        
        self.dense_params_lateral = nn.ParameterDict()
        init_tensor_lateral = torch.full((len(self.I_block_list)//2, self.dim_inner, 1, 1, 1), 
                                         -2 * math.log(10), dtype=torch.float32)
        self.dense_params_lateral = nn.Parameter(init_tensor_lateral)
        
        self.softmax = nn.Softmax(dim=0)
        self.conv_reduce = nn.Conv3d(dim_mem, self.dim_inner, 1, bias=False)
        

        self.non_local = NL(hidden_dim = 2*self.dim_inner, out_dim = self.dim_inner)

    # Lateral units. They will serve as inputs to the intra-modality aggregator
    def aggregate_lateral(self, I_block_object, I_block_hand, I_block_memory, query, object_key, hand_key, mem_key):
        object_feat= I_block_object(query, object_key)
        hand_feat = I_block_hand(query, hand_key)
        mem_feat = I_block_memory(query, mem_key)
        return object_feat, mem_feat, hand_feat

    def _aggregate(self, person_boxes, query, person_key, object_key, hand_key, mem_key):
        block_count = dict()
        history_query = []

        for idx, block_type in enumerate(self.I_block_list):
            block_count[block_type] = block_count.get(block_type, 0) + 1
            name = block_type + "_block_{}".format(block_count[block_type])
            I_block = getattr(self, name)

            # Initialize the lateral units
            name_object = 'O' + "_block_{}".format(block_count[block_type])
            I_block_object = getattr(self, name_object)
            name_hand = 'H' + "_block_{}".format(block_count[block_type])
            I_block_hand = getattr(self, name_hand)
            name_memory = 'M' + "_block_{}".format(block_count[block_type])
            I_block_memory = getattr(self, name_memory)
            

            if idx >= 2:
                query = torch.stack(history_query, dim=1)
                query = torch.sum(query * self.softmax(self.dense_params[str(idx)]), dim=1)
                person_key = query
            if block_type == "P":
                person_key = separate_roi_per_person(person_boxes, person_key, person_boxes, self.max_person, )
                query = I_block(query, person_key)
                person_message = query
            elif block_type == "O":
                P_O, P_M, P_K = self.aggregate_lateral(I_block_object, I_block_hand, I_block_memory, 
                                                       query, object_key, hand_key, mem_key)
                if P_M.shape[1] == self.dim_out:
                    P_M = self.conv_reduce(P_M)
                
                query = torch.stack([query, P_O, P_M, P_K], dim=1)
                query = torch.sum(query * self.softmax(self.dense_params_lateral), dim=1)
                query = I_block(query, object_key)
                object_message = query
                
            elif block_type == "H": 
                P_O, P_M, P_K = self.aggregate_lateral(I_block_object, I_block_hand, I_block_memory, 
                                                       query, object_key, hand_key, mem_key)
                if P_M.shape[1] == self.dim_out:
                    P_M = self.conv_reduce(P_M)
                
                query = torch.stack([query, P_O, P_M, P_K], dim=1)
                query = torch.sum(query * self.softmax(self.dense_params_lateral), dim=1)


                query = I_block(query, hand_key)
                hand_message = query

            elif block_type == "M": 
                
                P_O, P_M, P_K = self.aggregate_lateral(I_block_object, I_block_hand, I_block_memory, 
                                                       query, object_key, hand_key, mem_key)
                if P_M.shape[1] == self.dim_out:
                    P_M = self.conv_reduce(P_M)
                
                query = torch.stack([query, P_O, P_M, P_K], dim=1)
                query = torch.sum(query * self.softmax(self.dense_params_lateral), dim=1)
                if idx != 7:
                    query = I_block(query, mem_key)
            

            person_key = query
            history_query.append(query)
                
        return query, person_message, object_message, hand_message
    
    # Attentive Fusion and Temporal Interaction
    def _fuse(self, rgb, pose, mem_key):
        name_memory = 'M' + "_block_2"
        I_block_memory = getattr(self, name_memory)
        # Attentive Fusion
        query = torch.cat([rgb, pose], dim=1)
        query = self.non_local(query)
        # Temporal Interaction
        query = I_block_memory(query, mem_key)
        
        return query



def separate_roi_per_person(proposals, things, other_proposals, max_things):
    """
    :param things: [n2, c, t, h, w]
    :param proposals:
    :param max_things:
    :return [n, max_other, c, t, h, w]
    """
    res = []
    _, c, t, h, w = things.size()
    device = things.device
    index = 0
    for i, (person_box, other_box) in enumerate(zip(proposals, other_proposals)):
        person_num = len(person_box)
        other_num = len(other_box)
        tmp = torch.zeros((person_num, max_things, c, t, h, w), device=device)
        if other_num > max_things:
            idx = torch.randperm(other_num)[:max_things]
            tmp[:, :max_things] = things[index:index + other_num][idx]
        else:
            tmp[:, :other_num] = things[index:index + other_num]

        res.append(tmp)
        index += other_num
    features = torch.cat(res, dim=0)
    return features


def separate_batch_per_person(proposals, things):
    """
    :param things: [b, max_others, c, t, h, w]
    :return [n, max_others, c, t, h, w]
    """
    res = []
    b, max_others, c, t, h, w = things.size()
    device = things.device
    for i, person_box in enumerate(proposals):
        person_num = len(person_box)
        tmp = torch.zeros((person_num, max_others, c, t, h, w), device=device)
        tmp[:] = things[i]
        res.append(tmp)
    return torch.cat(res, dim=0)


def fuse_batch_num(things):
    n, number, c, t, h, w = things.size()
    return things.contiguous().view(-1, c, t, h, w)


def unfuse_batch_num(things, batch_size, num):
    assert things.size(0) == batch_size * num, "dimension should matches"
    _, c, t, h, w = things.size()
    return things.contiguous().view(batch_size, num, c, t, h, w)


def init_layer(layer, init_std, bias):
    if init_std == 0:
        nn.init.constant_(layer.weight, 0)
    else:
        nn.init.normal_(layer.weight, std=init_std)
    if bias:
        nn.init.constant_(layer.bias, 0)


def make_hit_structure(cfg, dim_in):
    func = registry.INTERACTION_AGGREGATION_STRUCTURES[
        cfg.MODEL.HIT_STRUCTURE.STRUCTURE
    ]
    return func(dim_in, cfg.MODEL.HIT_STRUCTURE.DIM_IN, cfg.MODEL.HIT_STRUCTURE.DIM_OUT, cfg.MODEL.HIT_STRUCTURE)