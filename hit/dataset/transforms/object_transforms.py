import numpy as np
import torch

class Compose(object):
    # Class used to compose different kinds of object transforms
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, object_boxes, _, transform_randoms):
        #should reuse the random varaible in video transforms
        for t in self.transforms:
            object_boxes = t(object_boxes, _, transform_randoms)
        return object_boxes


class PickTop(object):
    # pick top scored object boxes.
    def __init__(self, top_k):
        self.top_k = top_k

    def __call__(self, objects, boxes, _):
        objects = objects.top_k(self.top_k, boxes)
        return objects


class Resize(object):
    def resize_keypoint(self, keypoint, in_size, out_size):
        """Change values of keypoint according to paramters for resizing an image.
        Args:
            keypoint (~numpy.ndarray): Keypoints in the image.
                The shape of this array is :math:`(K, 2)`. :math:`K` is the number
                of keypoint in the image.
                The last dimension is composed of :math:`y` and :math:`x`
                coordinates of the keypoints.
            in_size (tuple): A tuple of length 2. The height and the width
                of the image before resized.
            out_size (tuple): A tuple of length 2. The height and the width
                of the image after resized.
        Returns:
            ~numpy.ndarray:
            Keypoint rescaled according to the given image shapes.
        """
        keypoint_list = []
        for i in range(keypoint.shape[0]):
            keypoint[i] = keypoint[i].copy()
            y_scale = float(out_size[0]) / in_size[0]
            x_scale = float(out_size[1]) / in_size[1]
            keypoint[i][:, 0] = y_scale * keypoint[i][:, 0]
            keypoint[i][:, 1] = x_scale * keypoint[i][:, 1]
            keypoint_list.append(keypoint[i])
        keypoint = torch.tensor(np.stack(keypoint_list))
        return keypoint

    def __call__(self, object_boxes, _, transform_randoms):
        # resize according to video transforms
        in_size = object_boxes.size
        out_size = transform_randoms['Resize']
        size = transform_randoms["Resize"]
        if "keypoints" in object_boxes.extra_fields:
            keypoints = object_boxes.extra_fields['keypoints'][:, :, :2] 
            object_boxes.extra_fields['keypoints'] = self.resize_keypoint(keypoints, in_size, out_size)
        if object_boxes is not None:
            object_boxes = object_boxes.resize(size)

        

        return object_boxes



class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, object_boxes, _, transform_randoms):
        # flip according to video transforms
        flip_random = transform_randoms["Flip"]
        if flip_random < self.prob:
            object_boxes.transpose(0)
        return object_boxes
