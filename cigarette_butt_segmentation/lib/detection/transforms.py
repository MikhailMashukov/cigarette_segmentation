import numpy as np
import random
import torch

from torchvision.transforms import functional as F

from ..net import CigDataset   
    # Ugly import (because it is from generally independent module of the same level), 
    # but it economizes a lot of work and copy-paste
    # for keeping the same structures and torch types in target

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class RandomMaskedObjectCopy(object):
    def __init__(self):
        pass

    def __call__(self, image, target):
        height, width = image.size[-2:]
        if target["boxes"].shape[0] == 0:
            return image, target
        bbox = np.array(target["boxes"][0], dtype=int)
        # print(bbox)
        cur_bbox_width = bbox[2] - bbox[0]
        cur_bbox_height = bbox[3] - bbox[1]
        if cur_bbox_width > width * 0.4 and cur_bbox_height > height * 0.5:
            return image, target

        while (True):
            new_coords = (random.randrange(width - cur_bbox_width),
                          random.randrange(height - cur_bbox_height))
            new_bbox = np.array([new_coords[0], new_coords[1], 
                                 new_coords[0] + cur_bbox_width, new_coords[1] + cur_bbox_height])
            lt = np.maximum(bbox[:2], new_bbox[:2])
            rb = np.minimum(bbox[2:], new_bbox[2:]) 
            wh = (rb - lt)
            # inter = wh[0] * wh[1]
            # print('%s - %s intersection: %s' % (str(bbox), str(new_bbox), str(wh)))
            if wh[0] > 0 and wh[1] > 0:
                continue

            # import matplotlib.pyplot as plt

            image = np.array(image)
            # Bboxes in our dataset has format (left x, top y, right x, bottom y)
            # and despite typical ranges convention, right and bottom pixels are included. So +1
            obj_mask = np.array(target['masks'][0][bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1])
            # plt.imshow(obj_mask)
            pixels_coords = np.where(obj_mask > 0)
            pixels_coords = (pixels_coords[0] + new_bbox[1] - bbox[1], 
                             pixels_coords[1] + new_bbox[0] - bbox[0])
            # image[pixels_coords] = 2
            
            image[new_bbox[1] : new_bbox[3] + 1, 
                  new_bbox[0] : new_bbox[2] + 1, :][obj_mask > 0, :] = \
                image[bbox[1] : bbox[3] + 1, 
                      bbox[0] : bbox[2] + 1, :][obj_mask > 0, :]
            # plt.imshow(image)
            if 0:
                target["boxes"] = np.stack([target["boxes"], np.expand_dims(new_bbox, 0)], axis=0)
                new_full_mask = np.zeros(image.shape[:2], dtype=obj_mask.dtype)
                new_full_mask[new_bbox[1] : new_bbox[3] + 1, 
                            new_bbox[0] : new_bbox[2] + 1][obj_mask > 0] = 2
                # print(new_full_mask.shape, target["masks"].shape, 
                #     np.expand_dims(new_full_mask[new_bbox[1] : new_bbox[3] + 1, 
                #               new_bbox[0] : new_bbox[2] + 1], 0).shape)
                target["masks"] = np.stack([target["masks"], np.expand_dims(new_full_mask, 0)], axis=0)
            else:
                num_objs = target["boxes"].shape[0]
                cur_masks = np.array(target['masks'])
                # print('cur', np.multiply(cur_masks[0], cur_masks[0]).shape)
                mask = cur_masks[0]
                assert num_objs <= 1 or (num_objs > 1 and \
                        np.sum(np.multiply(cur_masks[0], cur_masks[1])) == 0)
                    # Simplified condition that objects don't intersect. Actually usually 
                    # there will be 1 object
                new_mask = np.sum(cur_masks, axis=0)
                # print('new_mask', new_mask.min(), new_mask.max(), new_mask)

                num_objs += 1
                new_mask[new_bbox[1] : new_bbox[3] + 1, 
                         new_bbox[0] : new_bbox[2] + 1][obj_mask > 0] = num_objs
                target = CigDataset._create_target(new_mask, int(target["image_id"]))
               
            # plt.show()
            return image, target
            
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def get_transform(train):
    transforms = []
    # converts a PIL image into PyTorch Tensor
    if train:
        transforms.append(RandomMaskedObjectCopy(10))
    transforms.append(ToTensor())
    # if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        # transforms.append(RandomHorizontalFlip(0.5))        
    return Compose(transforms)