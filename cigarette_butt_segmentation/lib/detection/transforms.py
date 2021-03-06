"""Analogue of pytorch.transform.*. But they work on simple batch image and
transforms here work with images and complicated target structures together.
Each target is a dict with fields
    "boxes" : bounding boxes, 2D array, N objects * 4 coordinates (left x, top y, right x, bottom y)
    "labels"
    "masks" : set of bitmasks with the same as image size, 3D array
    "image_id"
    "area"
    "iscrowd".
Each field is an array or N+1-dimensional tensor, each element (or first dimension's index value)
corresponds to one object instance.

get_transform method here is intended for usage from Jupyter Notebooks.
Placed here in order not to overload them.
"""

import numpy as np
import random

from torchvision.transforms import functional as F

from ..dataset import CigDataset   
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


def debugPrint(str, printToConsole=False):
    with open('results/debug.log', 'a') as file:
        file.write(str + '\n')
    if printToConsole:
        print(str)

class RandomMaskedObjectCopy(object):
    """When we have one object with its mask on an image,
    this transformer copies this object maximum max_copy_count times on the same image.
    Bounding boxes of all copies are mutually non-intersecting. If the source object is big and it is hard
    to find proper place, the class gives up and creates so many copies it was able to the moment.
    """

    max_retry_count = 20
    """When new object's copies are created, the place is chosen randomly and then checked for intersections
    with already existing object and copies. If they intersect, the procedure repeats, but no more than
    this number of times.
    """

    def __init__(self, max_copy_count=1):
        self.max_copy_count = max_copy_count
        pass

    def __call__(self, image, target):
        height, width = image.size[-2:]
        if target["boxes"].shape[0] == 0:
            return image, target
        bboxes = np.array(target["boxes"] + 0.01, dtype=int)  
        base_bbox = bboxes[0]
            # This will be source bounding box during the entire method
        # print(base_bbox)
        base_bbox_width = int(base_bbox[2] - base_bbox[0])
        base_bbox_height = int(base_bbox[3] - base_bbox[1])
        num_objs = bboxes.shape[0]
        
        obj_mask = np.array(target['masks'][0][base_bbox[1] : base_bbox[3] + 1, base_bbox[0] : base_bbox[2] + 1])
            # Bboxes in our dataset has format (left x, top y, right x, bottom y)
            # and despite typical ranges convention, right and bottom pixels are included. So +1
        pixels_coords = np.where(obj_mask > 0)

        # Gathering potential array of masks with 1s, 2s and so on back into one matrix
        cur_masks = np.array(target['masks'])
        assert num_objs <= 1 or (num_objs > 1 and \
                np.sum(np.multiply(cur_masks[0], cur_masks[1])) == 0)
            # Simplified condition that objects don't intersect. Actually usually 
            # there will be 1 object
        full_mask = np.sum(cur_masks, axis=0)

        image = np.array(image)
        for new_obj_ind in range(self.max_copy_count):
            retry_ind = 1
            while retry_ind <= self.max_retry_count:
                new_coords = (random.randrange(width - base_bbox_width),
                              random.randrange(height - base_bbox_height))
                new_bbox = np.array([new_coords[0], new_coords[1], 
                                     new_coords[0] + base_bbox_width, new_coords[1] + base_bbox_height])
                intersect_found = False   
                for obj_ind in range(num_objs):
                    bbox2 = bboxes[obj_ind]
                    # The logic below is similar to the one in pycocotools
                    # Take maximums of minimum coordinates (left x and top y) and minimums of maximum ones
                    lt = np.maximum(bbox2[:2], new_bbox[:2])
                    rb = np.minimum(bbox2[2:], new_bbox[2:]) 
                    wh = (rb - lt)
                    if wh[0] > 0 and wh[1] > 0:
                        intersect_found = True
                        # debugPrint('New obj %d, retry %d: %s - %s (%d)' % \
                        #       (new_obj_ind + 1, retry_ind, str(new_bbox), str(bbox2), obj_ind))
                        break
                if not intersect_found:
                    break
                else:
                    retry_ind += 1
            if intersect_found:
                break

            # Finally free place for new object found
            
            image[new_bbox[1] : new_bbox[3] + 1,
                    new_bbox[0] : new_bbox[2] + 1, :][obj_mask > 0, :] = \
                image[base_bbox[1] : base_bbox[3] + 1,
                        base_bbox[0] : base_bbox[2] + 1, :][obj_mask > 0, :]
            num_objs += 1
            bboxes = np.concatenate([bboxes, np.expand_dims(new_bbox, 0)], axis=0)
            full_mask[new_bbox[1] : new_bbox[3] + 1, 
                      new_bbox[0] : new_bbox[2] + 1][obj_mask > 0] = num_objs
        target = CigDataset._create_target(full_mask, int(target["image_id"]))
        return image, target
        

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
