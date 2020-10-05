from PIL import Image, ImageOps
import json
import numpy as np
import os

import torch
from . import utils


class CigDataset(torch.utils.data.Dataset):
    """Our cigarettes train or test images in the form for feeding to model"""

    def __init__(self, root, transform=None):
        """

        :param root: train or test directory. Images has to be in images subfolders
        :param transform: optional processing class; they has to have __call__ method of the form
            def __call__(self, image, target):
            and return changed (image, target)
        """

        self.root = root
        self.transform = transform
        # load all image files's paths, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.img_ids = [int(path.split("/")[-1].split(".")[0]) for path in self.imgs]
        self.annotations = json.load(open(os.path.join(root, "coco_annotations.json"), "r"))
        print('%d images found' % (len(self.imgs)))

    def __getitem__(self, idx):
        # print('get', idx)
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        mask = utils.get_mask(self.img_ids[idx], self.annotations)

        target = self._create_target(mask, idx)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        # return 110      # The only way I have found to speed up evaluate on the dataset
        return len(self.imgs)

    @staticmethod
    def _create_target(mask, img_idx):
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([img_idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return target