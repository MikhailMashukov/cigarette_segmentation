"""For your Jupyter Notebook you will need train_one_epoch and evaluate methods from this module
   and no more.
"""

import math
import numpy as np
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from .det_utils import *
from ..metrics import EPS

def train_one_epoch(model, optimizer, data_loader, device, epoch, 
                    print_freq, printFunc=print):
    """Perform training of your model on the entire dataset. Meaning of all parameters is obvious.

    Parameters
    ----------
    ...
    printFunc : method of the form def print(str)
        Useful for saving train_one_epoch's output to a file for a case you disconnect your Jupyter Notebook
        from the server and wouldn't see some results.

    Notes
    ----------
    The method prints intermediate results since they can be useful while you are awaiting for finish
    and to show that the process is going forward.
    """

    model.train()
    metric_logger = MetricLogger(delimiter="  ", printFunc=printFunc)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # print('batch', len(images))
        images = images_to_device(images, device)
        targets = targets_to_device(targets, device)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            printFunc("Loss is {}, stopping training".format(loss_value))
            printFunc(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

s = None
p = None
d = None
def engine_get_dice(target, pred, threshold=0.5):
    """Analogue of lib.metrics.get_dice.

    Notes
    -----
    lib.metrics.get_dice. takes prepared arrays, but here we need to extract multiple masks
    from tricky data structures.
    Since target masks in cigarettes case shouldn't intersect, the method was speeded up
    by uniting target masks first and then separated checking of each predicted mask for union
    with this united target.
    """
    global s
    global p
    s = target
    p = pred
    # assert isinstance(src_targets, tuple) and len(src_targets) == 1
    # assert isinstance(outputs, list) and len(outputs) == 1
    # # print(outputs)
    # target = src_targets[0]
    # pred = outputs[0]

    # print(src_targets)
    target_mask_count = target['masks'].shape[0]
    if target_mask_count == 0:
        return 0
    target_mask_sum = target['masks'][0].numpy()
    # sumStr = ''
    for i in range(1, target_mask_count):
        target_mask_sum += target['masks'][i].numpy()
        # sumStr += ', ' + str((target['masks'][i].numpy() > 0).sum())
    target_mask = (target_mask_sum > 0)

    # print(target_mask_sum.shape, np.where(target_mask))
    # print('engine_get_dice', 'target_mask.sum()', target_mask.sum(), pred['masks'].shape)
    pred_mask_count = pred['masks'].shape[0]
    if pred_mask_count == 0:
        return 0
        
    intersection_sum = 0
    im_sum = target_mask.sum()
    # sumStr += '; '
    preds = pred['masks'].numpy() > threshold
    for mask_ind in range(pred_mask_count):
        pred_mask = preds[mask_ind, 0]
        # print('pred_mask ', pred_mask.shape)
        # sumStr += ', ' + str((target_mask & pred_mask).sum())
        intersection_sum += (target_mask & pred_mask).sum()
        im_sum += pred_mask.sum()
    # print(sumStr)
    return 2.0 * intersection_sum / (im_sum + EPS)
        
@torch.no_grad()
def evaluate(model, data_loader, device, printFunc=print):
    """Evaluates and prints extensive statistics on the entire dataset, given in the form of data_loader.

    Parameters
    ----------
    ...
    printFunc : method of the form def print(str)
        See comments for it at train_one_epoch's comments.

    Notes
    -----
    Warning: can be slow for big train dataset. I didn't catch (had no time for this) why
    get_coco_api_from_dataset is so slow. Just in case you need to speed this up to debug something,
    torch.utils.data.Subset(dataset, ...) didn't help for me. The only thing that helped me -
    return less in your source dataset's object __len__ method
    """

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ", printFunc=printFunc)
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    dices = []
    threshold = 0.5
    for images, src_targets in metric_logger.log_every(data_loader, 100, header):
        # Processing batch of images (can be for example 8 for train and 1 for test)
        # print('eval', image.size)
        images = images_to_device(images, device)
        targets = targets_to_device(src_targets, device)
        # images = list(img.to(device) for img in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in src_targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        assert isinstance(src_targets, tuple) 
        for i in range(len(src_targets)):
            dices.append(engine_get_dice(src_targets[i], outputs[i], threshold))

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    global d
    d = dices
    # printFunc("Dices: %s" % (str(dices)))
    print(type(np.mean(dices)))

    printFunc("Averaged stats: dice %.5f, %s" % (np.mean(dices), str(metric_logger)))
    printFunc("Dices: %s" % str(dices))
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
