"""Your module with methods, useful for data retrieval and visualization for example in Jupyter Notebooks.
I have also added for example methods for storing/loading state of a model to a file.
"""

import cv2
import numpy as np
import torch

def get_mask(img_id, annotations):
    """Returns mask.

    Parameters
    ----------
    img_id : int
        Image id.
    annotations : dict
        Ground truth.

    Returns
    -------
    np.ndarray, 2d
        Mask that contains only 2 unique values: 0 - denotes background, 255 - denotes object.
    
    """
    img_info = annotations["images"][img_id]
    assert img_info["id"] == img_id
    w, h = img_info["width"], img_info["height"]
    mask = np.zeros((h, w)).astype(np.uint8)
    gt = annotations["annotations"][img_id]
    assert gt["id"] == img_id
    polygon = np.array(gt["segmentation"][0]).reshape((-1, 2))
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=255)
    
    return mask


def encode_rle(mask):
    """Returns encoded mask (run length) as a string.

    Parameters
    ----------
    mask : np.ndarray, 2d
        Mask that consists of 2 unique values: 0 - denotes background, 1 - denotes object.

    Returns
    -------
    str
        Encoded mask.

    Notes
    -----
    Mask should contains only 2 unique values, one of them must be 0, another value, that denotes
    object, could be different from 1 (for example 255).

    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def decode_rle(rle_mask, shape=(512, 512)):
    """Decodes mask from rle string.

    Parameters
    ----------
    rle_mask : str
        Run length as string formatted.
    shape : tuple of 2 int, optional (default=(320, 240))
        Shape of the decoded image.

    Returns
    -------
    np.ndarray, 2d
        Mask that contains only 2 unique values: 0 - denotes background, 255 - denotes object.
    
    """
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 255

    return img.reshape(shape)


def expand_bbox(bbox, pixels, max_size):
    """Expands bounding box (left, top, right, bottom) by pixels pixels in each direction
       May work for some other formats too (top, left, bottom, right) for example
    """

    def correct(v):
        if v < 0:
            return 0
        elif v >= max_size:
            return max_size
        else:
            return v
    return (correct(bbox[0] - pixels), correct(bbox[1] - pixels), 
            correct(bbox[2] + pixels), correct(bbox[3] + pixels))


def printProgress(str, printToConsole=True):
    with open('results/progress.log', 'a') as file:
        file.write(str + '\n')
    if printToConsole:
        print(str)


def save_model_state(model, fileName):
    state = {'model': model.state_dict()}
    torch.save(state, fileName)

def load_model_state(model, fileName):
    state = torch.load(fileName)
    # print('state ', state)
    savedStateDict = state['model']
    try:
        c_replacements = [['module.', ''], ]
        stateDict = {}
        for name, data in savedStateDict.items():
            for replRule in c_replacements:
                name = name.replace(replRule[0], replRule[1])
            stateDict[name] = data
        result = model.load_state_dict(stateDict, strict=1)
        print('State loaded from %s (%s)' % (fileName, result))
        del state['model']
        return state
    except Exception as ex:
        print("Error in loadState: %s" % str(ex))

    if len(savedStateDict) != len(model.state_dict()):
        raise Exception('You are trying to load parameters for %d layers, but the model has %d' %
                        (len(savedStateDict), len(self.state_dict())))

    del state['model']
    return state

