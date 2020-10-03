import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

# For transforms' extension
from collections.abc import Sequence
from PIL import Image, ImageOps
import json
import numpy as np
import os

import torchvision
from .torchvision import transforms as T
from torchvision.transforms import functional as F
from lib import utils

# import DeepOptions

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     groups=groups, bias=False)

# @torch.jit.unused
class PadTo(object):
    def __init__(self, targetSize, fill=0, padding_mode="constant"):
        super().__init__()
        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if not isinstance(targetSize, Sequence) or len(targetSize) not in [2]:
            raise ValueError("Target size must 2 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.targetSize = targetSize
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        if isinstance(img, Image.Image):
            # img = np.array(img)
            shape = img.size
            # print('image shape', shape)
            padding = [self.targetSize[0] - shape[0], self.targetSize[1] - shape[1]]
        else:
            shape = img.shape
            # print('shape', shape) #, 'mode', img.mode)
            padding = [self.targetSize[1] - shape[0], self.targetSize[0] - shape[1]]
        # print('target shape', target.size, 'mode', target.mode)
        # print('param2', param2.items())

        if padding[0] <= 0 and padding[1] <= 0:
            return img
        padding = tuple((((v + 1) // 2) if v >= 0 else 0) for v in padding)

        if isinstance(img, Image.Image):
            # print('expanding image', img, padding)
            # print('expanded', ImageOps.expand(img, border=padding, fill=self.fill).size)
            return ImageOps.expand(img, border=padding, fill=self.fill)
        else:
            padding = tuple((v, v) for v in padding)
            # newTarget = param2.copy()
            # newTarget['masks'] =
            if len(shape) > 2:
                padding = (padding[0], padding[1], (0, 0))
            # print('expanding', padding)
            return np.pad(img, padding, constant_values=self.fill)
                # ImageOps.expand(img, border=padding, fill=self.fill), \

    def __repr__(self):
        return self.__class__.__name__ + '(targetSize={0}, fill={1}, padding_mode={2})'.\
            format(self.targetSize, self.fill, self.padding_mode)

class Crop(object):
    def __init__(self, top, left, height, width):
        self.rect = (top, left, height, width)

    def __call__(self, img):
        return F.crop(img, self.rect[1], self.rect[0], self.rect[3], self.rect[2])
            # left, top, width, height

class ToTensor_Chip(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        # print('target to t.')
        target = F.to_tensor(target)
        return image, target


class CigDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.img_ids = [int(path.split("/")[-1].split(".")[0]) for path in self.imgs]
        self.annotations = json.load(open(os.path.join(root, "coco_annotations.json"), "r"))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        print('%d images' % (len(self.imgs)))

    def __getitem__(self, idx):
        # print('get', idx)
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = utils.get_mask(self.img_ids[idx], self.annotations) 

        # img = np.array(img)

        # print(type(img))
        mask = np.array(mask)
        # print('loaded', img.size, mask.shape, mask.dtype)
        # mask = np.array(mask, dtype=np.uint8)
        # print('converted')
        mask[mask > 0] = 1
        # mask = torch.as_tensor(mask, dtype=torch.uint8)
        # target = torchvision.transforms.ToPILImage()(mask)
        target = Image.fromarray(mask)

        import inspect

        if self.transform is not None:
            # print('transform', type(self.transform), inspect.getsource(self.transform))
            img, target = self.transform(img, target)            
            # img = self.transforms(img)
            # target = self.transforms(target)
            # print(target.max())

        return img, target # {'mask': target}

    def __len__(self):
        return len(self.imgs)

if 0:
# ChipDataset
    def __getitem__(self, idx):
        # print('get', idx)
        idx = 0         #d_
        if idx in self.cache:
            cacheItem = self.cache[idx]
            return cacheItem[0].detach().clone(), cacheItem[1].detach().clone()

        img_path = os.path.join(self.root, self.imgs[idx])
        mask_path = os.path.join(self.root, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        # mask[mask > 0] = 255
        target = Image.fromarray(mask)

        path = os.path.join(self.root, '2D_simplified_weights.png')
        weights = Image.open(path)
        weights = np.array(weights) / 255.0
        # weights = weights[300:812, 250:954]      # Crop(250, 300, 704, 512)
        weights = Image.fromarray(weights)

        if self.transfSets is not None:
            # print('transform', img.__class__.__name__)
            img = self.transfSets[0](img)
            target = self.transfSets[1](target)
            # print(target.max())
            weights = self.transfSets[2](weights)

        self.cache[idx] = (img, target, weights)
        return img.detach().clone(), target.detach().clone()

    def __len__(self):
        return self.fakeLen       # In order to allow bigger batch
        # return len(self.imgs)


class UnetConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UnetResidConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        self.midConv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))
            
    def forward(self, x):
        x = self.conv1(x)
        x2 = self.midConv(x)
        x = self.conv2(x)
        x2 = self.midConv(x2)
        x2 = self.midConv(x2)
        return x + x2

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            UnetConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = UnetResidConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=scale_factor, stride=scale_factor)
            self.conv = UnetConv(in_channels, out_channels)


    def forward(self, x1, x2):    # x1 - smaller
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # print('up', x1.shape, x2.shape, diffY, diffX)

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width) #, groups=groups)
        self.bn1 = norm_layer(width)
        print('conv2: w %d, stride %d, groups %d, dilation %d' % \
                (width, stride, groups, dilation))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion) #, groups=groups)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # print('id', x.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # print('out3', out.shape)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ChipNet(nn.Module):
    def __init__(self, num_classes=2, basePlaneCount=32, norm_layer=None):
        super().__init__()
        planeCount = basePlaneCount
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.namedLayers = {}
        bilinear = True

        self.conv1 = conv3x3(3, planeCount, 1)
        # self.conv1 = nn.Conv2d(3, planeCount, kernel_size=5, stride=1,
        #                        padding=2, groups=1, bias=False, dilation=1)
        # self.namedLayers['conv1'] = self.conv1
        self.bn1 = norm_layer(planeCount)
        # self.conv12 = conv1x1(3, planeCount, 1)
        # self.bn12 = norm_layer(planeCount)
        self.relu = nn.ReLU(inplace=True)

        factor = 2 if bilinear else 1
        self.down1 = Down(planeCount, basePlaneCount)
        self.down2 = Down(basePlaneCount, basePlaneCount)
        self.down3 = Down(basePlaneCount, basePlaneCount)
        self.down4 = Down(basePlaneCount, basePlaneCount)
        self.resid = Bottleneck(basePlaneCount, basePlaneCount // 4, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=norm_layer)
        self.up1 = Up(basePlaneCount * 4 // factor, basePlaneCount, bilinear)
        self.up2 = Up(basePlaneCount * 4 // factor, basePlaneCount, bilinear)
        self.up3 = Up(basePlaneCount * 4 // factor, basePlaneCount, bilinear)
        self.up4 = Up(basePlaneCount * 4 // factor, planeCount, bilinear)

        # self.conv2 = conv3x3(planeCount, planeCount)
        # self.namedLayers['conv2'] = self.conv2
        self.bn2 = norm_layer(planeCount)
        self.conv3 = conv3x3(planeCount, planeCount)
        self.bn3 = norm_layer(planeCount)
        self.conv4 = conv1x1(planeCount, num_classes)
        # self.namedLayers['conv4'] = self.conv4
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # print('conv2d found', name)
                    # 'conv1', 'down1.maxpool_conv.1.double_conv.0', ...
                self.namedLayers[name] = m
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # print('forward', len(x), x.shape)
        # x = x[0]
        # x2 = self.conv12(x)
        # x2 = self.bn12(x2)
        # x2 = self.relu(x2)
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        # x = self.maxpool(x)

        if hasattr(self, 'conv2'):
            x = self.conv2(x1)
            # print('22', x.shape)
            x = self.bn2(x)
            x1 = self.relu(x)

        x2 = self.down1(x1)
        # print('2', x2.shape)
        if hasattr(self, 'down2'):
            x3 = self.down2(x2)
            # print('3', x3.shape)
        if hasattr(self, 'down3'):
            x4 = self.down3(x3)
        if hasattr(self, 'down4'):
            x5 = self.down4(x4)
        # x3 = self.resid.forward(x3)
        # x3 = self.resid.forward(x3)
        
        if hasattr(self, 'down4'):
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        elif hasattr(self, 'down3'):
            x = self.up2(x4, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        elif hasattr(self, 'down2'):
            x = self.up3(x3, x2)
            x = self.up4(x, x1)
        else:
            x = self.up4(x2, x1)

        # print('4', x.shape)
        # x = self.conv3(x)   # * x2
        # x = self.bn2(x)
        # x = self.relu(x)

        x = self.conv4(x)
        return x

    def getLayer(self, layerName):
        return self.namedLayers[layerName]

    def getAllLayers(self):
        return self.namedLayers


    def saveState(self, fileName,
                  additInfo={}, additFileName=None):
        if 1:
            state = {'model': self.state_dict()}
            if additFileName is None:
                state.update(additInfo)
                torch.save(state, fileName)
                    # os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epochNum + 1))
            else:
                torch.save(state, fileName)
                torch.save(additInfo, additFileName)
        else:
            torch.save(self.state_dict(), fileName)

    def loadState(self, fileName):
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
            result = self.load_state_dict(stateDict, strict=1)
            print('State loaded from %s (%s)' % (fileName, result))
            del state['model']
            return state
        except Exception as ex:
            print("Error in loadState: %s" % str(ex))

        if len(savedStateDict) != len(self.state_dict()):
            raise Exception('You are trying to load parameters for %d layers, but the model has %d' %
                            (len(savedStateDict), len(self.state_dict())))

        del state['model']
        return state

    def loadStateAdditInfo(self, fileName):
        return torch.load(fileName)

    def getMultWeights(self, layerName, allowCombinedLayers=False):  # , epochNum):
        try:
            layer = self.getLayer(layerName)
            allWeights = layer.state_dict()
            # print('allWeights', allWeights)
        except:
            allWeights = None
#         print('len ', len(allWeights))

        if allWeights:
#             assert len(allWeights) == 1 or 'weight' in allWeights
            weights = allWeights['weight']
        else:
            if allowCombinedLayers:
                allWeights = []
                for curLayerName, layer in model.getAllLayers():
                    if curLayerName.find(layerName + '_') == 0:
                        allLayerWeights = layer.get_weights()
#                         assert len(allLayerWeights) == 1 or len(allLayerWeights[0].shape) > len(allLayerWeights[1].shape)
                        allWeights.append(allLayerWeights['weight'])
                if not allWeights:
                    raise Exception('No weights found for combined layer %s' % layerName)
                weights = np.concatenate(allWeights, axis=3)
            else:
                raise Exception('No weights found for layer %s' % layerName)

        weights = weights.cpu().numpy()      # E.g. [96, 3, 11, 11]
        # Converting to channels_last
        # print('weights', weights.shape)
        if len(weights.shape) == 5:
            weights = weights.transpose((2, 3, 4, 0, 1))    # Not tested
        elif len(weights.shape) == 4:
            weights = weights.transpose((1, 0, 2, 3))
        elif len(weights.shape) == 3:
            weights = weights.transpose((2, 0, 1))          # Not tested
        return weights


def createSimpleChipNet(num_classes, trainable_backbone_layers=3, **kwargs):
    return ChipNet(num_classes)


# def get_transforms(train):
#     transforms = []
#     targetSize = None   # (640, 512)
#     # targetSize = (32, 32)
#     transforms.append(Crop(250, 300, 704, 512))
#     imageWeightsTransforms = list(transforms)
#     if 0: # train:
#         # during training, randomly flip the training images
#         # and ground-truth for data augmentation
#         # transforms.append(T.RandomHorizontalFlip(0.5))
#         transforms.append(torchvision.transforms.RandomHorizontalFlip())
#         transforms.append(torchvision.transforms.RandomVerticalFlip())
#     if not targetSize is None:
#         transforms.append(PadTo(targetSize, fill=0))
#         transforms.append(torchvision.transforms.CenterCrop((targetSize[1], targetSize[0])))
#     transforms.append(torchvision.transforms.ToTensor())
#     # return torchvision.transforms.Compose(transforms)

#     imageTransforms = transforms + [torchvision.transforms.Normalize(
#             [0.27922228] * 3, [0.22716749] * 3, inplace=True)]
#     maskTransforms = transforms # + [torchvision.transforms.Normalize(
#             # np.array([0.06156306]), np.array([0.23854734]), inplace=True)]
#     imageWeightsTransforms.append(torchvision.transforms.ToTensor())
#     transfSets = [torchvision.transforms.Compose(imageTransforms), 
#                   torchvision.transforms.Compose(maskTransforms),
#                   torchvision.transforms.Compose(imageWeightsTransforms)]
#     return transfSets


if __name__ == '__main__':
    import transforms as T
    from train import *
    import utils

    dataset = ChipDataset(r'E:\Projects\Freelance\INIRSibir\Images', get_transform(train=True), fakeLen=2)
    dataset_test = ChipDataset(r'E:\Projects\Freelance\INIRSibir\Images', get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    if len(indices) > 50:
        testImageCount = 50
    else:
        testImageCount = len(indices) // 3

    if testImageCount > 0:
        dataset = torch.utils.data.Subset(dataset, indices[:-testImageCount])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-testImageCount:])
    else:
        dataset = torch.utils.data.Subset(dataset, indices)
        dataset_test = torch.utils.data.Subset(dataset_test, indices)
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, num_workers=4,  # shuffle=True,
        sampler=train_sampler, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, num_workers=4,  # shuffle=False,
        sampler=test_sampler, collate_fn=utils.collate_fn)

    num_classes = 1
    model = ChipNet(num_classes, basePlaneCount=24)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,   # 0.005
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,  # 3
                                                   gamma=0.5)
    img, target = dataset[0]

    num_epochs = 500

    i = 1
    t = 1
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        model.eval()
        with torch.no_grad():
            if len(img.shape) == 3:
                img.unsqueeze_(0)
            prediction = model(img.to(device))
            prediction = prediction[0].cpu().numpy().transpose(1, 2, 0)
            # prediction[prediction > 1] = 1
            # prediction[prediction < 0] = 0
            print(prediction.shape, prediction.dtype, prediction.min(), np.mean(prediction), prediction.max())
        # fig = plt.imshow(np.squeeze(prediction, 2),
        #            vmin=-1, vmax=2, cmap='rainbow');
        # plt.colorbar()
        # plt.show()

     # def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler,
     # device, epoch, print_freq):
        train_one_epoch(model, getCriterion(), optimizer, data_loader, lr_scheduler, \
                        device, epoch, print_freq=10)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device, num_classes)
        # print(x)



import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model