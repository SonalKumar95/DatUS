#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
#from utils import unnorm
import cv2

MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3
BGR_MEAN = np.array([104.008, 116.669, 122.675])

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


#normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #coco
unnorm = UnNormalize([0.487, 0.423, 0.248], [0.246, 0.222, 0.221]) #suim

def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor,image_name):
    #print(image_tensor)
    image = np.array(VF.to_pil_image(unnorm(image_tensor.float())))[:, :, ::-1]
    H, W = image.shape[:2]
    image = np.ascontiguousarray(image)
    #print(output_logits.shape)
    output_probs = F.interpolate(output_logits.float().unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear",
                                  align_corners=False)
    #output_probs = F.softmax(output_logits, dim=0).cpu().numpy()
    #print(output_probs.shape)
    c = output_probs[0].shape[0]
    h = output_probs[0].shape[1]
    w = output_probs[0].shape[2]

    U = torch.flatten(output_probs).unsqueeze(0)
    #U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=image, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    cv2.imwrite(f"./outputs/crf_out/"+image_name,torch.tensor(Q).permute(1, 2, 0).numpy())

    return Q
