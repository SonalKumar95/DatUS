import os
import cv2
import math
import torch
import numpy
import requests
import numpy as np
from PIL import Image
import torch.nn.functional as nnf
from torchvision.utils import save_image
import torchvision.transforms as T
from torchvision.transforms.functional import crop

#root = "/home/ai-lab/AI_LAB/Segmentation/SUIM_LUV"

def view_save(F, imag,label,groups,split,patch_size):

    imag = cv2.resize(imag, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
    #print(imag.size)
    
    tenso = torch.tensor(imag)
    tenso = tenso.to(torch.float32)
    tenso = torch.permute(tenso, (2, 0, 1))
    
    sub_images(F, tenso,label,groups,split,patch_size)

class UnmaskPatches(object):
    def __init__(self, patch_size, mark):
        self.ps = patch_size
        self.mark = mark
    def __call__(self, x):
        
        # divide the batch of images into non-overlapping patches
        u = nnf.unfold(x, kernel_size=self.ps, stride=self.ps, padding=0)
        #black = torch.zeros(u.shape[0],u.shape[1])
        #white = torch.full((u.shape[0],u.shape[1]), 255)
        white = torch.full((u.shape[0],u.shape[1]), 0)
        
        #unmask non-object patches
        for i in range(u.shape[2]):
            if i not in self.mark:
                u[...,i] = white
        #fold all the patches
        f = nnf.fold(u, x.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
        return f

def sub_images(F, tenso,label,groups,split,patch_size):
    c = 0 #sub_group number to save partision
    factor = (224/patch_size)
   
    for g in groups:
        # Unmaskin patches
        ff = UnmaskPatches(patch_size,g)
        fff = ff(tenso[None])
        fff = fff.to(torch.int)
        
        #view object
        #plt.imshow(fff[0].detach().permute(1, 2, 0))
       
        #crop,reize and save objects
        xx = [(a%factor) for a in g]
        yy = [math.floor(a/factor) for a in g]
        min_x = min(xx)
        max_x = max(xx)
        min_y = min(yy)
        max_y = max(yy)

        #tl_patch = (min_y * factor) + min_x
        width = (max_x-min_x+1)*patch_size
        height = (max_y-min_y+1)*patch_size
        
        crop1 = crop(fff[0],int((min_y)*patch_size) ,int((min_x)*patch_size),int(height),int(width))
         
        #save image
        image_path = os.path.join('./outputs/CropView') #for single image testing
        image_path1 = os.path.join("./outputs/{}_CropView".format(split))
        image_path2 = os.path.join("./outputs/{}_CropViewPlus".format(split))
        
        
        #os.mkdir(image_path)
        image_name = label.rstrip('.jpg')+str('_')+str(c)
        #print(crop1.shape)
        if F == 'sample':
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            #cv2.imwrite(f"{image_path}/"+image_name+".jpg",crop1.detach().permute(1, 2, 0).numpy())
            cv2.imwrite(f"{image_path}/"+image_name+".jpg",crop1.detach().permute(1, 2, 0).numpy())
        if F == 0:
            if not os.path.exists(image_path1):
                os.mkdir(image_path1)
            #cv2.imwrite(f"{image_path}/"+image_name+".jpg",crop1.detach().permute(1, 2, 0).numpy())
            cv2.imwrite(f"{image_path1}/"+image_name+".jpg",crop1.detach().permute(1, 2, 0).numpy())
        if F == 1:
            if not os.path.exists(image_path2):
                os.mkdir(image_path2)
            cv2.imwrite(f"{image_path2}/"+image_name+".jpg",crop1.detach().permute(1, 2, 0).numpy())
        c+=1