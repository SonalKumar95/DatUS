import os
import cv2
import csv 
import torch
import requests
import argparse
import pandas as pd
from os import listdir
import torch.nn as nn
from PIL import Image
from queue import Queue
from vision_transformer_pytorch import vision_transformer as vits
from ipynb.fs.full.save_crops import view_save
from transformers import ViTFeatureExtractor, ViTModel
import torchvision.transforms as transforms
from tqdm import tqdm
from IPython.display import SVG
import numpy as np
#from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.clustering import Louvain, get_modularity
from sknetwork.linalg import normalize
from sknetwork.utils import get_membership
from sknetwork.visualization import svg_graph, svg_bigraph


def qkv(model,inputs):
    
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

    # Forward pass in the model
    attentions = model.get_last_selfattention(inputs['pixel_values'])
    #print('attension_shape = ',attentions.shape)

    # Dimensions
    nb_im = attentions.shape[0]  # Batch size
    nh = attentions.shape[1]  # Number of heads
    nb_tokens = attentions.shape[2]  # Number of tokens

 
    # Extract the qkv features of the last attention layer
    qkv = (feat_out["qkv"].reshape(nb_im, nb_tokens, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4))
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    #print('qkv_shape= ',qkv.shape)
    #print('q_shape = ',q.shape)
    #print('k_shape = ',k.shape)
    #print('v_shape = ',v.shape)
    return q,k,v


def independent_group(related_pixels,A_copy,patch_size):
    
    factor = (224/patch_size)
    related_pixels_plus = []
    
    ng = [-1, 1, -(factor-1), (factor-1), -factor, factor, -(factor+1), (factor+1)] #search window for non-edge patches
    ng_0 = [1, -(factor-1), -factor, factor, (factor+1)]  #search window for left-most edge patches
    ng_13 = [-1, (factor-1), -factor, factor, -(factor+1)]  #search window for right-most edge patches
    
    
    for rp in related_pixels:
        #print(rp)
        rp = torch.tensor(rp)
        q = Queue(maxsize = (factor)**2) #for bfs search of patch
        Flag = torch.zeros(rp.shape[0],dtype=int) #mark for seen patches
        #print(Flag)
    
        while(torch.min(Flag).item() == 0): # itterate over connected components
            connected_component = torch.tensor([])
            root = torch.argmin(Flag).item() #select first unseen pixel 
            Flag[root] = 1
            connected_component = torch.cat((connected_component,rp[root][None]),0)
        
            if rp[root] % factor == 0:
                NG = ng_0
            elif (rp[root] + 1) % factor == 0:
                NG = ng_13
            else:
                NG = ng
            
            for n in NG: #itterates over neighbour patches and push to the queue
                if (rp[root]+n) >= 0 and (rp[root]+n) <= A_copy.shape[0] - 1:
                    if len((rp == (rp[root]+n)).nonzero(as_tuple=True)[0]) != 0:
                        index = (rp == (rp[root]+n)).nonzero(as_tuple=True)[0]
                        if Flag[index].item() == 0:
                            #print('here')
                            q.put(rp[root]+n)
                            Flag[index] = 1
            #print(q.queue)
            while not q.empty(): #itterates over child patches of main root patches
                root_node = q.get() #deque fist patch from queue
                Flag[(rp == (root_node)).nonzero(as_tuple=True)[0].item()] = 1
                #print('a =',Flag[(rp == (root_node)).nonzero(as_tuple=True)[0].item()])
                connected_component = torch.cat((connected_component,root_node[None]),0)
            
                if root_node % factor == 0:
                    NG = ng_0
                elif (root_node + 1) % factor == 0:
                    NG = ng_13
                else:
                    NG = ng
            
                for n in NG:
                    #print(root_node)
                    if (root_node + n) >= 0 and (root_node + n) <= A_copy.shape[0] - 1:
                        if len((rp == (root_node+n)).nonzero(as_tuple=True)[0]) != 0:
                            index = (rp == (root_node+n)).nonzero(as_tuple=True)[0]
                            if not Flag[index].item():
                                q.put(root_node+n)
                                Flag[index] = 1
                                
            related_pixels_plus.append(connected_component.int()) #append connected patches of a group
    return related_pixels_plus

def main(args):
    
    fields = ['Annotation', 'groups']
    data_path = os.path.join(args.data_dir,args.split,'images')
    list_dir = os.listdir(data_path)

    model = torch.hub.load('facebookresearch/dino:main', 'dino_{}{}'.format(args.vit_size,str(args.patch_size)))

    for label in tqdm(list_dir):
        img_path = os.path.join(data_path, label)
        #image = cv2.imread(r'/home/sonalkumar/VisionLab/Segmentation/Transformer/train2017/000000000025.jpg')
        image = cv2.imread(img_path)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-{}{}'.format(args.vit_size,str(args.patch_size)))
        inputs = feature_extractor(images=image, return_tensors="pt")
         
        model = model.to(args.device)
        inputs = inputs.to(args.device)

        q,k,v = qkv(model,inputs)
        layer_norm = nn.LayerNorm(k.size()[1:]).to(args.device)
        #feats = layer_norm(k)
        feats = k
        feats = feats[:,1:,:]
        A = (feats @ feats.transpose(1, 2)).squeeze()
        A_copy = A.clone().detach()
        
        adj = np.zeros((A.shape[0], A.shape[1]))

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i][j] > 0:
                    adj[i][j] = 1
                else:
                    adj[i][j] = 0

        louvain = Louvain()#.to(device)
        #labels = louvain.fit_transform(adj)
        labels = louvain.fit_predict(adj)
        
        
        labels_unique, counts = np.unique(labels, return_counts=True)
        #print(labels_unique, counts)
        
        related_pix = []
        for u in range(len(labels_unique)):
            temp = []
            for l in range(len(labels)):
                #print(labels[l])
                #print(labels_unique[u])
                if labels[l] == labels_unique[u]:
                    temp.append(l)          
            related_pix.append(temp)

        related_pix_plus = independent_group(related_pix,A_copy,args.patch_size)
         
        view_save(0, image,label,related_pix,args.split,args.patch_size)
        view_save(1, image,label,related_pix_plus,args.split,args.patch_size)


        for i in range(len(related_pix)):
            related_pix[i] = related_pix[i]
        for j in range(len(related_pix_plus)):
            related_pix_plus[j] = related_pix_plus[j].tolist() 
        #print(related_pix)


        mydict1 ={'Annotation': label, 'groups': related_pix}
        mydict2 ={'Annotation': label, 'groups': related_pix_plus}


        with open('./outputs/{}_Group.csv'.format(args.split), 'a', newline='') as file: 

            writer = csv.DictWriter(file, fieldnames = fields)

            if args.head == 0:
                writer.writeheader() 
                args.head = 1

            writer.writerow(mydict1)

        with open('./outputs/{}_GroupPlus.csv'.format(args.split), 'a', newline='') as file_plus: 

            writer = csv.DictWriter(file_plus, fieldnames = fields)

            if args.head_plus == 0:
                writer.writeheader() 
                args.head_plus = 1

            writer.writerow(mydict2)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Data-driven Unsupervised Semantic Segmentation with Pre-trained Self-supervised Vision Transformer')
    parser.add_argument('--device', default='cpu', type=str, help='device to be used.')
    parser.add_argument('--head', default=0, type=int, help='to add header in csv file.')
    parser.add_argument('--head_plus', default=0, type=int, help='to add header in csv file.')
    parser.add_argument('--split', default='train', type=str, help='dataset split.')
    parser.add_argument('--patch_size', default=8, type=int, help='dataset split.')
    parser.add_argument('--data_dir', default="./data", type=str, help='path to dataset.')
    parser.add_argument('--vit_size', default="vitb", type=str, help='vision transformer size:s/b/l.')
    
    args = parser.parse_args()
    if not os.path.exists('./outputs'):
        os.mkdir('outputs')
    seen_pixels = torch.tensor([])
    related_pixels = []
    main(args)
