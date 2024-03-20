import cv2
import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as nnf
from tqdm import tqdm 
import argparse
 
def combine_patch(image_name,initial,pseudo_labels,patch_size):
    
    white = torch.full((224,224), 255)
    #white = torch.full((336,336), 255)
    white = white[None][None].to(torch.float32)
    u = nnf.unfold(white, kernel_size=patch_size, stride=patch_size, padding=0)
     
    for i in range(len(initial)):
        for j in range(len(initial[i])):
             
            if pseudo_labels[i] != -1:
            	u[...,initial[i][j]] = (torch.full((patch_size,patch_size), pseudo_labels[i])).view(1,-1)
       
    f = nnf.fold(u, white.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
     
    return f


def pseudo_map(split,patch_size,f_type,n_clusters):
    
    path_G = os.path.join('outputs','{}_Group{}.csv'.format(split, f_type))
    df_g = pd.read_csv(path_G)
    
    path_pl = os.path.join('./outputs/{}_PLabelLUV{}_{}.csv'.format(split,n_clusters,f_type)) #pseudo_labels path
    df_pl = pd.read_csv(path_pl)
    
    annotation_dir = os.path.join('outputs','{}_PseudoMaskLUV{}_{}'.format(split,n_clusters,f_type)) #here
    print('Go to ',annotation_dir)

    if not os.path.exists(annotation_dir):
            os.mkdir(annotation_dir)
            
    #print(df_pl.head)
    for i in tqdm(range(len(df_g['groups']))):
        initial = df_g['groups'][i]
        
        image_name = df_g['Annotation'][i]
        pseudo_labels = []
        
        for j in range(len(eval(initial))):
            subImage_name = ''.join([image_name[:-4],'_',str(j),'.jpg'])
            pseudo_labels.append(df_pl[df_pl['Annotation'] == subImage_name]['Label'].values[0]) #Switch between label, pred_label and 6label
            
        #print(pseudo_labels)
        pseudo_map_tensor = combine_patch(image_name,eval(initial),pseudo_labels,patch_size)
        
        annotation_path = os.path.join(annotation_dir,image_name[:-4]+'.png')
        cv2.imwrite(annotation_path,pseudo_map_tensor[0].to(torch.int).detach().permute(1, 2, 0).numpy())
        

def main(args):

    fields = ['Annotation', 'groups']
    for n_clust in range(27,args.n_clusters,4):
        pseudo_map(args.split, args.patch_size,args.f_type,n_clust)
    '''pseudo_map(args.split, args.patch_size,args.f_type,args.n_clusters)'''
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('GENERATE PSEUDO MASKS.')
    parser.add_argument('--device', default='cpu', type=str, help='device to be used.')
    parser.add_argument('--split', default='val', type=str, help='dataset split.')
    parser.add_argument('--patch_size', default=8, type=int, help='dataset split.')
    #parser.add_argument('--root', default="", type=str, help='path to dataset.')
    parser.add_argument('--f_type', default="Plus", type=str, help='crop type indicator.')
    parser.add_argument('--n_clusters', type=int, default=6, help='Number of clusters')
    args = parser.parse_args()
    
    main(args)
