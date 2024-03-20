import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy
from bilateral_solver import bilateral_solver_output
#H*W*C
#[0,0,0] Black
#[255, 0, 0] Blue
#[[0, 0, 255], Red
#[[0, 255, 0], Green
#[255, 0, 255] pink
#[[255, 255, 0] cyan/sky
#[0, 255, 255] yellow

parser = argparse.ArgumentParser(description='IM')
parser.add_argument('--mask_path', default='/home/multimedia/VisionLab/DUSS/outputs/val_PseudoMaskLUV_Plus', type=str)
parser.add_argument('--map_path', default='/home/multimedia/VisionLab/DUSS/outputs/Map.csv', type=str)
parser.add_argument('--split', default='train', type=str, help='dataset split.')
parser.add_argument('--n_clusters', type=int, default=6, help='Number of pseudo clusters')
parser.add_argument('--t_clusters', type=int, default=6, help='Number of true clusters')

args = parser.parse_args()
#bs = True
#dir_ = os.path.join(args.path,'{}_PseudoMaskLUV_Plus'.format(args.split))
df_map = pd.read_csv(args.map_path)
Map_ = eval(df_map['map'].tolist()[0])
Map = Map_
#Map = {(0,0),(1,1),(2,2),(3,3),(4,4),(5,5)} #TRUE MASKS
#Map = {(0, 4), (1, 1), (3, 0), (4, 5), (2, 2), (5, 3)} #(P,T) PSEUDO MAKS

if args.t_clusters != args.n_clusters:
    Map = set()
    centroide = [[0]]*args.t_clusters
    #print(centroide)
    temp = np.load('/home/multimedia/VisionLab/DUSS/outputs/centroid_{}.npy'.format(args.n_clusters))
    #print(temp[0])
    flag = []
    for (p,t) in Map_:
        if t < args.t_clusters:
            Map.add((p,t))
            flag.append(p)
            #print(p)
            #print(temp[p])
            centroide[t] = temp[p]

    for p in range(args.n_clusters):
        if p not in flag:
            # Calculate distances
            distances = np.linalg.norm(centroide - temp[p], axis=1)

            # Find the indices of the 3 nearest neighbors
            nearest_indices = np.argsort(distances)[:1]
            #print(nearest_indices[0])
            Map.add((p,nearest_indices[0]))

print(Map)
if args.n_clusters == 6:
    Color = [[255, 0, 0],[0, 0, 255],[0, 255, 255],[255, 0, 255],[255, 255, 0],[255,255,255]]
    #['BLUE', 'RED', 'YELLOW', 'MAGENTA', 'CYAN', 'WHITE']
elif args.n_clusters == 27:
    Color = [[64, 64, 64], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128],[64, 0, 0],[192, 0, 0], 
    [64, 128, 0],[192, 128, 0],[64, 0, 128],[192, 0, 128],[64, 128, 128], [192, 128, 128], 
    [0, 64, 0],[128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [128, 64, 128],
    [0, 192, 128], [128, 192, 128], [64, 64, 0], [192, 64, 0], [64, 192, 0],[192, 192, 0]]
    

mask_list = os.listdir(args.mask_path)

#new_dir = os.path.join('pseudo_masks','Classification/LUV/Trans',folder.rstrip('_Plus')+'_color')
new_dir = os.path.join('./temp/{}_color_mask'.format(args.split))

if not os.path.exists(new_dir):
    os.mkdir(new_dir)

#color_map = {i:color[i] for i in range(len(color))}
color_map = {p:Color[t] for p,t in Map}
#color_map = {-1:[255,255,255], 0: [0, 0, 0], 1: [255, 0, 0], 2:[0, 255, 0], 3:[0, 0, 255], 4:[255, 255, 0], 5:[255, 0, 255], 6: [0, 255, 255], 7: [128, 0, 0], 8: [0, 128, 0], 9: [0, 0, 128], 10: [128, 128, 128]}

color_map[255] = [0, 0, 0]
#color_map[27] = [0, 0, 0]
#color_map[5] = [255, 255, 255]

for name in tqdm(mask_list):
    gray_mask = cv2.imread(os.path.join(args.mask_path,name), cv2.IMREAD_GRAYSCALE)
    
    # Create an empty color mask with the same size as the grayscale mask
    color_mask = np.zeros((gray_mask.shape[0], gray_mask.shape[1], 3), dtype=np.uint8)

    
    # if bs:
    #     image_path = '/home/multimedia/VisionLab/DUSS/data/suim/val/images'
    #     image = cv2.imread(os.path.join(image_path,name.rstrip('png'))+'jpg')
    #     #true_shape = image.shape
    #     #gray_mask = cv2.resize(gray_mask, (true_shape[1],true_shape[0]),interpolation = cv2.INTER_LINEAR)
    #     image = cv2.resize(image, (224,224),interpolation = cv2.INTER_LINEAR)
       
    #     mask0 = bilateral_solver_output(image, gray_mask)[1]
    #     #print(mask0)
    #     save_dir = './temp/gray_{}'.format(args.split)
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     cv2.imwrite(save_dir+'/'+name, mask0)

    #     # Iterate through each pixel in the grayscale mask and assign the corresponding color from the color map
    #     for i in range(mask0.shape[0]):
    #         for j in range(mask0.shape[1]):
    #             #print(gray_mask[i, j])
    #             color_mask[i, j] = color_map[int(mask0[i, j])]
    #     # Save the color mask
    #     cv2.imwrite(new_dir+'/'+name, color_mask)
    # else:
    # Iterate through each pixel in the grayscale mask and assign the corresponding color from the color map
    for i in range(gray_mask.shape[0]):
        for j in range(gray_mask.shape[1]):
            #print(gray_mask[i, j])
            color_mask[i, j] = color_map[int(gray_mask[i, j])]
    # Save the color mask
    cv2.imwrite(new_dir+'/'+name, color_mask)
print('Folder saved to' + new_dir)
