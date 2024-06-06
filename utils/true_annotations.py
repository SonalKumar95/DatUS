import numpy as np
import cv2
import os
import collections
import argparse
from tqdm import tqdm

def getRobotFishHumanReefWrecks(mask):
    # for categories: HD/0, RO/1, FV/2, WR/4, RI/3
    imw, imh = mask.shape[0], mask.shape[1]
    new_mask = np.zeros((imw, imh),dtype=int)
    #Robot = np.zeros((imw, imh))
    #Fish = np.zeros((imw, imh))
    #Reef = np.zeros((imw, imh))
    #Wreck = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1): #HD(001)
                new_mask[i, j] = 0
                #print('0')
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0): #RO(100)
                new_mask[i, j] = 1
                #print('1') 
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0): #FV(110)
                new_mask[i, j] = 2
                #print('2')
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1): #RI(101)
                new_mask[i, j] = 3
                #print('3')
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1): #WR(011)
                new_mask[i, j] = 4 
                #print('4') 
            else:
                new_mask[i, j] = 5 #bBW.PF,SR(000,010,111)
                #print('5')
    return new_mask

def getAll(mask):
    # for categories: HD/0, RO/1, FV/2, WR/4, RI/3
    imw, imh = mask.shape[0], mask.shape[1]
    new_mask = np.zeros((imw, imh),dtype=int)
    #Robot = np.zeros((imw, imh))
    #Fish = np.zeros((imw, imh))
    #Reef = np.zeros((imw, imh))
    #Wreck = np.zeros((imw, imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1): #HD(001)
                new_mask[i, j] = 0
                #print('0')
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0): #RO(100)
                new_mask[i, j] = 1
                #print('1') 
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0): #FV(110)
                new_mask[i, j] = 2
                #print('2')
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1): #RI(101)
                new_mask[i, j] = 3
                #print('3')
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1): #WR(011)
                new_mask[i, j] = 4 
                #print('4') 
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1): #pf(010)
                new_mask[i, j] = 5
                #print('5')
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1): #sr(111)
                new_mask[i, j] = 6
            else:
                new_mask[i, j] = 7 #BW(000)
                #print('5')
    return new_mask
 
def main(args):
    #splits = ['train','val', 'TEST']
    mask_dir = os.path.join(args.root, '{}'.format(args.split), 'masks')
    true_mask_dir = os.path.join(args.root, '{}'.format(args.split), 'true_masks')
    
    if not os.path.exists(true_mask_dir):
        os.mkdir(true_mask_dir)
        
    for label in tqdm(os.listdir(mask_dir)):
        #print(label)
        mask_path = os.path.join(mask_dir, label)
        mask = cv2.imread(mask_path)
        #print(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        #new_mask = getAll(mask)
        if args.mask_category == 'coarse':
            new_mask = getRobotFishHumanReefWrecks(mask)
        elif args.mask_category == 'all':
            new_mask = getAll(mask) 

        #frequency = collections.Counter(new_mask.flatten())
        #print(frequency)

        #new_mask = getRobotFishHumanReefWrecks(mask)
        #print(label.rstrip('.bmp')+".png")
        cv2.imwrite(f"{true_mask_dir}/"+label.rstrip('.bmp')+".png",new_mask)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IM')
    #parser.add_argument('--device', default='cpu', type=str, help='device to be used.')
    parser.add_argument('--data_dir', type=str, default='./data', help='path to ground truth masks dir')
    parser.add_argument('--split', default='train', type=str, help='dataset split.')
    parser.add_argument('--mask_category', default='coarse', type=str, help='dataset split.')
    args = parser.parse_args()
    main(args)