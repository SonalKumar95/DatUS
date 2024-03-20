import cv2
import os
import torch
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import collections
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
#from PIL import Image
#from sklearn import metrics
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
#import torch.nn.functional as nnf
#import torchvision.transforms as transforms
#from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from crf import dense_crf

def H_match(flat_preds, flat_trues, k_pred, k_true):
     
    k_num1 = k_pred
    k_num2 = k_true
    k_num = max(k_num1,k_num2)
    #num_correct = np.array([[]])
    num_correct = np.zeros((k_num, k_num))
    num_samples = flat_trues.shape[0]
    
    for i in tqdm(range(k_num)):
        #print('*'*(i+1))
        #temp = [[]]
        for j in range(k_num):
            votes = int(((flat_preds == i) * (flat_trues == j)).sum())
            #temp[0].append(votes)
            num_correct[i,j] = votes
    
    match = linear_assignment(num_samples - num_correct)
    #print(match)
    res = set(zip(match[0],match[1]))
    print(res)
    return res

def plot_cm(C,n_c):
    sns.heatmap(C,
            annot=False,
            fmt='g')
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    #plt.show()
    plt.savefig("./outputs/confusion_matrix_{}.pdf".format(n_c), format="pdf", bbox_inches="tight")
    plt.close()

def get_result_metrics(y_true, y_pred, args, n_clusters,t_clusters):
    #n_clusters = args.n_clusters
    #t_clusters = args.t_clusters
    end = t_clusters
    histogram = confusion_matrix(y_true, y_pred)
    plot_cm(histogram,n_clusters)
 
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp 
    iou = tp / (tp + fp + fn)
    prc = tp / (tp + fn)
    opc = np.sum(tp) / np.sum(histogram)
    f1 = tp / (tp + (1/2)*(fp + fn))

    result = {"iou": iou[:end],
              "mean_iou": np.nanmean(iou[:end]),
              "precision_per_class (per class accuracy)": prc[:end],
              "mean precision_per_class (per class accuracy)": np.nanmean(prc[:end]),
              "overall_precision (pixel accuracy)": opc,"f1": f1[:end],"mf1":np.nanmean(f1[:end])}
    result = {k: 100*v for k, v in result.items()}
    
    print('Iou = ',result['iou'])
    print('Mean_iou = ',result['mean_iou'])
    #print('precision_per_class = ',result['precision_per_class (per class accuracy)'])
    #print('mean precision = ',result['mean precision_per_class (per class accuracy)'])
    print('pixel accuracy = ',result['overall_precision (pixel accuracy)'])
    #print('f1 = ',result['f1'])
    print('Mf1 = ',result['mf1'])
    return result


def data_prep(data_path,split,f_type,args,n_clusters,t_clusters):
    #n_clusters = args.n_clusters
    #t_clusters = args.t_clusters
    
    true_mask_dir = os.path.join(data_path,split,'true_masks')
    
    #pseudo_mask_dir = os.path.join('./outputs/suim_val','{}_PseudoMaskLUV{}_{}'.format(split,n_clusters,f_type))
    pseudo_mask_dir = os.path.join('./utils/temp','{}_gray_mask'.format(split))
   
    img_dir = os.path.join(data_path,split,'images')

    if not os.path.exists('./outputs/npy/true_{}.npy'.format(n_clusters)) and './outputs/npy/pred_{}.npy'.format(n_clusters):
        print('Creating True AND Pseudo List')
        true_all_ = []
        pseudo_all_ = []
    
        true_all = []
        pseudo_all = []
        
        #print('Creating TRUE and PRIDICTED list')
        #for image_name in tqdm(os.listdir(true_mask_dir)):
        for image_name in tqdm(os.listdir(pseudo_mask_dir)):
            
            true_img_path = os.path.join(true_mask_dir,image_name)
            true_label = cv2.imread(true_img_path,0)
            true_shape = true_label.shape
            true_list = torch.flatten(torch.tensor(true_label)).tolist()
            true_all_ += true_list
            
            pseudo_img_path = os.path.join(pseudo_mask_dir,image_name)
            pseudo_label = cv2.imread(pseudo_img_path,0)


            if args.crf:
                img = cv2.imread(os.path.join(img_dir,image_name.rstrip('.png')+'.jpg'))
                pseudo_label = dense_crf(torch.tensor(img).permute(2,0,1),torch.tensor(pseudo_label),image_name)
            else:
                pseudo_label = cv2.resize(pseudo_label, (true_shape[1],true_shape[0]),interpolation = cv2.INTER_LINEAR)


            pseudo_list = torch.flatten(torch.tensor(pseudo_label)).tolist()
            pseudo_all_ +=pseudo_list
        
        print('Removing Noise Pixels (For COCO activate if condition)')
        for i in tqdm(range(len(pseudo_all_))):
            if pseudo_all_[i] != 255:
                if true_all_[i] != 27: #only for coco corse
                    true_all.append(true_all_[i])
                    pseudo_all.append(pseudo_all_[i])
                
        '''if not os.path.exists('./outputs/npy'):
                                    os.mkdir('./outputs/npy')'''

        '''np.save('./outputs/npy/true_{}.npy'.format(n_clusters),true_all,allow_pickle=True)
        np.save('./outputs/npy/pred_{}.npy'.format(n_clusters),pseudo_all,allow_pickle=True)'''
    else:
        print('Loading True and Pseudo List without noise')
        true_all = np.load('./outputs/npy/true_{}.npy'.format(n_clusters))
        pseudo_all = np.load('./outputs/npy/pred_{}.npy'.format(n_clusters))
    
    '''true_all = torch.tensor(true_all,dtype=torch.int32).to(device)
    pseudo_all = torch.tensor(pseudo_all,dtype=torch.int32).to(device)'''
    
    true_all = torch.tensor(true_all,dtype=torch.int32).cpu()
    pseudo_all = torch.tensor(pseudo_all,dtype=torch.int32).cpu()

    print('Num_true: ',true_all.shape[0])
    print('Num_pseudo: ',pseudo_all.shape[0])
    
    #Mapping pseudo_all labels to true_all labels
    if not os.path.exists('./outputs/npy/Rpred_{}.npy'.format(n_clusters)):
        
        #res = H_match(pseudo_all,true_all,n_clusters, t_clusters)
        res = {(1, 0), (2, 4), (0, 5), (3, 1), (5, 3), (4, 2)}
        
        recorded_pseudo = torch.zeros(pseudo_all.shape[0], dtype=pseudo_all.dtype).cpu()
        
        #Converting pseudo_all label using map
        print('Converting pseudo_all label using map')
        for pred_i, target_i in tqdm(res):
            selected = (pseudo_all == pred_i)
            recorded_pseudo[selected] = target_i
        
        #recorded_pseudo = recorded_pseudo.to(device)
        recorded_pseudo = recorded_pseudo.cpu()
        
        '''np.save('./outputs/npy/Rpred_{}.npy'.format(n_clusters),recorded_pseudo,allow_pickle=True)'''
        map_ = [[item[0] for item in res], [item[1] for item in res]]
        '''np.save('./outputs/npy/Map_{}.npy'.format(n_clusters),map_,allow_pickle=True)'''
    else:
        recorded_pseudo = torch.tensor(np.load('./outputs/npy/Rpred_{}.npy'.format(n_clusters)))
        map_ = np.load('./outputs/npy/Map_{}.npy'.format(n_clusters))
        res = zip(map_[0],map_[1])
        #print('Map = ',res)

    
    #Minimizing confusin matrix like [Drive&Segment]
    if n_clusters > t_clusters:
        print('Minimizing confusin matrix like [Drive&Segment], if psudo_clusters > true_clusters.')
        t = true_all[None]
        p = recorded_pseudo[None]
        for i in tqdm(range(t_clusters, n_clusters)):
            #print(i)
            selected = (p == i)
            #print(selected)
            t = t[:,~selected[0]]
            p = p[:,~selected[0]]
        true_all = t[0]
        recorded_pseudo = p[0]
        print('Reduced number of true pixels:',len(true_all))
        print('Reduced number of predicted pixels:',len(recorded_pseudo))

    print('Computing Result Matrix')
    result_metrix = get_result_metrics(true_all.cpu(), recorded_pseudo.cpu(),args,n_clusters,t_clusters)
     
    '''nmi = normalized_mutual_info_score(true_all.cpu(), recorded_pseudo.cpu())
    print('NMI = ', nmi)
    ari = adjusted_rand_score(true_all.cpu(), recorded_pseudo.cpu())
    print('ARI = ', ari)'''
    
    nmi=ari=0
    return nmi,ari, result_metrix,res
    
def main(args):
    Map = []
    Iou = []
    Mean_iou = []
    precision_per_class = []
    mean_precision = []
    pixel_accuracy = []
    NMI = []
    ARI = []
    
    fields = ['Clusters','Iou', 'Mean_iou', 'precision_per_class','mean_precision','pixel_accuracy','NMI','ARI','f1','Mf1']
    with open('./outputs/suim_val/Result.csv', 'a', newline='') as file: 
            
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader() 

        for n_clust in range(6, args.n_clusters): #(27, args.n_clusters,4):
            print('#Cluster: ', n_clust)
            NMI_, ARI_, result,res = data_prep(args.data_dir,args.split,args.f_type,args,n_clust,args.t_clusters)
            mydict = {'Clusters':n_clust,'Iou': result['iou'], 'Mean_iou':result['mean_iou'],'precision_per_class':result['precision_per_class (per class accuracy)'],'mean_precision':result['mean precision_per_class (per class accuracy)'],'pixel_accuracy':result['overall_precision (pixel accuracy)'],'NMI':NMI_,'ARI':ARI,'f1':result['f1'],'Mf1':result['mf1']}
            Map.append(res)
            writer.writerow(mydict)

        '''print('\n')
                                print('#Cluster: ', args.n_clusters)
                                NMI_, ARI_, result,res = data_prep(args.data_dir,args.split,args.f_type,args,args.n_clusters,args.t_clusters)
                                mydict = {'Clusters':args.n_clusters,'Iou': result['iou'], 'Mean_iou':result['mean_iou'],'precision_per_class':result['precision_per_class (per class accuracy)'],'mean_precision':result['mean precision_per_class (per class accuracy)'],'pixel_accuracy':result['overall_precision (pixel accuracy)'],'NMI':NMI_,'ARI':ARI,'f1':result['f1'],'Mf1':result['mf1']}    
                                Map.append(res)
                                writer.writerow(mydict)
                                print('\n')'''
        
    dictt = {'map':Map}
    dff = pd.DataFrame(dictt)
 
    dff.to_csv('./outputs/suim_val/Map_bs.csv')
    print('2 csv sdaved at {}'.format('./outputs'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IM')
    parser.add_argument('--device', default='cpu', type=str, help='device to be used.')
    parser.add_argument('--n_clusters', type=int, default=6, help='Number of pseudo clusters')
    parser.add_argument('--t_clusters', type=int, default=6, help='Number of true clusters')
    parser.add_argument('--data_dir', type=str, default=r'/home/ai-lab/AI_LAB/Segmentation/SUIM_LUV', help='path to ground truth masks dir')
    parser.add_argument('--crf', default=False, type=bool, help='apply cfr to pseudo mask.')
    parser.add_argument('--split', default='train', type=str, help='dataset split.')
    parser.add_argument('--f_type', default="Plus", type=str, help='crop type indicator.')
   
    args = parser.parse_args()
    
    main(args)