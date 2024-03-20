import argparse
import gc
import os
import time
import pandas as pd
import numpy as np
import sklearn
import collections
import shutil
from scipy.optimize import linear_sum_assignment
from sklearn import cluster
from sklearn.datasets import make_blobs
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import trange, tqdm
#from torchvision import datasets

from torch_utils import get_loaders_objectnet

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='IM')
parser.add_argument('--model', dest='model', type=str, default='resnet50_mocov2',
                    help='Model: one of' + ', '.join('model_names'))
#parser.add_argument('--n-components', type=int, default=None, help='Number of components for PCA')
parser.add_argument('--n_clusters', type=int, default=6, help='Number of clusters')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for k-means')
parser.add_argument('--save_clusters', type=bool, default=False, help='Save clusters.')
parser.add_argument('--cropdir_path', type=str, default=r'./outputs/Filter_image', help='If save clusters is true.')
parser.add_argument('--csv_data', type=bool, default= True, help='If CLS then true else false.')
parser.add_argument('--n-components', type=int, default=None, help='Number of components for PCA')
parser.add_argument('--split', default='val', type=str, help='dataset split.')
    
args = parser.parse_args()
 
def save_clusters_(df,n_clusters):
    #df = pd.read_csv('{}_CLS{}.csv'.format(split,f_type))
    Annotations = df['Annotation'].tolist()
    label_list = df['Label'].tolist()
    #source = os.path.join(root,'coco',"{}_CocoLUV_{}".format(split,f_type))
    target_path = './outputs/seg_clusters_{}'.format(n_clusters)
    if not os.path.exists(target_path):
    	os.mkdir(target_path)
    #target = os.path.join(tar,"{}_ClustLUV{}_{}".format(split,n_clusters,f_type))

    for i in tqdm(range(len(Annotations))):
        new_target = os.path.join(target_path, str(label_list[i]))

        #creating folder for each clustr inside target
        if not os.path.exists(new_target):
            os.mkdir(new_target)

        new_source = os.path.join(args.cropdir_path, Annotations[i])
        shutil.copy2(new_source, new_target)
    print('Cluster file saved at ' + target_path)


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def train_pca(X_train):
    #print(X_train.shape)
    bs = max(4096, X_train.shape[1] * 2)
    #print(bs)
    #bs = max(64, X_train.shape[1] * 2)
    transformer = IncrementalPCA(batch_size=bs)  #
    for i, batch in enumerate(tqdm(batches(X_train, bs), total=len(X_train) // bs + 1)):
        #print(batch.shape)
        transformer = transformer.partial_fit(batch)
        # break
    #print(transformer.explained_variance_ratio_.cumsum())
    return transformer


def cluster_data(X_train, y_train, n_clusters):
    batch_size = max(2048, int(2 ** np.ceil(np.log2(n_clusters))))
    #seed = np.random.rand(n_clusters ,X_train.shape[1])
    #minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_no_improvement=None)
    minib_k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_no_improvement=None, init='random')

    # TODO: save to csv
    for e in trange(args.epochs):
        #print('epoch: {}'.format(e))
        #print(X_train)
        #print(y_train)
        X_train, y_train = shuffle(X_train, y_train)
        for batch in batches(X_train, batch_size):
            minib_k_means = minib_k_means.partial_fit(batch)
            
    centroids = minib_k_means.cluster_centers_
    np.save('./outputs/centroid_{}.npy'.format(n_clusters),centroids,allow_pickle=True)
    pred = minib_k_means.predict(X_train)
    frequency = collections.Counter(pred)
    print(frequency)
    #print(y_train)
    
    y_train = y_train.tolist()
    
    dict_ = {'Annotation': y_train, 'Label': pred}
    df = pd.DataFrame(dict_)
    if args.save_clusters:
    	save_clusters_(df,n_clusters)
    
    df_n = pd.read_csv('./outputs/coco_val/NoiseList.csv')
    df_n['Label'] = [-1] * len(df_n['Annotation'])
    
    df_f = pd.concat([df, df_n],join='outer')
    
    df_f.to_csv('./outputs/{}_PLabelLUV{}_Plus.csv'.format(args.split,n_clusters)) 
    #save_clusters(df)

def transform_pca(X, transformer):
    #print(len(X))
    n = max(4096, X.shape[1] * 2)
    #n = max(64, X.shape[1] * 2)

    for i in trange(0, len(X), n):
        #print(X[i:i + n].shape)
        X[i:i + n] = transformer.transform(X[i:i + n])
        # break
    return X


generate = False
#csv_data = True

if generate:
    pass 
else:
    if args.csv_data:
        filename = './outputs/' + 'TransCLS' + '_pca.npz'
    else:
        filename = './outputs/' + args.model + '_pca.npz' #CNN
        #filename = './scheme/results/' + 'TransCLS' + '_pca.npz' #Transformer
    
    if not os.path.exists(filename):

        t0 = time.time()
        if args.csv_data:
            #path = '/raid/workspace/sonal/sonalk/Unsupervised-ClassificationV1/coco/' + 'train_CLSLUV_Plus.csv'
            #noise_path = '/raid/workspace/sonal/sonalk/Unsupervised-ClassificationV1/coco/' + 'NoiseList.csv'
            
            path = './outputs/coco_val/{}_CLSLUV_Plus.csv'.format(args.split)
            noise_path = './outputs/coco_val/NoiseList.csv'
            
            data = pd.read_csv(path)
            noise = pd.read_csv(noise_path)
            df_ = data[~data.Annotation.isin(noise['Annotation'].tolist())]

            #X_train, y_train, X_test, y_test, X_test2, y_test2 = data['train_embs'], data['train_labs'], data['val_embs'], data['val_labs'], data['obj_embs'], data['obj_labs']
            
            X_train, y_train  = np.array(df_['CLS'].tolist()), np.array(df_['Annotation'].tolist())
            X_train = np.array([eval(x) for x in X_train])
        else:
            path = './outputs/' + args.model + '.npz' #for moco
            #path = './scheme/results/' + 'TransCLS' + '.npz' #for Transformer
            data = np.load(path)
            
            X_train, y_train  = data['train_embs'], data['train_labs']
        
        t1 = time.time()
         
        print('Loading time: {:.6f}'.format(t1 - t0))
        if len(y_train.shape) > 1:
            y_train = y_train.argmax(1)
        X_train, y_train = X_train.squeeze(), y_train.squeeze()
        print('train_shape: ',X_train.shape)

        if X_train.shape[0]>X_train.shape[1]:
            transformer = train_pca(X_train)
            X_train = transform_pca(X_train, transformer)
        
        gc.collect()

        if not args.csv_data:
            np.savez(filename, train_embs=X_train, train_labs=y_train, PCA=transformer)
    else:
        t0 = time.time()
        data = np.load(filename)
        print(filename)
        X_train, y_train = data['train_embs'], data['train_labs']
        
        # print(y_test2.shape, y_test2, y_test2.max())
        '''if len(y_test2.shape) > 1:
            y_test2 = y_test2.argmax(1)'''
        t1 = time.time()
        print('Loading time: {:.6f}'.format(t1 - t0))

    if args.n_components is not None:
        X_train  = X_train[:, :args.n_components]
        
     
    for n_clust in range(27,args.n_clusters,4):
        print('Working on {}'.format(n_clust))
        cluster_data(X_train, y_train, n_clust)
     
    '''print('Working on {}'.format(args.n_clusters))
                cluster_data(X_train, y_train, args.n_clusters)'''
    