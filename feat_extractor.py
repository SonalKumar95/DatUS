import argparse
import os
from types import SimpleNamespace
import pandas as pd
import shutil
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from PyContrast.pycontrast.networks.build_backbone import build_model
from torch_utils import get_loaders_suimcrop

#device, dtype = 'cuda:0', torch.float32

models = ['resnet50_mocov2']
parser = argparse.ArgumentParser(description='IM')
parser.add_argument('--model', dest='model', type=str, default='resnet50_mocov2', help='Model: one of ' + ', '.join(models))
parser.add_argument('--device', type=str, default='cpu', help='device to be used.')
parser.add_argument('--dtype', type=str, default = torch.float32 , help='device to be used.')
parser.add_argument('--viz_validSegs', type=bool, default=False, help='Visualize valid segments.')
parser.add_argument('--crop_path', default='./outputs/crop_dir', type=str)
parser.add_argument('--split', default='train', type=str, help='dataset split.')
    
args = parser.parse_args()


def get_model(model='resnet50_mocov2'):

    if model == 'resnet50_mocov2':
        args = SimpleNamespace()
        args.jigsaw = False
        args.arch, args.head, args.feat_dim = 'resnet50', 'linear', 2048
        args.mem = 'moco'
        args.modal = 'RGB'
        model, _ = build_model(args)
        cp = torch.load('./checkpoints/MoCov2.pth')

        sd = cp['model']
        new_sd = {}
        for entry in sd:
            new_sd[entry.replace('module.', '')] = sd[entry]
        model.load_state_dict(new_sd, strict=False)  # no head, don't need linear model

        #model = model.to(device=args.device)
        return model
    else:
        raise ValueError('Wrong model')

def eval_swav(model, loader):
    reses = []
    labs = []

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=args.device, dtype=args.dtype), target.to(device=args.device)
	
        output = model.forward(data)
        reses.append(output.detach().cpu().numpy())
        labs.append(target.detach().cpu().numpy())

    rss = np.concatenate(reses, axis=0)
    lbs = np.concatenate(labs, axis=0)
    return rss, lbs


def eval_me(model, loader):
    reses = []
    labs = []
    model.eval()
    
    with torch.no_grad():
    	for batch_idx, (data, target) in enumerate(tqdm(loader)):
        	data, target = data.to(device=args.device, dtype=args.dtype), target.to(device=args.device)

        	output = model.forward(data, mode=2)
        	reses.append(output.detach().cpu().numpy())
        	labs.append(target.detach().cpu().numpy())

    rss = np.concatenate(reses, axis=0)
    lbs = np.concatenate(labs, axis=0)
    
    return rss, lbs

#crop_path = r'./outputs/crop_dir'

def eval_and_save(model='resnet50_mocov2'):
    class_name = []

    print("Loading Model.")
    mdl = get_model(model)
    mdl = mdl.to(device=args.device)
    bs = 32 if model in ['resnet50_infomin'] else 16
    #train_loader, val_loader = get_loaders_imagenet(imagenet_path, bs, bs, 224, 8, 1, 0)
    #obj_loader, _, _, _, _ = get_loaders_objectnet(objectnet_path, imagenet_path, bs, 224, 8, 1, 0)
    
    print("Calling Dataloader.")
    train_loader,dataset = get_loaders_suimcrop(args.crop_path, bs, bs, 224, 1, 1, 0)

    print("Calling Feature Extractor.")
    eval_f = eval_swav if 'swav' in model else eval_me
    train_embs, train_labs = eval_f(mdl, train_loader)
    class_to_idx = dataset.class_to_idx
    
    for lab in tqdm(train_labs):
    	class_name.append(list(class_to_idx.keys())[list(class_to_idx.values()).index(lab)]+'.jpg')
    	#print("Image:", image)
    	#print("Label:", class_name)


    np.savez(os.path.join('./outputs', model + '.npz'), train_embs=train_embs, train_labs=class_name)

if not os.path.exists(args.crop_path):
    print("Filtering valid segments...")
    #Filter valid segments [line 112 to 159]
    df_plus = pd.read_csv('./outputs/{}_GroupPlus.csv'.fromat(args.split)) #list object with lenth of group less than equal to 5
    train_list = [] #store valid segment crops name
    noise_list = [] #store noisy segment crops name

    for i in tqdm(range(len(df_plus['Annotation']))):
        image_name = df_plus['Annotation'][i]
        group = eval(df_plus['groups'][i])
        
        for g in range(len(group)):
            img_name = image_name.rstrip('.jpg') + '_' + str(g) + '.jpg'
            
            #f img_name in df_sim['Annotation'].tolist():
            #if len(group[g])<=5: #for noise
            if len(group[g])>5: #for filtered
                train_list.append(img_name)
            else:
                noise_list.append(img_name)
                
                    
    dict1 = {'Annotation': train_list}
    df = pd.DataFrame(dict1)
    dict2 = {'Annotation': noise_list}
    df_n = pd.DataFrame(dict2)

    source = r"./outputs/{}_CropViewPlus".format(args.split)

    if args.viz_validSegs:
        target = r"./outputs/Filter_crops"
        if not os.path.exists(target):
        	os.mkdir(target)

        for t in range(len(train_list)):
        	new_source  = os.path.join(source, train_list[t])
        	shutil.copy2(new_source, target)

    df.to_csv('./outputs/FilterList.csv')
    df_n.to_csv('./outputs/NoiseList.csv')

    target1 = args.crop_path

    if not os.path.exists(target1):
        os.mkdir(target1)
    for name in tqdm(os.listdir(target)):
        if not os.path.exists(os.path.join(target1, name.rstrip('.jpg'))):
        	os.mkdir(os.path.join(target1, name.rstrip('.jpg')))
        shutil.copy(os.path.join(target, name),os.path.join(target1, name.rstrip('.jpg')))
else:
    print('Segment crops are already filtered..')
    
eval_and_save(args.model)