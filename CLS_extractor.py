import os
import sys
import argparse
import pandas as pd
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from tqdm import tqdm
import utils
import csv
import vision_transformer as vits
import numpy as np
import torch.nn.functional as F
import shutil

'''def l2_normalization(tensor):
    normalized_tensor = []
    for ten in tqdm(tensor):
        norm = np.linalg.norm(ten, ord=2, axis=0, keepdims=True)
        normalized_ten = tensor / norm
        normalized_tensor.append(normalized_ten)
    return normalized_tensor'''

def normalize_batch(batches):
    nor_batch = []
    for batch in tqdm(batches):
        nor_batch.append(nn.functional.normalize(batch, dim=1))
    return torch.cat(nor_batch)

def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.crop_path), transform=transform)
    #dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    '''data_loader_val = torch.utils.data.DataLoader(
                    dataset_val,
                    batch_size=args.batch_size_per_gpu,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False,
                )'''
    print(f"Data loaded with {len(dataset_train)} images.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train/val set...")
    extract_features(model, dataset_train, data_loader_train, args.use_cuda)

@torch.no_grad()
def extract_features(model, dataset_train, data_loader, use_cuda=True, multiscale=False):
    class_to_idx = dataset_train.class_to_idx
    metric_logger = utils.MetricLogger(delimiter="  ")
    #features = None
    #image_name = ['']*len(dataset_train)
    #print('List size: ', len(image_name))

    fields = ['Annotation','CLS']
    with open('./outputs/{}_CLSLUV_Plus.csv'.format(args.split), 'a', newline='') as file: 
            
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader() 

        for samples, index, labels in metric_logger.log_every(data_loader, 10):
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            #print('index:',index)
            #print('labels:',labels)

            #for i in range(len(index)):
            image_names = [list(class_to_idx.keys())[list(class_to_idx.values()).index(lab)]+'.jpg' for lab in labels]
            
            #print('samples:',samples)
            #print('Index:',index)
            if multiscale:
                feats = utils.multi_scale(samples, model)
            else:
                feats = model(samples).clone()

            # get indexes from all processes
            y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                dist.get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()

            #Normalize features
            output_l = nn.functional.normalize(torch.cat(output_l), dim=1, p=2)

            dict_list = [{'Annotation':image_names[i],'CLS': [output_l[i].tolist()]} for i in range(len(image_names))]
            #print(dict_list)
            writer.writerows(dict_list)


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx, lab


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extracting CLS tocken from ViT model')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--crop_path', default='./outputs/crop_dir', type=str)
    parser.add_argument('--split', default='train', type=str, help='dataset split.')
    parser.add_argument('--viz_validSegs', type=bool, default=True, help='Visualize valid segments (set False for large dataset).')

    args = parser.parse_args()

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        #test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        #test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        
        print("Filtering valid segments...")
        #Filter valid segments [line 112 to 159]
        df_plus = pd.read_csv('./outputs/{}_GroupPlus.csv'.format(args.split)) #list object with lenth of group less than equal to 5
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
            target = r"./outputs/Filter_image"
            if not os.path.exists(target):
                os.mkdir(target)

            for t in range(len(train_list)):
                new_source  = os.path.join(source, train_list[t])
                shutil.copy2(new_source, target)
            # Comment our line 220 t0 226 if image-wise visualization is not required
            target1 = args.crop_path
            if not os.path.exists(target1):
                os.mkdir(target1)
            for name in tqdm(os.listdir(target)):
                if not os.path.exists(os.path.join(target1, name.rstrip('.jpg'))):
                    os.mkdir(os.path.join(target1, name.rstrip('.jpg')))
                shutil.copy(os.path.join(target, name),os.path.join(target1, name.rstrip('.jpg')))

        df.to_csv('./outputs/FilterList.csv')
        df_n.to_csv('./outputs/NoiseList.csv')

        print('Extarcting Features.')
        extract_feature_pipeline(args)
        
        dist.barrier()
