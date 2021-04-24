import json
#import requests
from pathlib import Path
import datetime
import os
import time
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from natsort import natsorted
#import inspect
import sys
import pickle
import shutil
import copy
import pandas as pd

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups

from engine import train_one_epoch, evaluate
import utils
import transforms as T

import mlflow
import mlflow.pytorch
    
            
def log_all_metrics_mlflow(model_metrics, args, classes):
    '''logging scalars to mlflow'''
    for key, value in model_metrics.items():
        if type(value) == dict:
            #mlflow.log_metrics(key, value)
            pass
        else:
            mlflow.log_metric(key, value)
            
def format_metrics_checkpoint(eval_object, args, classes):
    classes_restruct = {}
    for entry in classes:
        classes_restruct[entry['Id']] = entry['Name']
    metric_dict = {}
    for key, value in eval_object.coco_eval['bbox'].stats.items():
        if type(value) == dict:
            renamed = dict((classes_restruct[key], value) for (key, value) in value.items()) # replace class indices with class names for per_class_metrics
            metric_dict[key] = renamed
        else:
            metric_dict[key] = value
    return metric_dict

class CSVDetectionDataset(object):
    def __init__(self, root, path_to_csv, transforms):
        self.root = root
        self.transforms = transforms
        
        #only train on images available in root
        p = Path(root).glob('**/*')
        files = [x.name for x in p if x.is_file()]
        files = natsorted(files)
        
        df = pd.read_csv(path_to_csv)
        df = df[['wsi','bbox']]
        df = df.dropna(subset=['bbox'])
        df['bbox'] = df['bbox'].apply(json.loads)
        
        imgs = list(df['wsi'].unique())
        imgs_png = [str(x)+'.png' for x in imgs]
        imgs_tif = [str(x)+'.tif' for x in imgs]
        imgs_png = [x for x in files if x in imgs_png]
        imgs_tif = [x for x in files if x in imgs_tif]
        imgs = natsorted(imgs_png+imgs_tif)
        self.imgs = imgs
        
        self.boxes = df

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.imgs[idx])
        box_df = self.boxes
        box_df = box_df[box_df['wsi'] == int(self.imgs[idx][:-4])]
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        for box in box_df['bbox']:
            box[0], box[1], box[2], box[3] = box[1], box[0], box[3], box[2]
            boxes.append(box)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(box_df),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(box_df),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(args, trial=None):
    print(args)
    
    device = torch.device(args.device)
    
    # Data loading code
    print("Loading data")

    dataset = CSVDetectionDataset(args.imgs_path, args.data_path, get_transform(train=True))
    dataset_test = CSVDetectionDataset(args.imgs_path, args.data_path, get_transform(train=False))

    num_classes = 2 #for now, cell and background
    classes = [{'Id':1,
              'Name': 'Zelle'}]
    
    # set point to split data at
    splitpoint=int(len(dataset)*0.8)
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:splitpoint])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[splitpoint:])

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, pin_memory=True)

    print("Creating model")
    # get the model, replace head to facilitate our num_classes
    model = torchvision.models.detection.__dict__[args.model](pretrained=args.pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)    
    model.roi_heads.detections_per_img=500
    
    # move model to the right device
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return
    
    if utils.is_main_process():
        if args.output_dir:
            if Path(args.output_dir).exists():
                shutil.rmtree(args.output_dir)
            utils.mkdir(args.output_dir)
    
    print("Start training")
    start_time = time.time()
    
    stopping_metric_dict = {'mAP':'AP_at_IoU_0.50to0.95',
                            'mAP0.5':'AP_at_IoU_0.50',
                            'mAR':'AR_at_IoU_0.50to0.95',
                            'mAR0.5':'AR_at_IoU_0.50'}
    
    best_model_performance = 0.0
    best_model_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        
        # evaluate after every epoch
        evaluation = evaluate(model, data_loader_test, device=device)
        
        if utils.is_main_process():
            # check for proper order (log to mlflow first, then prune afterwards) when refactoring
            if trial is not None:
                intermediate_value = evaluation.coco_eval['bbox'].stats[stopping_metric_dict[args.stopping_metric]]
                trial.report(intermediate_value, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()
        
            #track best model
            if evaluation.coco_eval['bbox'].stats[stopping_metric_dict[args.stopping_metric]] > best_model_performance:
                best_model = copy.deepcopy(model_without_ddp)
                best_model_optimizer = copy.deepcopy(optimizer)
                best_model_lr_scheduler = copy.deepcopy(lr_scheduler)
                best_model_performance = evaluation.coco_eval['bbox'].stats[stopping_metric_dict[args.stopping_metric]]
                best_model_metrics = format_metrics_checkpoint(evaluation, args, classes)
                best_model_epoch = epoch

            # early stopping
            if args.early_stopping == True:
                if epoch-best_model_epoch > args.early_stopping_num_epochs:
                    # save model
                    if args.output_dir:
                        utils.save_on_master({
                            'model': best_model.state_dict(),
                            'optimizer': best_model_optimizer.state_dict(),
                            'lr_scheduler': best_model_lr_scheduler.state_dict(),
                            'args': args,
                            'epoch': epoch,
                            'num_classes': num_classes,
                            'metrics': format_metrics_checkpoint(evaluation, args, classes)},
                            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                        
                        # write json with metrics
                        model_json = format_metrics_checkpoint(evaluation, args, classes)
                        with open (os.path.join(args.output_dir, 'model_{}.json'.format(epoch)), 'w') as outfile:
                            json.dump(model_json, outfile)
                    break

    # save model after training finishes (and early stopping didn't terminate the run already)
    if utils.is_main_process():
        if args.output_dir:
            utils.save_on_master({
                'model': best_model.state_dict(),
                'optimizer': best_model_optimizer.state_dict(),
                'lr_scheduler': best_model_lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch,
                'num_classes': num_classes,
                'metrics': format_metrics_checkpoint(evaluation, args, classes)},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            
            # write json with metrics
            model_json = format_metrics_checkpoint(evaluation, args, classes)
            with open (os.path.join(args.output_dir, 'model_{}.json'.format(epoch)), 'w') as outfile:
                json.dump(model_json, outfile)
    
        # log metrics
        log_all_metrics_mlflow(best_model_metrics, args, classes)
        
        # Log parameters
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        # log model
        print("\nLogging the trained model as a run artifact...")
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model", pickle_module=pickle)
        print(
            "\nThe model is logged at:\n%s" % os.path.join(mlflow.get_artifact_uri(), "pytorch-model")
        )

        # log trainingdata
        print("Uploading Dataset as run artifact...")
        mlflow.log_artifact(args.data_path, artifact_path="dataset")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return best_model_performance

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--name', default='default', help='name of the training run')
    parser.add_argument('--experiment-name', help='name of the experiment')
    parser.add_argument('--data-path', default='small_data.csv', help='csv containing image names, labels, etc')
    parser.add_argument('--imgs-path', default='kaggle/wsi', help='path to folder containing the images')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--early_stopping', default=True, help='use early stopping')
    parser.add_argument('--early_stopping_num_epochs', default=10, type=int, help='how many epochs to wait for improvement')
    parser.add_argument('--stopping_metric', default='mAP', help='metric to use for early stopping calculation: mAP, mAP0.5, mAR, mAR0.5')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu '
                        '::: 0.02/8*$NGPU')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=7, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--num_checkpoints', default=10, type=int, help='number of checkpoints to keep during training. Keeps memory requirements in check')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":
    args = parse_args()
    
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.name):
        main(args)