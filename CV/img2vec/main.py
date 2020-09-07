# %%
from __future__ import print_function, division
import copy
import time
import datetime
import random
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import warnings
import logging

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import utils.torchutil as torchutil
from utils.log import lg
from item_dataset import ItemDataset
from model import build_model
warnings.filterwarnings("ignore")
# python train.py --batch_size 512 --num_epochs 10
# %%

# %%
# Data augmentation and normalization for training
# Just normalization for validation
IMG_SIZE = 256
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(680),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lg.info("device: %s", device)
embedding_dim = 300

# %%


def train_model(items, model, SAVE_DIR, num_epochs=10, st_epoch=None, 
                stop_window=None, bs=512, test=False, flod_k=0):
    train_items, val_items = train_test_split(items, test_size=0.33, random_state=42)
    lg.info('train_items shape: %s, val_items shape: %s', train_items.shape, val_items.shape)
    train_d = ItemDataset(train_items, train_transforms)
    train_d_val = ItemDataset(train_items, val_transform)
    val_d = ItemDataset(val_items, val_transform)

    train_d, val_d = torchutil.SafeDataset(train_d), torchutil.SafeDataset(val_d)
    train_d_val = torchutil.SafeDataset(train_d_val)
    train_loader = DataLoader(train_d, batch_size=bs, shuffle=True, num_workers=4)
    train_val_loader = DataLoader(train_d_val, batch_size=bs, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_d, batch_size=bs, shuffle=False, num_workers=4)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    # optimizer = optim.SGD(filter(lambda p:
    #                       p.requires_grad, model.parameters()),
    # lr=0.001, momentum=0.9)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()))
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    epochs_no_improve = 0
    since = time.time()
    for epoch in range(num_epochs):
        lg.info('Epoch %d/%d', epoch, num_epochs - 1)
        if st_epoch is not None and epoch < st_epoch:
            lg.info("st_epoch: %d, skip %d", st_epoch, epoch)
            continue
        lg.info('-' * 20)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            acc_sum = 0.0
            top5_acc_sum = 0.0
            top3_acc_sum = 0.0

            # Iterate over data.
            # lg_tqdm = myutil.TqdmToLogger(lg.getLogger(), level=lg.INFO)
            with tqdm(total=len(dataloader)) as t:
                for _, labels, imgs in dataloader:  # labels: [[1],[3]]
                    inputs = imgs.to(device)
                    labels = labels.to(device)
                    lg.debug('labels shape: %s, %s', labels.size(), labels)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        lg.debug('outputs shape: %s, %s', outputs.size(), outputs)
                        # values, indices = torch.max()
                        # _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    bs = inputs.size(0)
                    running_loss += loss.item() * bs
                    # running_corrects += torch.sum(preds == labels.data)
                    top_acc = torchutil.accuracy(outputs, labels, topk=(1, 3, 5))
                    acc_sum += top_acc[0]
                    top3_acc_sum += top_acc[1]
                    top5_acc_sum += top_acc[2]

                    t.set_description_str(
                        "{} loss {:.4f}".format(phase, loss.item()))
                    t.update()
                    if test:
                        break
            if phase == 'train':
                scheduler.step()
                data_size = len(train_items)
            else:
                data_size = len(val_items)
            epoch_loss = running_loss / data_size
            epoch_acc = acc_sum.item() / data_size
            epoch_top3_acc = top3_acc_sum.item() / data_size
            epoch_top5_acc = top5_acc_sum.item() / data_size

            lg.info('fold[{}] epoch[{}] {} Loss: {:.4f},Acc: {:.2f}%|Top3 Acc:{:.2f}%| Top5 Acc:{:.2f}%'.format(flod_k, 
                epoch, phase, epoch_loss, epoch_acc*100, 
                epoch_top3_acc*100, epoch_top5_acc*100))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    epochs_no_improve = 0
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts,
                               '{}/model_fold_{}_epoch_{}'.
                               format(SAVE_DIR, flod_k, epoch))
                else:
                    epochs_no_improve += 1
        if stop_window is not None and epochs_no_improve >= stop_window:
            lg.info("early stopping, epochs_no_improve: %d", epochs_no_improve)
            break
        if test:
            break

    time_elapsed = time.time() - since
    time_elapsed = datetime.timedelta(seconds=time_elapsed)
    lg.info('fold[%d] Training complete in %s', flod_k, time_elapsed)
    lg.info("best_epoch: %d, best_acc: %f ", best_epoch, best_acc)

    # load best model weights
    model.load_state_dict(best_model_wts)

    em_train = get_embedding(model, train_val_loader, test)
    em_val = get_embedding(model, val_loader, test)

    em = np.vstack((em_train, em_val))
    np.save('{}/em.npy'.format(SAVE_DIR), em)

    return model


def get_embedding(model, dataloader, test=False):
    my_embedding = None

    def hook(m, i, o):
        my_embedding.copy_(o.data)
    h = model.base.fc.register_forward_hook(hook)

    model.to(device)
    model.eval()
    item_ems = None
    # lg_tqdm = myutil.TqdmToLogger(lg.getLogger(), level=lg.INFO)
    with tqdm(total=len(dataloader)) as t:
        for item_id, labels, imgs in dataloader:
            inputs = imgs.to(device)
            my_embedding = torch.zeros(inputs.size()[0], embedding_dim)
            _ = model(inputs)
            em = my_embedding.numpy()
            item_ids = np.expand_dims(item_id.numpy(), 1)
            labels = np.expand_dims(labels.numpy(), 1)
            lg.debug('item_ids shape:{}, em shape: {}'.format(
                item_ids.shape, em.shape))
            rows = np.hstack((labels, item_ids, em))
            lg.debug('rows shape: %s', rows.shape)
            if item_ems is None:
                item_ems = rows
            else:
                item_ems = np.vstack((item_ems, rows))
            t.set_description('embedding with label and itemId shape: {}, batch min: {:.3f}'.format(
                item_ems.shape, np.min(em)))
            t.update()
            if test:
                break
    h.remove()
    return item_ems


# %%
def run(item_fp, bs=512, save_dir=None, fold=None, model_path=None,
        num_epochs=10, st_epoch=0, stop_window=3, test=False, gene_em=False):
    items = pd.read_csv(item_fp)
    num_class = items['label'].nunique()
    lg.info('item shape: %s, num_class:%s', items.shape, num_class)

    if save_dir is None:
        save_dir = '{}_{}'.format('result', 
                                datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    model_ft = build_model(num_class, embedding_dim)

    if model_path is not None:
        model_ft.load_state_dict(torch.load('{}'.format(model_path)))

    train_model(items, model_ft, save_dir,
                num_epochs=num_epochs, st_epoch=st_epoch,
                stop_window=stop_window, bs=bs, test=test)
    # reset for next fold
    model_path = None
    st_epoch = 0
    if test:
        lg.info('just test, return')
#        break

# %%
def build_em(items, model_ft, flod_k, train_idx, val_index, save_dir, bs):
    val_items = items.iloc[val_index, :]
    val_d = ItemDataset(val_items, val_transform)
    val_d = torchutil.SafeDataset(val_d)
    val_loader = DataLoader(val_d, batch_size=bs,
                            shuffle=False, num_workers=4)

    em = get_embedding(model_ft, val_loader)
    em_df = pd.DataFrame(em)
    em_df.to_csv('{}/embedding_{}.csv'.format(save_dir, flod_k),
                 index=False, encoding='utf-8', header=False)

    np.savez('{}/index_fold_{}'.format(save_dir, flod_k),
             train_idx=train_idx, val_idx=val_index)

# %%
# --bs 800 --refit *** --fold k --em
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug', action='store_true',
                        help='debug level')
    parser.add_argument('--test', action='store_true',
                        help='test')
    parser.add_argument('--em', action='store_true',
                        help='is only build embedding')
    parser.add_argument('--items', type=str, default='items_with_label.csv',
                        help='items meta csv file')
    parser.add_argument('--save_dir', type=str, default='result',
                        help='save dir')
    parser.add_argument('--input_size', type=int, default=256,
                        help='size of input image')
    parser.add_argument('--em_size', type=int, default=300,
                        help='size of image embedding')
    parser.add_argument('--bs', type=int, default=64,
                        help='batch size')  # 800 for 16GB gpu
    parser.add_argument('--epochs', type=int, default=10,
                        help='num_epochs')
    parser.add_argument('--es', type=int, default=1,
                        help='early stopping step')
    parser.add_argument('--model', type=str, default='152',
                        help='resnet model type')
    parser.add_argument('--refit', type=str, default=None,
                        help='base model')
    parser.add_argument('--fold', type=int, default=None,
                        help='fold')
    parser.add_argument('--epoch', type=int, default=None,
                        help='start epoch')
    parser.add_argument('--gpu', type=str, default='0',
                        help='use which gpu')
    args = parser.parse_args()

    if args.debug:
        lg.setLevel(logging.DEBUG)

    run(args.items, bs=args.bs, fold=args.fold, save_dir=args.save_dir, num_epochs=args.epochs,
        test=args.test, model_path=args.refit, gene_em=args.em, stop_window=args.es)
