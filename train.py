import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

default_train_dir = "data/train"
default_test_dir = "data/test"

dir_img_train = f'{default_train_dir}/imgs/'
dir_mask_train = f'{default_train_dir}/masks/'
dir_img_test = f'{default_test_dir}/imgs/'
dir_mask_test = f'{default_test_dir}/masks/'
dir_checkpoint = 'checkpoints/'

def validation_only(net,
                    device,
                    batch_size=1,
                    img_width=0, 
                    img_height=0,
                    img_scale=1.0,
                    use_bw=False,
                    standardize=False,
                    compute_statistics=False):

    load_statstics = not compute_statistics
    dataset = BasicDataset(dir_img_test, dir_mask_test, img_width, img_height, img_scale, use_bw,
                           standardize=standardize, load_statistics=load_statstics, save_statistics=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    val_score = eval_net(net, val_loader, device)
    if net.n_classes > 1:
        logging.info('Validation cross entropy: {}'.format(val_score))
    else:
        logging.info('Validation Dice Coeff: {}'.format(val_score))

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_width=0, 
              img_height=0,
              img_scale=1.0,
              use_bw=False,
              standardize=False,
              compute_statistics=False):

    load_statstics = not compute_statistics
    dataset = BasicDataset(dir_img_train, dir_mask_train, img_width, img_height, img_scale, use_bw,
                           standardize=standardize, load_statistics=load_statstics, save_statistics=True)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images resizing: {img_width}x{img_height}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor of the images. Takes priority over resize')
    parser.add_argument('-r', '--resize', dest='resize_string', type=str,
                        help='Size images should be resized to, in format: NxM. Example: 24x24')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--test', dest='test', action='store_true',
                        help='Perform validation only and display score')
    parser.add_argument('--bw', dest='use_bw', action='store_true',
                        help='Use black-white images')
    parser.add_argument('--standardize', dest='standardize', action='store_true',
                        help='Standardize images based on dataset mean and std values')
    parser.add_argument('--compute_statistics', dest='compute_statistics', action='store_true',
                        help='Calculate dataset statistics even if there\'s json file present')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true',
                        help='Use cpu even if gpu is available')
    parser.add_argument('--train_data', dest='train_dir', type=str, default=default_train_dir,
                        help='Path to training data dir')
    parser.add_argument('--test_data', dest='test_dir', type=str, default=default_test_dir,
                        help='Path to training data dir')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    logging.info(f'Using device {device}')

    # Set train and test data directories
    if args.train_dir:
        dir_img_train = f'{args.train_dir}/imgs/'
        dir_mask_train = f'{args.train_dir}/masks/'
    if args.test_dir:
        dir_img_test = f'{args.test_dir}/imgs/'
        dir_mask_test = f'{args.test_dir}/masks/'

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_channels=1 for B-W images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    if args.use_bw:
        n_channels = 1
    else:
        n_channels = 3
    net = UNet(n_channels=n_channels, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    if args.resize_string:
        resize = list(map(int, args.resize_string.split("x")))
        img_width = resize[0]
        img_height = resize[1]
    else:
        img_width = 0
        img_height = 0

    try:
        if args.test:
            validation_only(net=net,
                            device=device,
                            batch_size=args.batchsize,
                            img_width=img_width,
                            img_height=img_height,
                            img_scale=args.scale,
                            use_bw=args.use_bw,
                            standardize=args.standardize,
                            compute_statistics=args.compute_statistics) 
        else:
            train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      device=device,
                      img_width=img_width,
                      img_height=img_height,
                      img_scale=args.scale,
                      use_bw=args.use_bw,
                      standardize=args.standardize,
                      compute_statistics=args.compute_statistics,
                      val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
