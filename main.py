import os

import torch
import torch.utils.data
from opts import opts
from logger import Logger
from models.model import create_model, load_model, save_model
from datasets.layout_dataset import LayoutData
# from trains.train_factory import train_factory
import cv2
from trains.train_factory import train_factory

import numpy as np
import pandas as pd

import multiprocessing

def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test  # Search for the most suitable convolution algorithm for each convolution layer of the whole network, and then accelerate the network
    Dataset = LayoutData
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)  # The program parameters are obtained, including training parameters, model parameters, etc

    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    # 'cuda' if opt.gpus[0] >= 0 else
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')   # Choose CPU or multiple GPUs

    print('creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)  # Create the model
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)  # Creating the optimizer
    start_epoch = 0


    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    # If this is the test phase, you need to load the existing model



    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    # Get the trainer, set up the optimizer and the training equipment


    print('Setting up data...')
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    # Get the DataSet of the validation set
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # Get the dataset of the training set

    print('starting training...')
    best = 1e10

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        mark = epoch if opt.save_all else 'last'


        log_dict_train, _ = trainer.train(epoch, train_loader)  # 每个epoch的训练

        logger.write('epoch:{} | train |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        # record training log


     # Verify the set to reason and get loss
        if (opt.val_intervals > 0 and epoch % opt.val_intervals == 0):
            logger.write('\n')
            logger.write('epoch:{} | val |'.format(epoch))
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            # Print the log and save the training model

            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            # Inference model in the absence of back propagation

            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            # Log to TXT

            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
            # save model
            logger.write('\n')
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')


        if epoch in opt.lr_step:  # reduce learning rate
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            logger.write('Drop learning rate to {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
