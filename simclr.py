import logging
import os
import sys

import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils import accuracy, save_checkpoint, save_config_file

torch.manual_seed(0)

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.start_epoch = 1
        self.best_valid_loss = np.inf

    def info_nce_loss(self, features):
        batch_targets = torch.arange(self.args.batch_size, dtype=torch.long).to(self.args.device)
        batch_targets = torch.cat(self.args.n_views * [batch_targets])

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)

        mask = torch.eye(len(batch_targets)).to(self.args.device)
        similarities = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)
        similarities = similarities / self.args.temperature
        return similarities, batch_targets

    def save_json(self, result, record_name):
        result_path = self.args.record_dir

        if not os.path.exists(result_path):
            os.makedirs(result_path)
        with open(os.path.join(result_path, '{}.json'.format(record_name)), 'w') as f:
            json.dump(result, f)
        print("logs saved in", result_path)

    def resume_checkpoint(self, resume_path):
        try:
            resume_path = str(resume_path)
            print("Loading checkpoint: {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_valid_loss = checkpoint['best_valid_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        except Exception as err:
            print("[Load Checkpoint Failed {}!]\n".format(err))

    def train(self, train_loader, valid_loader):
        print("train, valid", len(train_loader), len(valid_loader))
        complete_reslts = {}

        if apex_support and self.args.fp16_precision:
            logging.debug("Using apex for fp16 precision training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                        opt_level='O2',
                                                        keep_batchnorm_fp32=True)
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        if self.args.resume is not None:
            self.resume_checkpoint(self.args.resume)

        for epoch_counter in range(self.start_epoch, self.args.epochs+1):
            train_loss = 0
            epoch_reslts = {}
            epoch_reslts['epoch'] = epoch_counter

            for images, _ in tqdm(train_loader):
                # print("\nbefor cat:", len(images), images[0].shape, images[1].shape)
                images = torch.cat(images, dim=0)
                # print("after cat:", images.shape)

                images = images.to(self.args.device)

                features = self.model(images)
                # print("features", features.shape)

                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                if apex_support and self.args.fp16_precision:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()

                # if n_iter % self.args.log_every_n_steps == 0:
                #     top1, top5 = accuracy(logits, labels, topk=(1, 5))
                #     self.writer.add_scalar('loss', loss, global_step=n_iter)
                #     self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                #     self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                #     self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            valid_loss = self._validate(self.model, valid_loader)
            if valid_loss < self.best_valid_loss:
                # save the best model weights
                self.best_valid_loss = valid_loss
                # save model checkpoints
                checkpoint_name = 'best_checkpoint.pth.tar'
                save_checkpoint({
                    'epoch': epoch_counter,
                    'best_valid_loss': self.best_valid_loss,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=True, filename=os.path.join(self.args.result_dir, checkpoint_name))

                print("save best model checkpoint in", os.path.join(self.args.result_dir, checkpoint_name))

            epoch_reslts['contrastive_train_loss'] = train_loss / len(train_loader)
            epoch_reslts['contrastive_valid_loss'] = valid_loss
            epoch_reslts['learning_rate'] = self.scheduler.get_lr()[0]
            complete_reslts[epoch_counter] = epoch_reslts
            print(epoch_reslts)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            # logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")


        self.save_json(complete_reslts, 'training_logs')
        print("Training has finished")
        logging.info("Training has finished.")

        # save model checkpoints
        checkpoint_name = 'currrent_checkpoint.pth.tar'
        save_checkpoint({
            'epoch': self.args.epochs,
            'best_valid_loss': self.best_valid_loss,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.args.result_dir, checkpoint_name))

        # logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        print(f"Model checkpoint and metadata has been saved at {os.path.join(self.args.result_dir, checkpoint_name)}.")

    def _validate(self, model, valid_loader):
        print("validation")
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, _ in tqdm(valid_loader):
                # print("\nbefor cat:", len(images), images[0].shape, images[1].shape)
                images = torch.cat(images, dim=0)
                # print("after cat:", images.shape)

                images = images.to(self.args.device)

                features = self.model(images)
                # print("features", features.shape)

                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)
                valid_loss += loss.item()
            valid_loss = valid_loss/len(valid_loader)
        return valid_loss