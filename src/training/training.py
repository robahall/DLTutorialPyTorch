import argparse
import sys
import os
import datetime as datetime

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

#from src.util.util import enumerateWithEstimate
from src.datasets.datasets import LunaDataset
from src.util.logconf import logging
#from src.training.model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=32,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            defualt=1,
                            type=int,
                            )
        parser.add_argument('--tb-prefix',
                            default='rah_12',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )
        parser.add_argument('comment',
                            help="Comment suffix for TensorBoard run.",
                            nargs='?',
                            defualt='DLTest'
                            )
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices.")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr = 0.001, momentum=0.99)

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool = True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers = self.cli_args.num_workers,
            pin_memory= self.use_cuda,
        )

        return val_dl

    def initTensorboardWriter(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir = log_dir + '-trn_cls-' + self.cli_args.comment)
            self.val_writer = SummaryWriter(
                log_dir = log_dir + '-val_cls-' + self.cli_args.comment)

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args} ")

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info(f"Epoch {epoch_ndx} of {self.cli_args.epochs}, {len(train_dl)}/{len(val_dl)} \\"
                     f"batches of size {self.cli_args.batch_size}*{(torch.cuda.device_count() if self.use_cuda else 1)}")

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training.",
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computerBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

            self.totalTrainingSamples_count += len(train_dl.dataset)

            return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                f"E{epoch_ndx} Training.",
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computerBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    ## Finish first-pass NN design








if __name__ == '__main__':
    LundaTrainingApp().main()
