import argparse
import time
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import numpy as np
from datasets import CUFED
from model import ModelGCNConcAfterLocalFrame as Model_Basic_Local
from model import ModelGCNConcAfterGlobalFrame as Model_Basic_Global
from model import ModelGCNConcAfterClassifier as Model_Cls
from model import ExitingGatesGATCNN as Model_Gate

parser = argparse.ArgumentParser(description='GCN Album Classification')
parser.add_argument('--seed', type=int, default=2024, help='seed for randomness')
parser.add_argument('vigat_model', nargs=1, help='Frame trained model')
parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
parser.add_argument('--dataset', default='cufed', choices=['cufed', 'pec'])
parser.add_argument('--dataset_root', default='/kaggle/input/thesis-cufed/CUFED', help='dataset root directory')
parser.add_argument('--feats_dir', default='/kaggle/input/mask-cufed-feats', help='global and local features directory')
parser.add_argument('--split_dir', default='/kaggle/input/cufed-full-split', help='train split and val split')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--milestones', nargs="+", type=int, default=[16, 35], help='milestones of learning decay')
parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
parser.add_argument('--resume', default=None, help='checkpoint to resume training')
parser.add_argument('--save_dir', default='/kaggle/working/weights', help='directory to save checkpoints')
parser.add_argument('--cls_number', type=int, default=5, help='number of classifiers')
parser.add_argument('--t_step', nargs="+", type=int, default=[3, 5, 7, 9, 13], help='Classifier frames')
parser.add_argument('--t_array', nargs="+", type=int, default=[1, 2, 3, 4, 5], help='e_t calculation')
parser.add_argument('--beta', type=float, default=1e-6, help='Multiplier of gating loss schedule')
parser.add_argument('--patience', type=int, default=20, help='patience of early stopping')
parser.add_argument('--min_delta', type=float, default=1e-3, help='min delta of early stopping')
parser.add_argument('--stopping_threshold', type=float, default=0.01, help='stopping threshold of val loss for early stopping')
parser.add_argument('-v', '--verbose', action='store_true', help='show details')
args = parser.parse_args()

class EarlyStopper:
    def __init__(self, patience, min_delta, stopping_threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float('inf')
        self.stopping_threshold = stopping_threshold

    def early_stop(self, validation_mAP):
        if validation_mAP <= self.stopping_threshold:
            return True, True
        if validation_mAP < self.min_val_loss:
            self.min_val_loss = validation_mAP
            self.counter = 0
            return False, True
        if validation_mAP > (self.min_val_loss - self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True, False
        return False, False

def train_frame(model_cls, model_gate, model_vigat_local, model_vigat_global, dataset, loader, crit, crit_gate, opt, sched, device):
    model_gate.train()
    epoch_loss = 0
    for feats, feat_global, label in loader:
        feats = feats.to(device)
        feat_global = feat_global.to(device)
        label = label.to(device)

        feat_global_single, wids_frame_global = model_vigat_global(feat_global, get_adj=True)

        # Cosine Similarity
        normalized_global_feats = F.normalize(feat_global, dim=2)
        squared_euclidian_dist = torch.square(torch.cdist(normalized_global_feats, normalized_global_feats))
        cosine_disimilarity = (squared_euclidian_dist / 4.0)
        index_bestframes = np.argsort(wids_frame_global, axis=1)[:, -1:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(wids_frame_global.T)
        new_wids = scaler.transform(wids_frame_global.T).T
        for j in range(args.t_step[-1]):
            index_bestwid = np.argsort(new_wids, axis=1)[:, -1:]
            if j != 0:
                index_bestframes = np.append(index_bestframes, index_bestwid, axis=1)
            index_bestwid = torch.tensor(index_bestwid).to(device)
            specific_cosine = cosine_disimilarity[
                torch.arange(cosine_disimilarity.shape[0]).unsqueeze(-1), index_bestwid].squeeze(1)
            new_wids = (torch.tensor(new_wids).to(device) * specific_cosine).cpu().numpy()
            scaler.fit(new_wids.T)
            new_wids = scaler.transform(new_wids.T).T
        index_bestframes = torch.tensor(index_bestframes)

        opt.zero_grad()
        feat_gate = feat_global_single
        feat_gate = feat_gate.unsqueeze(dim=1)
        loss_gate = 0.

        for t in range(args.cls_number):
            indexes = index_bestframes[:, :args.t_step[t]].to(device)
            feats_bestframes = feats.gather(dim=1, index=indexes.unsqueeze(-1).unsqueeze(-1)
                                            .expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
            feat_local_single = model_vigat_local(feats_bestframes)
            feat_single_cls = torch.cat([feat_local_single, feat_global_single], dim=-1)
            feat_gate = torch.cat((feat_gate, feat_local_single.unsqueeze(dim=1)), dim=1)
            out_data = model_cls(feat_single_cls)
            if args.dataset != 'cufed':
                loss_t = crit(out_data, label)
            else:
                loss_t = crit(out_data, label).mean(dim=-1)
            e_t = args.beta * torch.exp(torch.tensor(args.t_array[t])/2.)
            labels_gate = loss_t < e_t
            out_data_gate = model_gate(feat_gate.to(device), t)
            loss_gate += crit_gate(out_data_gate, torch.Tensor.float(labels_gate).unsqueeze(dim=1))

        loss_gate = loss_gate / args.cls_number
        loss_gate.backward()
        opt.step()
        epoch_loss += loss_gate.item()

    sched.step()
    return epoch_loss / len(loader)

def evaluate_frame(model_cls, model_gate, model_vigat_local, model_vigat_global, dataset, loader, crit, crit_gate, device):
    model_gate.eval()
    with torch.no_grad():
        epoch_loss = 0
        for feats, feat_global, label in loader:
            feats = feats.to(device)
            feat_global = feat_global.to(device)
            label = label.to(device)

            feat_global_single, wids_frame_global = model_vigat_global(feat_global, get_adj=True)

            # Cosine Similarity
            normalized_global_feats = F.normalize(feat_global, dim=2)
            squared_euclidian_dist = torch.square(torch.cdist(normalized_global_feats, normalized_global_feats))
            cosine_disimilarity = (squared_euclidian_dist / 4.0)
            index_bestframes = np.argsort(wids_frame_global, axis=1)[:, -1:]
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(wids_frame_global.T)
            new_wids = scaler.transform(wids_frame_global.T).T
            for j in range(args.t_step[-1]):
                index_bestwid = np.argsort(new_wids, axis=1)[:, -1:]
                if j != 0:
                    index_bestframes = np.append(index_bestframes, index_bestwid, axis=1)
                index_bestwid = torch.tensor(index_bestwid).to(device)
                specific_cosine = cosine_disimilarity[
                    torch.arange(cosine_disimilarity.shape[0]).unsqueeze(-1), index_bestwid].squeeze(1)
                new_wids = (torch.tensor(new_wids).to(device) * specific_cosine).cpu().numpy()
                scaler.fit(new_wids.T)
                new_wids = scaler.transform(new_wids.T).T
            index_bestframes = torch.tensor(index_bestframes)

            feat_gate = feat_global_single
            feat_gate = feat_gate.unsqueeze(dim=1)
            loss_gate = 0.

            for t in range(args.cls_number):
                indexes = index_bestframes[:, :args.t_step[t]].to(device)
                feats_bestframes = feats.gather(dim=1, index=indexes.unsqueeze(-1).unsqueeze(-1)
                                                .expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
                feat_local_single = model_vigat_local(feats_bestframes)
                feat_single_cls = torch.cat([feat_local_single, feat_global_single], dim=-1)
                feat_gate = torch.cat((feat_gate, feat_local_single.unsqueeze(dim=1)), dim=1)
                out_data = model_cls(feat_single_cls)
                if args.dataset != 'cufed':
                    loss_t = crit(out_data, label)
                else:
                    loss_t = crit(out_data, label).mean(dim=-1)
                e_t = args.beta * torch.exp(torch.tensor(args.t_array[t])/2.)
                labels_gate = loss_t < e_t
                out_data_gate = model_gate(feat_gate.to(device), t)
                loss_gate += crit_gate(out_data_gate, torch.Tensor.float(labels_gate).unsqueeze(dim=1))

            loss_gate = loss_gate / args.cls_number
            epoch_loss += loss_gate.item()

        return epoch_loss / len(loader)

def main():
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.dataset == 'cufed':
        train_dataset = CUFED(args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir)
        val_dataset = CUFED(args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_val=True)
        crit = nn.BCEWithLogitsLoss(reduction='none')
        crit_gate = nn.BCEWithLogitsLoss()
    else:
        sys.exit("Unknown dataset!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("train_set={}".format(len(train_dataset)))
        print("val_set={}".format(len(val_dataset)))

    start_epoch = 0
    # Gate Model
    model_gate = Model_Gate(args.gcn_layers, train_dataset.NUM_FEATS, num_gates=args.cls_number).to(device)
    opt = optim.Adam(model_gate.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, stopping_threshold=args.stopping_threshold)
    data_vigat = torch.load(args.vigat_model[0])
    # Classifier Model
    model_cls = Model_Cls(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.NUM_CLASS).to(device)
    model_cls.load_state_dict(data_vigat['model_state_dict'])
    model_cls.eval()
    # Vigat Model Local
    model_vigat_local = Model_Basic_Local(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.NUM_CLASS).to(device)
    model_vigat_local.load_state_dict(data_vigat['model_state_dict'])
    model_vigat_local.eval()
    # Vigat Model Global
    model_vigat_global = Model_Basic_Global(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.NUM_CLASS).to(device)
    model_vigat_global.load_state_dict(data_vigat['model_state_dict'])
    model_vigat_global.eval()

    for epoch in range(start_epoch, args.num_epochs):
        epoch_cnt = epoch + 1

        t0 = time.perf_counter()
        train_loss = train_frame(model_cls, model_gate, model_vigat_local, model_vigat_global, train_dataset, train_loader, crit,
                           crit_gate, opt, sched, device)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        val_loss = evaluate_frame(model_cls, model_gate, model_vigat_local, model_vigat_global, val_dataset, val_loader, crit, crit_gate, device)
        t3 = time.perf_counter()

        is_early_stopping, is_save_ckpt = early_stopper.early_stop(val_loss)

        model_config = {
            'epoch': epoch_cnt,
            'loss': train_loss,
            'model_state_dict': model_gate.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'sched_state_dict': sched.state_dict()
        }

        torch.save(model_config, os.path.join(args.save_dir, 'last-gate-{}.pt'.format(args.dataset)))

        if is_save_ckpt:
            torch.save(model_config, os.path.join(args.save_dir, 'best-gate-{}.pt'.format(args.dataset)))

        if is_early_stopping:
            print('Stop at epoch {}'.format(epoch_cnt)) 
            break

        if args.verbose:
            print("[epoch {}] train_loss={} val_loss={} dt_train={:.2f}sec dt_val={:.2f}sec dt={:.2f}sec".format(epoch_cnt, train_loss, val_loss, t1 - t0, t3 - t2, t1 - t0 + t3 - t2))


if __name__ == '__main__':
    main()