import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
from dataset import CUFED
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from options.train_options import TrainOptions
from model import ExitingGatesGATCNN as Model_Gate
from model import ModelGCNConcAfterClassifier as Model_Cls
from model import ModelGCNConcAfterLocalFrame as Model_Basic_Local
from model import ModelGCNConcAfterGlobalFrame as Model_Basic_Global

args = TrainOptions().parse()
cls_number = len(args.t_step)

class EarlyStopper:
    def __init__(self, patience, min_delta, stopping_threshold):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float('inf')
        self.stopping_threshold = stopping_threshold

    def early_stop(self, val_loss):
        if val_loss <= self.stopping_threshold:
            return True, True
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
            return False, True
        if val_loss > (self.min_val_loss - self.min_delta):
            self.counter += 1
            if self.counter > self.patience:
                return True, False
        return False, False


def train_frame(model_cls, model_gate, model_vigat_local, model_vigat_global, dataset, loader, crit, crit_gate, opt, sched, device):
    model_gate.train()
    epoch_loss = 0
    for batch in loader:
        feats_local, feats_global, label = batch
        feats_local = feats_local.to(device)
        feats_global = feats_global.to(device)
        label = label.to(device)

        feat_global_single, wids_frame_global = model_vigat_global(feats_global, get_adj=True)

        # Cosine Similarity
        normalized_global_feats = F.normalize(feats_global, dim=2)
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

        for t in range(cls_number):
            indexes = index_bestframes[:, :args.t_step[t]].to(device)
            feats_bestframes = feats_local.gather(dim=1, index=indexes.unsqueeze(-1).unsqueeze(-1)
                                            .expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
            feat_local_single = model_vigat_local(feats_bestframes)
            feat_single_cls = torch.cat([feat_local_single, feat_global_single], dim=-1)
            feat_gate = torch.cat((feat_gate, feat_local_single.unsqueeze(dim=1)), dim=1)
            out_data = model_cls(feat_single_cls)
            loss_t = crit(out_data, label).mean(dim=-1)
            e_t = args.beta * torch.exp(torch.tensor(args.t_array[t])/2.)
            labels_gate = loss_t < e_t
            out_data_gate = model_gate(feat_gate.to(device), t)
            loss_gate += crit_gate(out_data_gate, torch.Tensor.float(labels_gate).unsqueeze(dim=1))

        loss_gate = loss_gate / cls_number
        loss_gate.backward()
        opt.step()
        epoch_loss += loss_gate.item()

    sched.step()
    return epoch_loss / len(loader)


def evaluate_frame(model_cls, model_gate, model_vigat_local, model_vigat_global, dataset, loader, crit, crit_gate, device):
    model_gate.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch in loader:
            feats_local, feats_global, label, _ = batch
            feats_local = feats_local.to(device)
            feats_global = feats_global.to(device)
            label = label.to(device)

            feat_global_single, wids_frame_global = model_vigat_global(feats_global, get_adj=True)

            # Cosine Similarity
            normalized_global_feats = F.normalize(feats_global, dim=2)
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

            for t in range(cls_number):
                indexes = index_bestframes[:, :args.t_step[t]].to(device)
                feats_bestframes = feats_local.gather(dim=1, index=indexes.unsqueeze(-1).unsqueeze(-1)
                                                .expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
                feat_local_single = model_vigat_local(feats_bestframes)
                feat_single_cls = torch.cat([feat_local_single, feat_global_single], dim=-1)
                feat_gate = torch.cat((feat_gate, feat_local_single.unsqueeze(dim=1)), dim=1)
                out_data = model_cls(feat_single_cls)
                loss_t = crit(out_data, label).mean(dim=-1)
                e_t = args.beta * torch.exp(torch.tensor(args.t_array[t])/2.)
                labels_gate = loss_t < e_t
                out_data_gate = model_gate(feat_gate.to(device), t)
                loss_gate += crit_gate(out_data_gate, torch.Tensor.float(labels_gate).unsqueeze(dim=1))

            loss_gate = loss_gate / cls_number
            epoch_loss += loss_gate.item()

        return epoch_loss / len(loader)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.dataset == 'cufed':
        train_dataset = CUFED(args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir)
        val_dataset = CUFED(args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
    else:
        sys.exit("Unknown dataset!")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("train_set = {}".format(len(train_dataset)))
        print("val_set = {}".format(len(val_dataset)))

    # Gate Model
    model_gate = Model_Gate(args.gcn_layers, train_dataset.NUM_FEATS, num_gates=cls_number)
    crit = nn.BCEWithLogitsLoss(reduction='none')
    crit_gate = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model_gate.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
    early_stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, stopping_threshold=args.stopping_threshold)
    
    data_vigat = torch.load(args.vigat_model[0], map_location=device)
    # Classifier Model
    model_cls = Model_Cls(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.NUM_CLASS)
    model_cls.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_cls.eval()
    # Vigat Model Local
    model_vigat_local = Model_Basic_Local(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.NUM_CLASS)
    model_vigat_local.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_vigat_local.eval()
    # Vigat Model Global
    model_vigat_global = Model_Basic_Global(args.gcn_layers, train_dataset.NUM_FEATS, train_dataset.NUM_CLASS)
    model_vigat_global.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_vigat_global.eval()

    start_epoch = 0
    if args.resume:
        data = torch.load(args.resume, map_location=device)
        start_epoch = data['epoch']
        model_gate.load_state_dict(data['model_state_dict'], strict=True)
        opt.load_state_dict(data['opt_state_dict'])
        sched.load_state_dict(data['sched_state_dict'])
        print("resuming from epoch {}".format(start_epoch))

    for epoch in range(start_epoch, args.num_epochs):
        epoch_cnt = epoch + 1
        model_gate = model_gate.to(device)
        model_cls = model_cls.to(device)
        model_vigat_local = model_vigat_local.to(device)
        model_vigat_global = model_vigat_global.to(device)

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

        torch.save(model_config, os.path.join(args.save_dir, 'last_gate_{}.pt'.format(args.dataset)))

        if is_save_ckpt:
            torch.save(model_config, os.path.join(args.save_dir, 'best_gate_{}.pt'.format(args.dataset)))

        if is_early_stopping:
            print('Stop at epoch {}'.format(epoch_cnt)) 
            break

        if args.verbose:
            print("[epoch {}] train_loss={} val_loss={} dt_train={:.2f}sec dt_val={:.2f}sec dt={:.2f}sec".format(epoch_cnt, train_loss, val_loss, t1 - t0, t3 - t2, t1 - t0 + t3 - t2))


if __name__ == '__main__':
    main()