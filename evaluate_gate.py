import sys
import torch
import numpy as np
import torch.nn as nn
from dataset import CUFED, PEC
import torch.nn.functional as F
from utils import AP_partial, showCM
from torch.utils.data import DataLoader
from options.test_options import TestOptions
from model import ModelGCNConcAfter as Model
from sklearn.preprocessing import MinMaxScaler
from model import ExitingGatesGATCNN as Model_Gate
from model import ModelGCNConcAfterClassifier as Model_Cls
from model import ModelGCNConcAfterLocalFrame as Model_Basic_Local
from model import ModelGCNConcAfterGlobalFrame as Model_Basic_Global
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report

args = TestOptions().parse()
cls_number = len(args.t_step)


def evaluate_gate(model_gate, model_cls, model_vigat_local, model_vigat_global, dataset, loader, scores, class_of_video, class_vids, device):
    gidx = 0
    class_selected = 0
    album_frames = {}
    for i in range(cls_number):
        album_frames[i] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats_local, feats_global, _ = batch
            feats_local = feats_local.to(device)
            feats_global = feats_global.to(device)
            feat_global_single, wids_frame_global = model_vigat_global(feats_global, get_adj=True)

            # Cosine Similarity
            with torch.no_grad():
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

            for t in range(cls_number):
                indexes = index_bestframes[:, :args.t_step[t]].to(device)
                feats_bestframes = feats_local.gather(dim=1, index=indexes.unsqueeze(-1).unsqueeze(-1).
                                                expand(-1, -1, dataset.NUM_BOXES, dataset.NUM_FEATS)).to(device)
                feat_local_single = model_vigat_local(feats_bestframes)
                feat_single_cls = torch.cat([feat_local_single, feat_global_single], dim=-1)
                feat_gate = torch.cat((feat_gate, feat_local_single.unsqueeze(dim=1)), dim=1)

                out_data = model_cls(feat_single_cls)

                out_data_gate = model_gate(feat_gate.to(device), t)
                class_selected = t
                exit_switch = out_data_gate >= 0.5
                if exit_switch or t == (cls_number - 1):
                    class_vids[t] += 1
                    album_frames[t].append(dataset.albums[i])
                    break
            shape = out_data.shape[0]
            class_of_video[gidx:gidx+shape] = class_selected
            scores[gidx:gidx+shape, :] = out_data.cpu()
            gidx += shape

    # with open('/kaggle/working/album_frames.json', 'w') as f:
    #     json.dump(album_frames, f)

    if isinstance(dataset, CUFED):
        # Change tensors to 1d-arrays
        m = nn.Sigmoid()
        preds = m(scores)
        preds[preds >= args.threshold] = 1
        preds[preds < args.threshold] = 0
        preds = preds.numpy()
        scores = scores.numpy()
        # Ensure no row has all zeros
        for i in range(preds.shape[0]):
            if np.sum(preds[i]) == 0:
                preds[i][np.argmax(scores[i])] = 1
    else:
        scores = scores.numpy()
        preds = np.zeros(scores.shape, dtype=np.float32)

        # Find the index of the maximum value along each row
        max_indices = np.argmax(scores, axis=1)

        # Set the corresponding elements in 'preds' to 1
        preds[np.arange(preds.shape[0]), max_indices] = 1

    class_of_video = class_of_video.numpy()
    class_vids = class_vids.numpy()
    num_total_vids = int(np.sum(class_vids))
    assert num_total_vids == len(dataset)
    class_vids_rate = class_vids / num_total_vids
    avg_frames = int(np.sum(class_vids_rate*args.t_step))

    map_micro, map_macro = AP_partial(dataset.labels, scores)[1:3]
    acc = accuracy_score(dataset.labels, preds)
    class_ap = np.zeros(cls_number)   
    cms = multilabel_confusion_matrix(dataset.labels, preds)
    cr = classification_report(dataset.labels, preds)

    print('albums with only 1 frame:', np.array(dataset.albums)[class_of_video == 0])
        
    for t in range(cls_number):
        if sum(class_of_video == t) == 0:
            print('No Videos fetched by classifier {}'.format(t))
            continue
        current_labels = dataset.labels[class_of_video == t, :]
        current_scores = scores[class_of_video == t, :]
        columns_to_delete = []
        for check in range(current_labels.shape[1]):
            if sum(current_labels[:, check]) == 0:
                columns_to_delete.append(check)
        current_labels = np.delete(current_labels, columns_to_delete, 1)
        current_scores = np.delete(current_scores, columns_to_delete, 1)
        class_ap[t] = AP_partial(current_labels, current_scores)[2]

        
        # current_preds = preds[class_of_video == t, :]
        # current_preds = np.delete(current_preds, columns_to_delete, 1)
        # sv = np.where(class_of_video == t)[0]
        # idx = np.where(current_labels != current_preds)[0]
        # print('inaccurate albums of cls {}: {}'.format(t, np.array(dataset.albums)[sv[idx]]))

#             class_ap[t] = AP_partial(dataset.labels[class_of_video == t, :],
#             scores[class_of_video == t, :], average='samples')

#     for t in range(cls_number):
#         print('classifier_{}: map={:.2f} cls_frames={}'.format(t, class_ap[t], args.t_step[t]))
#     print('map_micro={:.2f} map_macro={:.2f} accuracy={:.2f}'.format(map_micro, map_macro, acc * 100))
#     print('Total Exits per Classifier: {}'.format(class_vids))
#     print('Average Frames taken: {}'.format(avg_frames))
#     print(cr)
#     showCM(cms)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'cufed':
        dataset = CUFED(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
    elif args.dataset == 'pec':
        dataset = PEC(root_dir=args.dataset_root, feats_dir=args.feats_dir, split_dir=args.split_dir, is_train=False)
    else:
        sys.exit("Unknown dataset!")
        
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.verbose:
        print("running on {}".format(device))
        print("test_set = {}".format(len(dataset)))

    data_gate = torch.load(args.gate_model[0], map_location=device)
    data_vigat = torch.load(args.vigat_model[0], map_location=device)
    # Gate Model
    model_gate = Model_Gate(args.gcn_layers, dataset.NUM_FEATS, num_gates=cls_number)
    model_gate.load_state_dict(data_gate['model_state_dict'], strict=True)
    model_gate.eval()
    model_gate = model_gate.to(device)
    # Vigat Model
    model = Model(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS)
    model.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model.eval()
    model = model.to(device)
    # Vigat Model Local
    model_vigat_local = Model_Basic_Local(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS)
    model_vigat_local.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_vigat_local.eval()
    model_vigat_local = model_vigat_local.to(device)
    # Vigat Model Global
    model_vigat_global = Model_Basic_Global(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS)
    model_vigat_global.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_vigat_global.eval()
    model_vigat_global = model_vigat_global.to(device)
    # Classifier Model
    model_cls = Model_Cls(args.gcn_layers, dataset.NUM_FEATS, dataset.NUM_CLASS)
    model_cls.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_cls.eval()
    model_cls = model_cls.to(device)

    num_test = len(dataset)
    scores = torch.zeros((num_test, dataset.NUM_CLASS), dtype=torch.float32)
    class_of_video = torch.zeros(num_test, dtype=torch.int)
    class_vids = torch.zeros(cls_number)

    evaluate_gate(model_gate, model_cls, model_vigat_local, model_vigat_global, dataset, loader, scores,
             class_of_video, class_vids, device)


if __name__ == '__main__':
    main()