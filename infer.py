import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from dataset import CUFED
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import AP_partial, showCM
from torchvision.utils import make_grid
from model import ModelGCNConcAfter as Model
from sklearn.preprocessing import MinMaxScaler
from options.infer_options import InferOptions
from model import ExitingGatesGATCNN as Model_Gate
from model import ModelGCNConcAfterClassifier as Model_Cls
from model import ModelGCNConcAfterLocalFrame as Model_Basic_Local
from model import ModelGCNConcAfterGlobalFrame as Model_Basic_Global
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
import cv2
import pickle
import math

args = InferOptions().parse()


album_imgs_path = os.path.join(args.split_dir, "album_imgs_mask.json")
with open(album_imgs_path, 'r') as f:
    album_imgs = json.load(f)
imgs_path = album_imgs[args.album_path.split('/')[-1]]
imgs_path_np = np.array([os.path.join(args.album_path, img_path) for img_path in imgs_path])

def get_album(args):
    tensor_batch = torch.zeros(len(imgs_path), args.input_size, args.input_size, 3)
    for i, img_path in enumerate(imgs_path_np):
        im = Image.open(img_path)
        im_resize = im.resize((args.input_size, args.input_size))
        np_img = np.array(im_resize, dtype=np.uint8)
        tensor_batch[i] = torch.from_numpy(np_img).float() / 255.0
    tensor_batch = tensor_batch.permute(0, 3, 1, 2)   # HWC to CHW
    montage = make_grid(tensor_batch).permute(1, 2, 0).cpu()
    return tensor_batch, montage

def display_image(montage, tags, filename, path_dest):
    if not os.path.exists(path_dest):
        os.makedirs(path_dest)
    plt.figure()
    plt.axis('off')
    plt.rcParams["axes.titlesize"] = 16
    plt.title(tags)
    plt.imshow(montage)
    plt.savefig(os.path.join(path_dest, filename))


def infer_gate(model_gate, model_cls, model_vigat_local, model_vigat_global, device):
    path_splits = args.album_path.split('/')
    album_name = path_splits[-1]
    album_type = path_splits[-2]
    album_path = album_type + '/' + album_name
    output_path = os.path.join(args.path_output, album_path)

    class_selected = 0
    local_folder = 'clip_local'
    global_folder = 'clip_global'

    with torch.no_grad():
        label_path = os.path.join(args.dataset_root, "event_type.json")
        with open(label_path, 'r') as f:
          album_data = json.load(f)
        labels_np = np.zeros((1, CUFED.NUM_CLASS), dtype=np.float32)
        for lbl in album_data[album_name]:
            idx = CUFED.event_labels.index(lbl)
            labels_np[0, idx] = 1

        local_path = os.path.join(args.feats_dir, local_folder, album_name + '.npy')
        global_path = os.path.join(args.feats_dir, global_folder, album_name + '.npy')
        feats_local = torch.from_numpy(np.expand_dims(np.load(local_path), 0)).to(device)
        feats_global = torch.from_numpy(np.expand_dims(np.load(global_path), 0)).to(device)

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

        for t in range(args.cls_number):
            indexes = index_bestframes[:, :args.t_step[t]].to(device)
            feats_bestframes = feats_local.gather(dim=1, index=indexes.unsqueeze(-1).unsqueeze(-1).
                                            expand(-1, -1, CUFED.NUM_BOXES, CUFED.NUM_FEATS)).to(device)
            feat_local_single, wids_objects, _ = model_vigat_local(feats_bestframes, get_adj=True)
            feat_single_cls = torch.cat([feat_local_single, feat_global_single], dim=-1)
            feat_gate = torch.cat((feat_gate, feat_local_single.unsqueeze(dim=1)), dim=1)

            out_data = model_cls(feat_single_cls)

            out_data_gate = model_gate(feat_gate.to(device), t)
            class_selected = t
            exit_switch = out_data_gate >= 0.5
            if exit_switch or t == (args.cls_number - 1):
                break

        n_salient_frames = args.t_step[class_selected]
        top_indexes = index_bestframes[0, :n_salient_frames]
        scores = out_data.cpu()

        selected_top_indexes = index_bestframes[0, :args.n_frames] # n_frames <= n_salient_frames
        top_imgs_path = imgs_path_np[selected_top_indexes]
        top_obj_idx = np.argsort(wids_objects[selected_top_indexes], axis=1)[-args.n_objects:]
        with open(args.detic_path, 'rb') as f:
            detic_model = pickle.load(f)
        with open(args.metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        detected_imgs = []
        for img_path in top_imgs_path:
            im = cv2.imread(img_path)
            output = detic_model(im)
            boxes = output['instances'][top_obj_idx].pred_boxes.tensor.cpu()
            classes = [metadata.thing_classes[x] for x in output["instances"][top_obj_idx].pred_classes.cpu().tolist()]
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [cord.item() for cord in box]
                x, y = math.floor(x1), math.floor(y1)
                w, h = math.floor(x2 - x1 + 1), math.floor(y2 - y1 + 1)
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(im, classes[i], (int(x + w / 2), int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (36, 255, 12), 1)
            im_resize = im.resize((args.input_size, args.input_size))
            detected_imgs.append(im_resize)
        detected_montage = make_grid(detected_imgs).cpu()
           
    # Change tensors to 1d-arrays
    m = nn.Sigmoid()
    preds = m(scores)
    preds[preds >= args.threshold] = 1
    preds[preds < args.threshold] = 0
    preds = preds.numpy()
    scores = scores.numpy()

    if args.dataset == 'cufed':
        map_micro, map_macro = AP_partial(labels_np, scores)[1:3]
        acc = accuracy_score(labels_np, preds)
        cms = multilabel_confusion_matrix(labels_np, preds)
        cr = classification_report(labels_np, preds)
        
        print('cls_frames={} map_micro={:.2f} map_macro={:.2f} accuracy={:.2f}'.format(n_salient_frames, map_micro, map_macro, acc * 100))
        print(cr)
        showCM(cms)

        album_tensor, montage = get_album(args)
        filtered_tensor = torch.index_select(album_tensor, dim=0, index=top_indexes)
        top_montage = make_grid(filtered_tensor).permute(1, 2, 0).cpu()
        display_image(montage, np.array(CUFED.event_labels)[list(map(bool,preds.squeeze(0)))], 'montage.jpg', output_path)
        display_image(top_montage, 'all {} salient frames'.format(n_salient_frames), 'salient_montage.jpg', output_path)
        display_image(detected_montage, '{} salient frames'.format(args.n_frames), 'detected_montage.jpg', output_path)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.verbose:
        print("running on {}".format(device))

    data_gate = torch.load(args.gate_model[0], map_location=device)
    data_vigat = torch.load(args.vigat_model[0], map_location=device)
    # Gate Model
    model_gate = Model_Gate(args.gcn_layers, CUFED.NUM_FEATS, num_gates=args.cls_number)
    model_gate.load_state_dict(data_gate['model_state_dict'])
    model_gate.eval()
    model_gate = model_gate.to(device)
    # Vigat Model
    model = Model(args.gcn_layers, CUFED.NUM_FEATS, CUFED.NUM_CLASS)
    model.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model.eval()
    model = model.to(device)
    # Vigat Model Local
    model_vigat_local = Model_Basic_Local(args.gcn_layers, CUFED.NUM_FEATS, CUFED.NUM_CLASS)
    model_vigat_local.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_vigat_local.eval()
    model_vigat_local = model_vigat_local.to(device)
    # Vigat Model Global
    model_vigat_global = Model_Basic_Global(args.gcn_layers, CUFED.NUM_FEATS, CUFED.NUM_CLASS)
    model_vigat_global.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_vigat_global.eval()
    model_vigat_global = model_vigat_global.to(device)
    # Classifier Model
    model_cls = Model_Cls(args.gcn_layers, CUFED.NUM_FEATS, CUFED.NUM_CLASS)
    model_cls.load_state_dict(data_vigat['model_state_dict'], strict=True)
    model_cls.eval()
    model_cls = model_cls.to(device)

    infer_gate(model_gate, model_cls, model_vigat_local, model_vigat_global, device)


if __name__ == '__main__':
    main()