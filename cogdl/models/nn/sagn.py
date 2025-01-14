import copy
import os
import os.path as osp
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BaseModel, register_model
from .mlp import MLP
from cogdl.utils import spmm
from cogdl.trainers import BaseTrainer, register_trainer


def average_neighbor_features(graph, feats, nhop, norm="sym", style="all"):
    results = []
    if norm == "sym":
        graph.sym_norm()
    elif norm == "row":
        graph.row_norm()
    else:
        raise NotImplementedError

    x = feats
    results.append(x)
    for i in range(nhop):
        x = spmm(graph, x)
        if style == "all":
            results.append(x)
    if style != "all":
        results = x
    return results


def entropy(probs):
    eps = 1e-9
    res = -probs * torch.log(probs + eps) - (1 - probs) * torch.log(1 - probs + eps)
    return res


def prepare_feats(dataset, nhop, norm="sym"):
    print("Preprocessing features...")
    dataset_name = dataset.__class__.__name__
    data = dataset.data
    if not osp.exists("./sagn"):
        os.makedirs("sagn", exist_ok=True)

    feat_emb_path = f"./sagn/feats_{dataset_name}_hop_{nhop}.pt"
    is_inductive = data.is_inductive()
    train_nid = data.train_nid

    if is_inductive:
        if osp.exists(feat_emb_path):
            print("Loading existing features")
            feats = torch.load(feat_emb_path)
        else:
            data.train()
            feats_train = average_neighbor_features(data, data.x, nhop, norm=norm, style="all")
            data.eval()
            feats = average_neighbor_features(data, data.x, nhop, norm=norm, style="all")

            # train_nid, val_nid, test_nid = data.train_nid, data.val_nid, data.test_nid
            for i in range(len(feats)):
                feats[i][train_nid] = feats_train[i][train_nid]
                # feats_train, feats_val, feats_test = feats[train_nid], feats[val_nid], feats[test_nid]
                # feats[i] = torch.cat([feats_train, feats_val, feats_test], dim=0)
            torch.save(feats, feat_emb_path)
    else:
        if osp.exists(feat_emb_path):
            feats = torch.load(feat_emb_path)
        else:
            feats = average_neighbor_features(data, data.x, nhop, norm=norm, style="all")
    print("Preprocessing features done...")
    return feats


def prepare_labels(dataset, stage, nhop, threshold, probs=None, norm="row", load_emb=False):
    dataset_name = dataset.__class__.__name__
    data = dataset.data
    is_inductive = data.is_inductive()
    multi_label = len(data.y.shape) > 1

    device = data.x.device
    num_classes = data.num_classes
    train_nid = data.train_nid
    val_nid = data.val_nid
    test_nid = data.test_nid

    if not osp.exists("./sagn"):
        os.makedirs("sagn", exist_ok=True)
    label_emb_path = f"./sagn/label_emb_{dataset_name}.pt"
    # teacher_prob_path = f"./sagn/teacher_prob_{dataset_name}.pt"
    teacher_probs = probs

    if stage > 0 and probs is not None:
        # teacher_probs = torch.load(teacher_prob_path)
        node_idx = torch.cat([train_nid, val_nid, test_nid], dim=0)
        if multi_label:
            threshold = -threshold * np.log(threshold) - (1 - threshold) * np.log(1 - threshold)
            entropy_distribution = entropy(teacher_probs)
            confident_nid = torch.arange(len(teacher_probs))[(entropy_distribution.mean(1) <= threshold)]
        else:
            confident_nid = torch.arange(len(teacher_probs))[teacher_probs.max(1)[0] > threshold]
        extra_confident_nid = confident_nid[confident_nid >= len(train_nid)]
        confident_nid = node_idx[confident_nid]
        extra_confident_nid = node_idx[extra_confident_nid]

        if multi_label:
            pseudo_labels = teacher_probs
            pseudo_labels[pseudo_labels >= 0.5] = 1
            pseudo_labels[pseudo_labels < 0.5] = 0
            labels_with_pseudos = torch.ones_like(data.y)
        else:
            pseudo_labels = torch.argmax(teacher_probs, dim=1)
            labels_with_pseudos = torch.zeros_like(data.y)
        train_nid_with_pseudos = np.union1d(train_nid.cpu().numpy(), confident_nid.cpu().numpy())
        train_nid_with_pseudos = torch.from_numpy(train_nid_with_pseudos).to(device)
        labels_with_pseudos[train_nid] = data.y[train_nid]
        labels_with_pseudos[extra_confident_nid] = pseudo_labels[extra_confident_nid]
    else:
        # confident_nid = train_nid
        train_nid_with_pseudos = train_nid
        labels_with_pseudos = data.y.clone()
        # teacher_probs = None
        # pseudo_labels = None

    if (not is_inductive) or stage > 0:
        if multi_label:
            label_emb = 0.5 * torch.ones((data.num_nodes, num_classes), device=device)
            label_emb[train_nid_with_pseudos] = labels_with_pseudos.float()[train_nid_with_pseudos]
        else:
            label_emb = torch.zeros((data.num_nodes, num_classes), device=device)
            label_emb[train_nid_with_pseudos] = F.one_hot(
                labels_with_pseudos[train_nid_with_pseudos], num_classes=num_classes
            ).float()
    else:
        label_emb = None

    if is_inductive:
        if osp.exists(label_emb_path) and load_emb:
            label_emb = torch.load(label_emb_path)
        elif label_emb is not None:
            data.train()
            label_emb_train = average_neighbor_features(data, label_emb, nhop, norm=norm, style="last")
            data.eval()
            label_emb = average_neighbor_features(data, label_emb, nhop, norm=norm, style="last")
            label_emb[train_nid] = label_emb_train[train_nid]
            if load_emb:
                torch.save(label_emb, label_emb_path)
    else:
        if osp.exists(label_emb_path) and load_emb:
            label_emb = torch.load(label_emb_path)
        elif label_emb is not None:
            if label_emb is not None:
                label_emb = average_neighbor_features(data, label_emb, nhop, norm=norm, style="last")
            if stage == 0 and load_emb:
                torch.save(label_emb, label_emb_path)

    return label_emb, labels_with_pseudos, train_nid_with_pseudos


@register_model("sagn")
class SAGN(BaseModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-size", type=int, default=512)
        parser.add_argument("--negative-slope", type=float, default=0.2)
        parser.add_argument("--dropout", type=float, default=0.5)
        parser.add_argument("--input-drop", type=float, default=0.0)
        parser.add_argument("--attn-drop", type=float, default=0.4)
        parser.add_argument("--nhead", type=int, default=2)
        parser.add_argument("--mlp-layer", type=int, default=4)
        parser.add_argument("--use-labels", action="store_true")
        parser.add_argument("--nhop", type=int, default=4)

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.num_features,
            args.num_classes,
            args.hidden_size,
            args.nhop,
            args.mlp_layer,
            args.nhead,
            args.dropout,
            args.input_drop,
            args.attn_drop,
            args.negative_slope,
            args.use_labels,
        )

    def __init__(
        self,
        in_feats,
        out_feats,
        hidden_size,
        nhop,
        mlp_layer,
        nhead,
        dropout=0.5,
        input_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        use_labels=False,
    ):
        super(SAGN, self).__init__()
        self.dropout = dropout
        self.nhead = nhead
        self.hidden_size = hidden_size
        self.attn_dropout = attn_drop
        self.input_dropout = input_drop
        self.use_labels = use_labels
        self.negative_slope = negative_slope

        self.norm = nn.BatchNorm1d(hidden_size)
        self.layers = nn.ModuleList(
            [
                MLP(in_feats, hidden_size * nhead, hidden_size, mlp_layer, norm="batchnorm", dropout=dropout)
                for _ in range(nhop + 1)
            ]
        )

        self.mlp = MLP(hidden_size, out_feats, hidden_size, mlp_layer, norm="batchnorm", dropout=dropout)
        self.res_conn = nn.Linear(in_feats, hidden_size * nhead, bias=False)
        if use_labels:
            self.label_mlp = MLP(out_feats, out_feats, hidden_size, 2 * mlp_layer, norm="batchnorm", dropout=dropout)

        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, nhead, hidden_size)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, nhead, hidden_size)))

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            layer.reset_parameters()
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if self.use_labels:
            self.label_mlp.reset_parameters()
        self.norm.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, features, y_emb=None):
        out = 0
        features = [F.dropout(x, p=self.input_dropout, training=self.training) for x in features]
        hidden = [self.layers[i](features[i]).view(-1, self.nhead, self.hidden_size) for i in range(len(features))]
        a_r = (hidden[0] * self.attn_r).sum(dim=-1).unsqueeze(-1)
        a_ls = [(h * self.attn_l).sum(dim=-1).unsqueeze(-1) for h in hidden]
        a = torch.cat([(a_l + a_r).unsqueeze(-1) for a_l in a_ls], dim=-1)
        a = F.leaky_relu(a, negative_slope=self.negative_slope)
        a = F.softmax(a, dim=-1)
        a = F.dropout(a, p=self.attn_dropout, training=self.training)

        for i in range(a.shape[-1]):
            out += hidden[i] * a[:, :, :, i]
        out += self.res_conn(features[0]).view(-1, self.nhead, self.hidden_size)
        out = out.mean(1)
        out = F.relu(self.norm(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.mlp(out)

        if self.use_labels and y_emb is not None:
            out += self.label_mlp(y_emb)
        return out

    @staticmethod
    def get_trainer(args=None):
        return SAGNTrainer


# @register_trainer("sagn_trainer")
class SAGNTrainer(BaseTrainer):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--nstage", type=int, nargs="+", default=[1000, 500, 500])
        parser.add_argument("--batch-size", type=int, default=2000)
        parser.add_argument(
            "--threshold", type=float, default=0.9, help="threshold used to generate pseudo hard labels"
        )
        parser.add_argument("--label-nhop", type=int, default=4)
        parser.add_argument("--data-gpu", action="store_true")

    @classmethod
    def build_trainer_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(SAGNTrainer, self).__init__(args)
        self.batch_size = args.batch_size
        self.nstage = args.nstage
        self.nhop = args.nhop
        self.threshold = args.threshold
        self.data_device = self.device if args.data_gpu else "cpu"
        self.label_nhop = args.label_nhop if args.label_nhop > -1 else args.nhop

    def fit(self, model, dataset):
        data = dataset.data
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = dataset.get_loss_fn()
        self.evaluator = dataset.get_evaluator()

        data.to(self.data_device)
        feats = prepare_feats(dataset, self.nhop)

        train_nid, val_nid, test_nid = data.train_nid, data.val_nid, data.test_nid
        all_nid = torch.cat([train_nid, val_nid, test_nid])

        val_loader = torch.utils.data.DataLoader(val_nid, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_nid, batch_size=self.batch_size, shuffle=False)
        all_loader = torch.utils.data.DataLoader(all_nid, batch_size=self.batch_size, shuffle=False)
        patience = 0
        best_val = 0
        best_model = None
        probs = None

        test_metric_list = []
        for stage in range(len(self.nstage)):
            print(f"In stage {stage}..")
            with torch.no_grad():
                (label_emb, labels_with_pseudos, train_nid_with_pseudos) = prepare_labels(
                    dataset, stage, self.label_nhop, self.threshold, probs=probs
                )

            labels_with_pseudos = labels_with_pseudos.to(self.data_device)
            if label_emb is not None:
                label_emb = label_emb.to(self.data_device)

            epoch_iter = tqdm(range(self.nstage[stage]))
            for epoch in epoch_iter:
                train_loader = torch.utils.data.DataLoader(
                    train_nid_with_pseudos.cpu(), batch_size=self.batch_size, shuffle=True
                )
                self.train_step(train_loader, feats, label_emb, labels_with_pseudos)
                val_loss, val_metric = self.test_step(val_loader, feats, label_emb, data.y[val_nid])
                if val_metric > best_val:
                    best_val = val_metric
                    best_model = copy.deepcopy(model)
                    patience = 0
                else:
                    patience += 1
                    if patience > self.patience:
                        epoch_iter.close()
                        break
                epoch_iter.set_description(f"Epoch: {epoch: 03d}, ValLoss: {val_loss: .4f}, ValAcc: {val_metric: .4f}")
            temp_model = self.model
            self.model = best_model
            test_loss, test_acc = self.test_step(test_loader, feats, label_emb, data.y[test_nid])
            test_metric_list.append(round(test_acc, 4))

            self.model = temp_model
            probs = self.test_step(all_loader, feats, label_emb, data.y[all_nid], return_probs=True)
        test_metric = ", ".join([str(x) for x in test_metric_list])
        print(test_metric)

        return dict(Acc=test_metric_list[-1])

    def train_step(self, train_loader, feats, label_emb, y):
        device = next(self.model.parameters()).device
        self.model.train()
        for batch in train_loader:
            self.optimizer.zero_grad()
            batch = batch.to(device)
            batch_x = [x[batch].to(device) for x in feats]

            if label_emb is not None:
                batch_y_emb = label_emb[batch].to(device)
            else:
                batch_y_emb = None
            pred = self.model(batch_x, batch_y_emb)
            loss = self.loss_fn(pred, y[batch].to(device))
            loss.backward()
            self.optimizer.step()

    def test_step(self, eval_loader, feats, label_emb, y, return_probs=False):
        self.model.eval()
        preds = []

        device = next(self.model.parameters()).device
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                batch_x = [x[batch].to(device) for x in feats]
                if label_emb is not None:
                    batch_y_emb = label_emb[batch].to(device)
                else:
                    batch_y_emb = None
                pred = self.model(batch_x, batch_y_emb)
                preds.append(pred.to(self.data_device))
        preds = torch.cat(preds, dim=0)
        if return_probs:
            return preds
        loss = self.loss_fn(preds, y)
        metric = self.evaluator(preds, y)
        return loss, metric
