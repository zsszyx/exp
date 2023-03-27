import argparse
import os.path as osp
import random
import torch
import numpy as np
from torch.backends import cudnn
import datasets
import sys
from utils.logging import Logger
from torch import nn
from utils.data.fetch import get_data, get_test_loader, get_train_loader
from utils.data import CamLoader
from models import create, Trainer, device, low_rank_cov_change
from evaluation_metrics import Evaluator, extract_features
from utils.faiss_rerank import compute_jaccard_distance
from sklearn.cluster import DBSCAN
import collections
from torch.nn import functional as F
from utils.serialization import load_checkpoint, save_checkpoint
from IPython import embed

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
start_epoch = best_mAP = 0


def work(args):
    global start_epoch, best_mAP
    cudnn.benchmark = True
    # sys.stdout = Logger('examples/logs/log.txt')
    print("==========\nArgs:{}\n==========".format(args))
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, '/media/lab225/diskA/dataset/ReID-data')
    test_loader = get_test_loader(dataset, 256, 128, args.batch_size, 4)
    # Create model
    model = create()
    # model = low_rank_cov_change(model)
    model = nn.DataParallel(model)
    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.00035, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    train = Trainer(model, args.iters, args.temp, momentum=args.momentum, temperature=1).to(device)
    for i in range(args.epochs):
        epoch = i + 1
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, 256, 128, args.batch_size, 4, testset=sorted(dataset.train))
            features, _ = extract_features(model, cluster_loader, print_freq=50)
            f_list = []
            cams = []
            files = []
            for f, _, c in sorted(dataset.train):
                f_list.append(features[f].unsqueeze(0))
                files.append(f)
                cams.append(c)
            features = torch.cat(f_list, 0)
            cams = np.array(cams)
            files = np.array(files)
            rerank_dist = compute_jaccard_distance(features, k1=30, k2=6)
            if epoch == 1:
                # DBSCAN cluster
                eps = 0.6
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

        centrals = generate_cluster_features(pseudo_labels, features)
        cams = cams[np.where(pseudo_labels != -1)]
        files = files[np.where(pseudo_labels != -1)]
        cam_loader = CamLoader(files, cams, args.batch_size, args.dataset)

        del cluster_loader, features, cams, files
        train.register_buffer('centrals', F.normalize(centrals, dim=1).to(device))
        if epoch == 1:
            old_labels = pseudo_labels.copy()

        pseudo_labeled_dataset = []
        for j, ((fname, _, cid), label, old_label) in enumerate(zip(sorted(dataset.train), pseudo_labels, old_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), old_label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))
        train_loader = get_train_loader(dataset, 256, 128, args.batch_size, 4, 16, iters,
                                        trainset=pseudo_labeled_dataset, old_label=True)
        train_loader.new_epoch()
        train.iters = iters
        train.start(train_loader, epoch, optimizer, cam_loader=cam_loader)
        old_labels = pseudo_labels

        if epoch % 10 == 0 or (epoch == args.epochs):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join('examples/logs', 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join('examples/logs', 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, file=args.dataset)


@torch.no_grad()
def generate_cluster_features(labels, features):
    features = features.to(device)
    labels = torch.tensor(labels).to(device)
    unique_ids = torch.unique(labels)
    unique_ids = unique_ids[torch.where(unique_ids != -1)]
    # 对于每个唯一的 id 标签，找到相同的 camera 标签
    centrals = []
    unique_ids, _ = torch.sort(unique_ids)
    for i in unique_ids:
        centrals.append(features[torch.where(labels == i)].mean(0))
    centrals = torch.stack(centrals, dim=0)

    return centrals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)

    # cluster
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")

    # optimizer
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    args = parser.parse_args()
    work(args)
