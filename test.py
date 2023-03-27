import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import torch
from models import create
from evaluation_metrics import Evaluator
from utils.data.fetch import get_data, get_test_loader
from models import low_rank
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create().to(device)
model = low_rank.low_rank_cov_change(model)
# evaluator = Evaluator(model)
# dataset = get_data('market1501', '/media/lab225/diskA/dataset/ReID-data')
# test_loader = get_test_loader(dataset, 256, 128, 64, 4)
#
# evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True, file='test')