# python DDP_simsiam_ccrop.py path/to/this/config
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import BDD100K_boxes
# model
dim, pred_dim = 512, 128
model = dict(type='ResNet', depth=18, num_classes=dim, maxpool=False, zero_init_residual=True)
simsiam = dict(dim=dim, pred_dim=pred_dim)
loss = dict(type='CosineSimilarity', dim=1)

# data
root = './datasets/BDD100K_link/train'
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
batch_size = 512
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='BDD100K_boxes',
            root=root,
            train=True,
        ),
        rcrop_dict=dict(
            type='bdd100k_train_rcrop',
            mean=mean, std=std
        ),
        ccrop_dict=dict(
            type='bdd100k_train_ccrop',
            alpha=0.1,
            mean=mean, std=std
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            type='bdd100k_boxes',
            root=root,
            train=True,
        ),
        trans_dict=dict(
            type='bdd100k_test',
            mean=mean, std=std
        ),
    ),
)

# boxes
warmup_epochs = 100
loc_interval = 100
box_thresh = 0.10

# training optimizer & scheduler
epochs = 500
lr = 0.5
fix_pred_lr = True
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=1e-4)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='Cosine',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    # decay_steps=[100, 150]
    warmup_steps=0,
    # warmup_from=0.01
)


# log & save
log_interval = 20
save_interval = 250
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
