# python DDP_byol_ccrop.py path/to/this/config

# model
hidden_dim = 512
model = dict(type='ResNet', depth=18, num_classes=hidden_dim, maxpool=False)
byol = dict(dim=hidden_dim, pred_dim=128, m=0.996)
loss = dict(type='CrossEntropyLoss')
# data
root_train = './datasets/BDD100K/100k/train'
root_val = './datasets/BDD100K/100k/val'
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 256
num_workers = 4
data = dict(
    train=dict(
        ds_dict=dict(
            type='BDD100KCCrop',
            root=root_train,
        ),
        rcrop_dict=dict(
            type='BDD100K_pretrain_rcrop',
            mean=mean, std=std
        ),
        ccrop_dict=dict(
            type='BDD100K_pretrain_ccrop',
            alpha=0.1,
            mean=mean, std=std
        ),
    ),
    eval_train=dict(
        ds_dict=dict(
            type='BDD100KCCrop',
            root=root_val,
            train=True,
        ),
        trans_dict=dict(
            type='BDD100K_val',
            mean=mean, std=std
        ),
    ),
)
# training optimizer & scheduler
epochs = 100
lr = 30.
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0)
lr_cfg = dict(  # passed to adjust_learning_rate(cfg=lr_cfg)
    type='MultiStep',
    steps=epochs,
    lr=lr,
    decay_rate=0.1,
    decay_steps=[60, 80],
    warmup_steps=0,
    # warmup_from=0.01
    )


# log & save
log_interval = 100
work_dir = None  # rewritten by args
resume = None
load = None
port = 10001
