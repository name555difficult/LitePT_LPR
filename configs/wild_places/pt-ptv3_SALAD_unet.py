_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
num_worker = 4
batch_size = 2048  # bs: total bs in all gpus
batch_split_size = 64
batch_size_val = 64
# batch_size_val = 64 
batch_size_test = batch_size_val
mix_prob = 0
empty_cache = False
enable_amp = False

enable_wandb = False
wandb_project = "LitePT_LPR"  # custom your project name e.g. Sonata, PTv3

# model settings
model = dict(
    type="DefaultPlaceRecognitioner",
    backbone=dict(
        type="LitePT",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(72, 144, 288, 576, 864),
        enc_num_head=(4, 8, 16, 32, 48),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 144, 288, 576),
        dec_num_head=(4, 8, 16, 32),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    head=dict(
        type="SALAD",
        feature_size=72,
        num_clusters=64,
        cluster_dim=64,
        token_dim=512,
        dropout=0.3,
        use_sinkhorn=True,
        coarse_cfg = dict(
            feature_size=576,
            p=3,
            eps=1e-6,
        ),
    ),
    criteria=[
        dict(type="TruncatedSmoothAP", tau1=0.01, positives_per_query=4),
    ],
)

# scheduler settings
epoch = 80
eval_epoch = 80
eval_interval_epoch = 10
# optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
# scheduler = dict(
#     type="OneCycleLR",
#     max_lr=[0.002, 0.0002],
#     pct_start=0.04,
#     anneal_strategy="cos",
#     div_factor=10.0,
#     final_div_factor=100.0,
# )
# param_dicts = [dict(keyword="block", lr=0.0002)]

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)

# 注意：max_lr 的顺序要与 optimizer 参数组顺序一致
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],   # 两个 param group 对应两路峰值 LR
    pct_start=0.0,            # 无上升阶段
    anneal_strategy="cos",    # 余弦式下降
    div_factor=1.0,           # 初始 LR = max_lr（无 warmup）
    final_div_factor=100.0,   # 末端 LR = max_lr / 100
    cycle_momentum=False,     # 用 AdamW 建议关闭
    # total_steps 由训练循环/runner 推断；若需要手动：epochs * iters_per_epoch（考虑梯度累积）
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "WildPlacesDataset"
data_root = "data/wild_places/data"

data = dict(
    train=dict(
        type=dataset_type,
        split="training_wild-places.pickle",
        data_root=data_root,
        transform=[
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type='NormalizeCoord'),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "positives", "non_negatives", "label"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
        # dense=True,
    ),
    val=dict(
        type=dataset_type,
        split="testing_wild-places.pickle",
        data_root=data_root,
        transform=[
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(type='NormalizeCoord'),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "positives", "non_negatives", "label"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
        dense=True,
    ),
    test=dict(
        type='WildPlacesEvalDataset',
        split="test",
        data_root=data_root,
        pickle_files=[
            ['Karawatha_evaluation_database.pickle', 'Karawatha_evaluation_query.pickle'],
            ['Venman_evaluation_database.pickle', 'Venman_evaluation_query.pickle'],
        ],
        transform=[
            # dict(type="PointClip", point_cloud_range=(-51.2, -51.2, -4, 51.2, 51.2, 2.4)),
            dict(type='NormalizeCoord'),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord"),
                feat_keys=("coord", ),
            ),
        ],
        test_mode=False,
        # dense=True,
    ),
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # dict(type="SemSegEvaluator"),
    dict(type="WildPlacesEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="WildPlacesTrainer")

# Tester
test = dict(
    type="WildPlacesTester", 
    verbose=True,
    skip_same_run=True,
    eval_no_neighbors=False,
    no_neighbors_sample_ratio=0.1,
    auto_threshold_scale=1.0,
    )