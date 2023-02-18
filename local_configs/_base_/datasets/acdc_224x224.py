# dataset settings
dataset_type = 'ACDCDataset'
data_root = 'data/acdc/'
img_norm_cfg = dict(
    mean=[21.998, 21.998, 21.998], std=[48.642, 48.642, 48.642], to_rgb=True)
img_scale = (256, 256)
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.75, 1.5)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=(-30., 30.)),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/image',
            ann_dir='train/label',
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='val/image',
            ann_dir='val/label',
            pipeline=train_pipeline)
    ],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/image',
        ann_dir='test/label',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/image',
        ann_dir='test/label',
        pipeline=test_pipeline)
)
