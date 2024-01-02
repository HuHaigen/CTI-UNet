norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='CTIUNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False,
        patch_size=16,
        num_layers=1,
        embed_dims=384,
        num_heads=6,
        mlp_ratio=4,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        num_fcs=2,
        qkv_bias=True,
        trans_norm_cfg=dict(type='LN'),
        trans_act_cfg=dict(type='GELU'),
        pos_emb=True,
        interacted=(1, 2, 3, 4),
        ca_cfg=dict(type='CBAM_CA'),
        sa_cfg=dict(type='CBAM_SA'),
        aux_head=True,
        pre_num_layers=4),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=0.5),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    auxiliary_head=dict(
        type='CTIPUPHead',
        in_channels=384,
        channels=64,
        in_index=5,
        num_classes=9,
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_convs=2,
        up_scale=4,
        kernel_size=3,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                use_sigmoid=False,
                loss_weight=0.5),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5)
        ]))
dataset_type = 'SynapseDataset'
data_root = 'data/synapse/'
img_norm_cfg = dict(
    mean=[11.827, 11.827, 11.827], std=[38.205, 38.205, 38.205], to_rgb=True)
img_scale = (336, 336)
crop_size = (224, 224)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(336, 336), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(224, 224), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=(-30.0, 30.0)),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[11.827, 11.827, 11.827],
        std=[38.205, 38.205, 38.205],
        to_rgb=True),
    dict(type='Pad', size=(224, 224), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[11.827, 11.827, 11.827],
                std=[38.205, 38.205, 38.205],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
    train=dict(
        type='SynapseDataset',
        data_root='data/synapse/',
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(336, 336), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(224, 224), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='RandomRotate', prob=0.5, degree=(-30.0, 30.0)),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[11.827, 11.827, 11.827],
                std=[38.205, 38.205, 38.205],
                to_rgb=True),
            dict(type='Pad', size=(224, 224), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='SynapseDataset',
        data_root='data/synapse/',
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[11.827, 11.827, 11.827],
                        std=[38.205, 38.205, 38.205],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='SynapseDataset',
        data_root='data/synapse/',
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[11.827, 11.827, 11.827],
                        std=[38.205, 38.205, 38.205],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=2000, metric=['mIoU', 'mDice'], pre_eval=True)
