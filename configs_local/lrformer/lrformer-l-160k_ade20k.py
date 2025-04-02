_base_ = ['../_base_/datasets/ade20k_640x640.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (640, 640)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    )

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
	backbone=dict(
        type='LRFormer',
        _delete_=True,
        q_conv=False,
        use_ls=True,
        extra_act=False,
        q_pooled_sizes=[16, 16, 16, 16],
        embed_dims=[96, 192, 480, 640],
        depths=[4, 6, 18, 8],
        drop_path_rate=0.6,
        checkpoint='pretrained/lrformer_large.pth',
    ),
    decode_head=dict(
        type='LRHead',
        in_channels=[192, 480, 640],
        channels=640,
        in_index=[1, 2, 3],
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='AlignResize', scale=(2560, 640), size_divisor=32, keep_ratio=True),
    # In LRFormerv1, we follow SegFormer to use AlignResize
    # Using 'Resize' will decrease 0.2% performance in validation
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
    custom_keys={'norm': dict(decay_mult=0.), 'head': dict(lr_mult=10.)}
    )
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1,
                      dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader
