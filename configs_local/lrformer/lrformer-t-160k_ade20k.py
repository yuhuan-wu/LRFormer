_base_ = ['lrformer-s-160k_ade20k.py']

model = dict(
	backbone=dict(
        type='LRFormer',
        _delete_=True,
        q_conv=False,
        use_ls=True,
        extra_act=False,
        q_pooled_sizes=[16, 16, 16, 16],
        embed_dims=[48, 96, 240, 384],
        depths=[2,2,6,3],
        drop_path_rate=0.1,
        checkpoint='pretrained/lrformer_tiny.pth',
    ),
    decode_head=dict(
        in_channels=[96, 240, 384],
        channels=256,
    ),
)
