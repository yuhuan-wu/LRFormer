_base_ = ['lrformer_s_ade20k_160k.py']

model = dict(
	backbone=dict(
        type='LRFormer',
        _delete_=True,
        q_conv=False,
        use_ls=True,
        extra_act=False,
        q_pooled_sizes=[16, 16, 16, 16],
        embed_dims=[80, 160, 400, 512],
        depths=[4,4,15,8],
        drop_path_rate=0.4,
        checkpoint='pretrained/lrformer_base.pth',
    ),
    decode_head=dict(
        in_channels=[160, 400, 512],
        channels=512,
    ),
)
