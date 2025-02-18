_base_ = ['./mask2former_r50_8xb2-160k_ade20k-512x512.py']
pretrained = 'pretrained/LRFormer_base.pth'  # noqa
depths = [4,4,15,8]
model = dict(
	backbone=dict(
        type='LRFormer',
        _delete_=True,
        q_conv=False,
        use_ls=True,
        extra_act=False,
        q_pooled_sizes=[16, 16, 16, 16],
        embed_dims=[80, 160, 400, 512],
        depths=depths,
        drop_path_rate=0.4,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    decode_head=dict(in_channels=[80, 160, 400, 512])
)

train_dataloader = dict(batch_size=4)


# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.block{stage_id+1}.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.patch_embed{stage_id+1}.norm': backbone_norm_multi
    for stage_id in range(len(depths))
})
custom_keys.update({
    f'backbone.block{stage_id+1}.{block_id}.attn.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

env_cfg = dict(
cudnn_benchmark = False
)
