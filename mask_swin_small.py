

_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_minne.py',
    '../_base_/datasets/minneapple_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize',img_scale=[(1280,720)], # (640, 360) , (320, 180), (1280,720)
    multiscale_mode='value', keep_ratio=True),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(type='Rotate',
                     level=2,
                     img_fill_val=114),
                dict(type='Translate',
                     level=2,
                     direction='horizontal',
                     img_fill_val=114),
                dict(type='Translate',
                     level=2,
                     direction='vertical',
                     img_fill_val=114),
            ],
            [
                dict(type='Rotate',
                     level=2,
                     img_fill_val=114)
            ],
            [
                dict(type='Translate',
                     level=2,
                     direction='vertical',
                     img_fill_val=114)
            ],
            [
                dict(type='Translate',
                     level=2,
                     direction='horizontal',
                     img_fill_val=114)
            ],
            [
                dict(type='Translate',
                     level=2,
                     direction='vertical',
                     img_fill_val=114),
                dict(type='Translate',
                     level=2,
                     direction='horizontal',
                     img_fill_val=114),
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=8),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[24, 42, 75])
runner = dict(type='EpochBasedRunner', max_epochs=90)

