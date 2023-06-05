import os
import click

@click.command()
@click.option('--reg_target_fake', default='gfeat', help='The regularization when discriminating fake images',
              type=click.Choice(['gfeat', 'gfwimage']))
def main(reg_target_fake):
    if reg_target_fake == 'gfeat':
        reg_loss_fake = 'cos'
        reg_weight_fake = 10
    elif reg_target_fake == 'gfwimage':
        reg_loss_fake = 'lpips'
        reg_weight_fake = 3

    cmd = f'python train.py \
    --data=/home/qingyan/data/lsun_church_train_crop256.zip \
    --outdir=exp  \
    --gpus=8 \
    --cfg=paper256 \
    --aug=noaug \
    --metrics=fid50k_full \
    --kimg=50000 \
    --reg_type=glead \
    --decoder_conv_kernel_size=1 \
    --glead_res_g=32 \
    --glead_res_d=64 \
    --reg_target_real=gfwimage \
    --reg_loss_real=lpips \
    --reg_weight_real=10 \
    --reg_target_fake={reg_target_fake} \
    --reg_loss_fake={reg_loss_fake} \
    --reg_weight_fake={reg_weight_fake} \
    '
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    main()
