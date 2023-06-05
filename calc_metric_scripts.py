import os

data_path_church = '/home/qingyan/data/lsun_church_train_crop256.zip'
data_path_bedroom = '/home/qingyan/data/lsun_bedroom_train_crop256.zip'
data_path_ffhq = '/home/qingyan/data/ffhq_tf_256x256.zip'

pkl_list = [
       # ----------------Models on FFHQ
       '/home/qingyan/checkpoints/glead/ffhq256-paper256-glead-real_gfwimage-fake_gfwimage-~3.24.pkl',
       '/home/qingyan/checkpoints/glead/ffhq256-paper256-glead-real_gfwimage-fake_gfeat-~2.90.pkl',
       # ----------------Models on Church
       '/home/qingyan/checkpoints/glead/church-paper256-glead-real_gfwimage-fake_gfwimage-~2.82.pkl',
       '/home/qingyan/checkpoints/glead/church-paper256-glead-real_gfwimage-fake_gfeat-~2.15.pkl',
       # ----------------Models on Bedroom
       '/home/qingyan/checkpoints/glead/bedroom-paper256-glead-real_gfwimage-fake_gfwimage-~2.55.pkl',
       '/home/qingyan/checkpoints/glead/bedroom-paper256-glead-real_gfwimage-fake_gfeat-~2.72.pkl',
]

for pkl_path in pkl_list:
       if 'ffhq' in pkl_path:
              data = data_path_ffhq
       elif 'church' in pkl_path:
              data = data_path_church
       elif 'bedroom' in pkl_path:
              data = data_path_bedroom
       metric = 'fid50k' if 'bedroom' in pkl_path else 'fid50k_full'
       mirror = True if 'ffhq' in data else False
       cmd = f'python calc_metrics.py \
              --data={data} \
              --network={pkl_path} \
              --metrics={metric} \
              --mirror={mirror} \
              --gpus=3 \
              '

       print(cmd)
       os.system(cmd)