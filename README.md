# GLeaD: Improving GANs with A Generator-Leading Task

> **GLeaD: Improving GANs with A Generator-Leading Task** <br>
> Qingyan Bai, Ceyuan Yang, Yinghao Xu, Xihui liu, Yujiu Yang, Yujun Shen <br>
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023

<div align=center>
<img src="./docs/assets/teaser.png" width=500px>
</div>


**Figure:** Concept diagram of our proposed generator-leading
task (bottom), as complementary to the discriminator-leading task
in the original formulation of GANs (upper). D is required to
extract representative features that can be adequately decoded by
G to reconstruct the input.

**[**[**Paper**](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_GLeaD_Improving_GANs_With_a_Generator-Leading_Task_CVPR_2023_paper.html)**]**
**[**[**Project Page**](https://ezioby.github.io/glead/)**]**

This work aims at improving Generative adversarial network (GAN) with a generator-leading task. 
GAN is formulated as a two-player game between a generator (G) and a discriminator (D), 
where D is asked to differentiate whether an image comes from real data or is produced by G. 
Under such a formulation, D plays as the rule maker and hence tends to dominate the competition. 
Towards a fairer game in GANs, we propose a new paradigm for adversarial training, 
which makes **G assign a task to D** as well. Specifically, given an image, 
we expect D to extract representative features that can be adequately decoded by G to reconstruct the input. 
That way, instead of learning freely, D is urged to align with the view of G for domain classification.

<div align=center>
<img src="./docs/assets/framework.png" width=500px>
</div>

## Preparing

**Environment.**
Pytorch 1.8.1 + CUDA 11.1 + Python 3.8. Use the following script to install other packages: 
```
pip install -r requirements.txt
```


**Data.** 
Please download FFHQ for face domain, and LSUN Church and Bedroom for indoor and outdoor scene, respectively. 
Note that we follow [StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) to preprocess the images.

**Pre-trained models.**
The pre-trained GAN models can be found at [Google Drive](https://drive.google.com/drive/folders/1mbv98tU93X44awMlqkMw2y-wPG65--dc?usp=sharing) for reproduction.
The following table could also be of use - Ours* indicates the model is trained with a combined strategy of GLeaD and GGDR, as in Table 1 in the paper:

| Model | Link
| :--- | :----------
|FFHQ_Ours | [Google Drive](https://drive.google.com/file/d/1ONjsojNMU89ASRxkrgATvp6nFIt5TDnw/view?usp=sharing)
|FFHQ_Ours* | [Google Drive](https://drive.google.com/file/d/16QmpdAED7MosC2GKy9ETliqEpP_uTFeq/view?usp=sharing)
|Church_Ours | [Google Drive](https://drive.google.com/file/d/1QApN10lRP54lxk9HLknxtH_mKRw19AXY/view?usp=sharing)
|Church_Ours* | [Google Drive](https://drive.google.com/file/d/1sFQxlDPcNf0WB3XfIwz3RWDtuXdPvrVX/view?usp=sharing)
|Bedroom_Ours | [Google Drive](https://drive.google.com/file/d/1AvPmkD_R-PmwCN7c-TZGavkHbFEzL3kG/view?usp=sharing)
|Bedroom_Ours* | [Google Drive](https://drive.google.com/file/d/1KetzBDxedTJoFzz_exFnTWpDItAVAPTb/view?usp=sharing)


## Training
To train the models under GLeaD, run:
```
python train_scripts_ffhq.py
python train_scripts_church.py
python train_scripts_bedroom.py
```
where the "reg_target_fake" option indicates the regularization strategy - using GLeaD (gfwimage) or GGDR (gfeat)  when discriminating generated images.
And "data" indicates the path of the dataset, "outdir" indicates the output directory, and "gpus" means the GPU amount for training. 

## Evaluating
To evaluate the models, run:
```
python calc_metric_scripts.py
```
Remember to reset the data_paths and pkl_paths in "pkl_list".

## Generating
To generate samples with the pre-trained model, run:
```
python generate.py --outdir=out --trunc=0.7 --seeds=600-605 --network=your_pkl_path
```


## Acknowledgement
Thanks to
[StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) and
[GGDR](https://github.com/naver-ai/GGDR)
for sharing the code.

## BibTeX

If you find our work helpful for your research, please consider to cite:
```bibtex
@inproceedings{bai2023glead,
    author    = {Bai, Qingyan and Yang, Ceyuan and Xu, Yinghao and Liu, Xihui and Yang, Yujiu and Shen, Yujun},
    title     = {GLeaD: Improving GANs With a Generator-Leading Task},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023},
    pages     = {12094-12104}
}
```