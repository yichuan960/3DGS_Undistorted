实验环境，数据及运行步骤：

环境：
基于原始3DGS的环境，渲染模块使用diff-gaussian。

数据：
需要使用Robust_nerf和Nerf-on_the_go数据集的去畸变数据
数据处理步骤：
Robust-Nerf 未去畸变版本数据集下载地址：
https://storage.googleapis.com/jax3d-public/projects/robustnerf/robustnerf.tar.gz
Nerf-on-the-go 去畸变版本数据集下载地址：
https://huggingface.co/datasets/jkulhanek/nerfonthego-undistorted/tree/main
两个数据集的Stable Diffusion提取特征下载地址：
https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683%2FSP3%2FWOFXFT

下载数据后如果文件名未按照clutter clean extra分类好（主要是on-the-go数据集）：
对图像进行修改名称分类测试集 训练集：
python prep_data.py --dataset 数据集路径
对图像进行降采样：
python downsample.py --data_dir 数据集路径 --factor 降采样倍率

由于下载提供的相机参数文件无法直接使用：Robust未去畸变 Nerf-on-the-go的image内图像名称与预处理后的名称不一致
需在数据集路径下手动执行colmap命令生成相应参数并去畸变。
colmap feature_extractor --database_path database.db --image_path images --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 1

colmap exhaustive_matcher --database_path database.db --SiftMatching.use_gpu 1

colmap mapper --database_path database.db --image_pathimages  --output_path sparse/0 --Mapper.ba_global_function_tolerance=0.000001

colmap image_undistorter --image_path images --input_path sparse/0 --output_path dense --output_type COLMAP

数据集的相机参数生成好后，运行命令前检查config.json文件：
train_keyword 和test_keyword代表区分训练测试集的文件名关键字
factor则是代表降采样倍率（如图片大小和参数为4032*3024 一般降采样8 直接下载好的去畸变完成的nerf-on-the-go数据集则为1）

运行命令：
训练：
python train.py -s 数据集路径
渲染：
python render.py -m 训练生成的output模型路径

目前尝试调大或者调小了seg_overlap的参数值，0.40左右浮动0.02差别不大，但0.40左右的值确实是指标最为好的情况（对应mountain和statue数据集）
但psnr的值仅有20.8 不符合预期
目前怀疑的原因有以下几种情况：
1. 最初复现spotless时，使用colmap生成的pinhole相机参数和后续的原生opencv相机参数通过去畸变方法后的psnr值确实存在差距，后者的mountain的psnr能够达到22 但前者只有21
2. gsplat和原始3dgs渲染方法的差距（目前测试中）
3. 数据集本身的差别 robust数据集待测试验证 
