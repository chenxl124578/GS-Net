# GS-Net: Point Cloud Sampling with Graph Neural Networks




## Environment
Environment:Python3.6.9, PyTorch1.7.1, gcc 6 or 7 or higher!(gcc 4 will get ninja bug, this is important)

### if meet ninja bug
PW: if you get bug like: "RuntimeError: Ninja is required to load C++ extensions" "Command ‘[‘ninja‘, ‘-v‘]‘ returned non-zero exit status 1" may be the problem is ninja version or ninja install error, you should reinstall ninja. 

renew debug method: there are 2 methods can resolve this problem. The first one is change "command = ['ninja', '-v']" to "command = ['ninja', '--version']" in "envs/gsnet/lib/python3.6/site-packages/torch/utils/cpp_extension.py/_run_ninja_build". The seconde one is reinstall ninja. I recommend the first method.

Reinstall ninja:<br>
First: install re2c in http://re2c.org/index.html<br>
Here I install re2c3.0: wget https://github.com/skvadrik/re2c/releases/download/3.0/re2c-3.0.tar.xz

Second: unzip and make install re2c-3.0 <br>
```
tar -xvf re2c-3.0.tar.xz 
cd re2c-3.0 
autoreconf -i -W all 
./configure --prefix=/home/yourName/ninjatest 
make 
make install 
make check   (for test re2c install) 
```
Third: export re2c PATH <br>
```
vi ~/.bashrc (and add low command) 
export PATH="/disk2/cxl/ninjatest/bin:$PATH" 
source ~/.bashrc 
re2c -V (test re2c) 
```
Fourth: start install ninja <br>
```
cd .. 
git clone https://github.com/ninja-build/ninja.git && cd ninja 
./configure.py --bootstrap 
```
fifth: export ninja PATH <br>
```
vi ~/.bashrc (and add low command)
export PATH="/disk2/cxl/ninja:$PATH" 
source ~/.bashrc 
ninja --version  
```
### Install environment

1.install pointnet2_ops
```
down load Pointnet2_Pytorch:
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
```
then install CUDA kernels(use grouping_operation in softprojection in samplenet)<br>
```
pip install pointnet2_ops_lib/.
```
2.install KNN_CUDA
KNN_CUDA provide by https://github.com/unlimblue/KNN_CUDA   <br>

First: install KNN_CUDA<br>
```
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
Second: import knn_cuda in python3 (in terminal), may be post error "RuntimeError: Ninja is required to load C++ extensions"
https://blog.csdn.net/xiaoyaolangwj/article/details/119382717


3.install PyG(torch_geometric) for building Point Graph
https://github.com/pyg-team/pytorch_geometric
https://zhuanlan.zhihu.com/p/381204915

actually, you can install by following command, （与1.7.0相同版本）:
```
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_scatter-2.0.7-cp36-cp36m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_sparse-0.6.9-cp36-cp36m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_cluster-1.5.9-cp36-cp36m-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_spline_conv-1.2.1-cp36-cp36m-linux_x86_64.whl
pip install torch-geometric
```

## dataset prepare
ModelNet40: https://modelnet.cs.princeton.edu/ <br>
ScanObjectNN: https://hkust-vgd.github.io/scanobjectnn/ <br>
SemanticKITTI-cls: https://drive.google.com/file/d/1mf0GIVmd0BU_lakTBGpMcdT1jrCxdUyM/view?usp=sharing <br>
<br>
data<br>
├── modelnet40_ply_hdf5_2048               # ModelNet40<br>
├── scanobjectnn/h5_files/main_split            # ScanObjectNN<br>
└── semantickitti_cls       # SemanticKITTI-cls<br>


## Usage

### Classifier Training:
train pointnet classifier on Modelnet40 use:
```
 python train_classifier --dataset modelnet40 --num_category 40 
```
train pointnet classifier on ScanObjectNN use:
```
 python train_classifier --dataset scanobjectnn --dataclass OBJ_BG --num_category 15
```
train pointnet classifier on SemanticKITTI-cls use:
```
 python train_classifier --dataset semantickitti_cls --num_category 6
```

### Sampling Network Training:
After train classifier, joinly train sampling network.
For training GS-Net on ModelNet40 and PointNet classifier with sampling rate r=32, you can use: 
```
 python train_gsnet.py --dataset modelnet40 --classifier_model pointnet_cls --classifier_model_path weights/your_PointNet_classifier_modelnet_path.pth --num_category 40 --not_debug --assign_ratio 0.03125
```

For training GS-Net on ScanObjectNN and PointNet classifier with sampling rate r=32, you can use: 
```
 python train_gsnet.py --dataset scanobjectnn --classifier_model pointnet_cls --classifier_model_path weights/your_PointNet_classifier_scanobjectnn_path.pth --num_category 15 --not_debug --assign_ratio 0.03125
```

For training GS-Net on SemanticKITTI-cls and PointNet classifier with sampling rate r=32, you can use:
```
 python train_gsnet.py --dataset semantickitti_cls --classifier_model pointnet_cls --classifier_model_path weights/your_PointNet_classifier_semantickitticls_path.pth --num_category 6 --not_debug --assign_ratio 0.03125 --alpha 0.01
```

<br>

### Classifier Evaluating:
if need visulization add ```--vis_sampled_point```, if need save to ply add ```--save_sampled_point```
evaluate pointnet classifier on ModelNet40 use:
```
python eval_classifier.py --dataset modelnet40 --classifier_model_path weights/PointNet_classifier_model_modelnet.pth  --num_category 40 --not_debug 
```

evaluate pointnet classifier on ScanObjectNN use:
```
python eval_classifier.py --dataset scanobjectnn --classifier_model_path weights/PointNet_classifier_model_scanobjectnn.pth  --num_category 15 --not_debug 
```

evaluate pointnet classifier on SemanticKITTI-cls use:
```
python eval_classifier.py --dataset semantickitti_cls --classifier_model_path weights/PointNet_classifier_model_semantickitti_cls.pth  --num_category 6 --not_debug 
```

### GS-Net Evaluating:
Evaluating GS-Net base on PointNet classifier and ModelNet40 with r=32, you can use:
```
python eval_gsnet.py --dataset modelnet40 --not_debug --match --assign_ratio 0.03125 --points_noise 0.1 --sampler_model_path log/gsnet_pyg/log_dir/checkpoints/best_model.pth
```


## Thanks
Thanks for [Samplenet(TF)](https://github.com/itailang/SampleNet), [Pointnet_Pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch),[PointNet_Pytorch(for environment install)](https://github.com/erikwijmans/Pointnet2_PyTorch).
