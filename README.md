# 3D-Classification-and-Recognition-on-point-sets-by-Deep-Learning

### Deep Learning on Point Sets using PointNet architecture

### Introduction
This work is based on [arXiv tech report](https://arxiv.org/abs/1612.00593), which appeared in CVPR 2017. A novel deep net architecture for point clouds (as unordered point sets). You can also check [project webpage](http://stanford.edu/~rqi/pointnet) for a deeper introduction.

Point cloud is an important type of geometric data structure. Due to its irregular format, most researchers transform such data to regular 3D voxel grids or collections of images. This, however, renders data unnecessarily voluminous and causes issues. In this approach, a novel type of neural network that directly consumes point clouds, which well respects the permutation invariance of points in the input. PointNet provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing. Though simple, PointNet is highly efficient and effective.

  
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. You may also need to install h5py. 

If you are using PyTorch, you can find a third-party pytorch implementation <a href="https://github.com/fxia22/pointnet.pytorch" target="_blank">here</a>.

To install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

### Usage
To train a model to classify point clouds sampled from 3D shapes in classification directory:

    python train.py

Log files and network parameters will be saved to `log` folder in default. Download the Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models to the data folder, then convert these files after subsampling to hdf5 format using files provided in `classification/data` directory. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. 

To see HELP for the training script:

    python train.py -h

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir log

After the above training, we can evaluate the model and output some visualizations of the error cases.

    python evaluate.py --visu

Point clouds that are wrongly classified will be saved to `dump` folder in default. We visualize the point cloud by rendering it into three-view images.


### To Do:
Part segmentation and Semantic segmantation datasets preparation are still under process
