# General
num_channels = 3
num_labels = 8
coarse_dims = [64,112,3]
batch_size = 128

stddev = 0.2


# Convolutions
conv_kernel = 3
conv_stride = 1

pretrained_path = '../../PreTrained_Models/VGG_19/variables/'
conv_depth = [64, 64,
              128, 128,
              256, 256] #, 256, 256,
             # 512, 512, 512, 512]
# Pooling
pool_kernel = 2
pool_stride = 2

# Dense layers
nodes_after_conv = 4096

fc_depth = [1024, 1024, 256, 64]
keep_prob = [0.7, 0.7, 0.7, 0.7]

fov_dim = 72
fov_crop = 64
