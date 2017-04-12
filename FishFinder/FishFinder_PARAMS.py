# General
num_channels = 3
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
nodes_after_conv = 7168

fc_depth = [2048, 512, 128, 32]
keep_prob = [0.6, 0.7, 0.8, 0.9]

fov_dim = 72

counter = 0
predict_every_z = 20
