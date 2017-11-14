from trainval_net import *


imdb_name = "coco_style_pill"
imdbval_name = "coco_style_pill_val"

np.random.seed(cfg.RNG_SEED)

# train set
imdb, roidb = combined_roidb(imdb_name)
print('{:d} roidb entries'.format(len(roidb)))

# output directory where the models are saved
output_dir = get_output_dir(imdb, None)
print('Output will be saved to `{:s}`'.format(output_dir))

# tensorboard directory where the summaries are saved during training
tb_dir = get_output_tb_dir(imdb, None)
print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

# also add the validation set, but with no flipping images
orgflip = cfg.TRAIN.USE_FLIPPED
cfg.TRAIN.USE_FLIPPED = False
_, valroidb = combined_roidb(imdbval_name)
print('{:d} validation roidb entries'.format(len(valroidb)))
cfg.TRAIN.USE_FLIPPED = orgflip

# load network
net = resnetv1(num_layers=50)

train_net(net,
          imdb,
          roidb,
          valroidb,
          output_dir,
          tb_dir,
          pretrained_model="/home/xum/Documents/Git/PillDetection/data/imagenet_weights/res50.ckpt",
          max_iters=9000)


