from trainval_net import *

"""
Parse input arguments
"""
parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
parser.add_argument('--weight', dest='weight', help='initialize with pretrained model weights', type=str)
parser.add_argument('--imdb', dest='imdb_name', help='dataset to train on', default='coco_style_pill', type=str)
parser.add_argument('--imdbval', dest='imdbval_name', help='dataset to validate on', default='coco_style_pill_val',
                    type=str)
parser.add_argument('--iters', dest='max_iters', help='number of iterations to train', default=7000, type=int)
parser.add_argument('--tag', dest='tag', help='tag of the model', default=None, type=str)
parser.add_argument('--net', dest='net', help='vgg16, res50, res101, res152, mobile', default='res50', type=str)
parser.add_argument('--set', dest='set_cfgs', help='set config keys', default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()

print('Called with args:')
print(args)

if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

print('Using config:')
pprint.pprint(cfg)

np.random.seed(cfg.RNG_SEED)

# train set
imdb, roidb = combined_roidb(args.imdb_name)
print('{:d} roidb entries'.format(len(roidb)))

# output directory where the models are saved
output_dir = get_output_dir(imdb, args.tag)
print('Output will be saved to `{:s}`'.format(output_dir))

# tensorboard directory where the summaries are saved during training
tb_dir = get_output_tb_dir(imdb, args.tag)
print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

# also add the validation set, but with no flipping images
orgflip = cfg.TRAIN.USE_FLIPPED
cfg.TRAIN.USE_FLIPPED = False
_, valroidb = combined_roidb(args.imdbval_name)
print('{:d} validation roidb entries'.format(len(valroidb)))
cfg.TRAIN.USE_FLIPPED = orgflip

# load network
net = resnetv1(num_layers=50)

train_net(net, imdb, roidb, valroidb, output_dir, tb_dir, pretrained_model=args.weight, max_iters=args.max_iters)


