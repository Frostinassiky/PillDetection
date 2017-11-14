import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import pprint, pickle

filename = '/home/xum/Documents/Git/0_MissingLabel/tf-faster-rcnn/data/demo/COCO_test2017_000000000241.jpg'
id = 240/6+1
img = mpimg.imread(filename)
# gt_box = [
# [0.5, 976.125, 220.625, 104.375]
# ]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')
imgplot = ax1.imshow(img)
#
# for k in range(len(gt_box)):
#     p  = patches.Rectangle(
#         (gt_box[k][0]+np.random.normal(0, 10, 1), gt_box[k][1]+np.random.normal(0, 10, 1)[0]),
#         gt_box[k][2]+np.random.normal(0, 10, 1)[0], gt_box[k][3]-np.random.normal(0, 10, 1)[0],
#         fill=False,edgecolor="black"    )
#     ax1.add_patch( p  )
# #


det_file = "/home/xum/Documents/Git/0_MissingLabel/tf-faster-rcnn/output/res50/coco_2017_test/default/res50_faster_rcnn_iter_10000/detections.pkl"
data1 = pickle.load( open(det_file, 'rb') ) # cls*img*box*[x,y,h,w,s]
data = [data1[k][id-1] for k in range(len(data1))]
threshold = 0.5
color8 = [
'#a6cee3'  ,
'#1f78b4'  ,
'#b2df8a'  ,
'#33a02c'  ,
'#fb9a99'  ,
'#e31a1c'  ,
'#fdbf6f'  ,
'#ff7f00'
]
for k in range(1,13):
    data_cls = data[k]
    for l in range(len(data_cls)):
        box = data_cls[l]
        if box[4]>threshold:
            p = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor=color8[k%8])
            ax1.add_patch(p)
patch = [patches.Patch(color='black', label='Ground Truth')]
patch = []
for k in range(1,13):
    patch.append( patches.Patch(color=color8[k%8], label='Prediction for class '+ str(k)))#+', '+str(scope)+'X') )
ax1.legend(handles=patch)
plt.show()