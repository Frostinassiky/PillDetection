# task 2, analyse iou

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pprint, pickle

## define color set
colorA = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
]


## def functions (from google)
def box_overlaps(a, b):
    c = np.zeros((len(a), len(b)), dtype=np.float)
    if len(a) < 1:
        return c
    for k in range(len(a)):
        for l in range(len(b)):
            pre_box = a[ k ]
            gt_box = b[ l ]
            c[ k, l ] = calc_iou(pre_box, gt_box)
    return c


def calc_iou(box1, box2):
    # change to [x1,y1,x2,y2]
    boxA = [ box1[ 0 ], box1[ 1 ], box1[ 0 ] + box1[ 2 ], box1[ 1 ] + box1[ 3 ] ]
    boxB = [ box2[ 0 ], box2[ 1 ], box2[ 0 ] + box2[ 2 ], box2[ 1 ] + box2[ 3 ] ]
    return bb_intersection_over_union(boxA, boxB)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[ 0 ], boxB[ 0 ])
    yA = max(boxA[ 1 ], boxB[ 1 ])
    xB = min(boxA[ 2 ], boxB[ 2 ])
    yB = min(boxA[ 3 ], boxB[ 3 ])

    # compute the area of intersection rectangle
    if (xB - xA) < 0 or (yB - yA) < 0:
        return 0
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[ 2 ] - boxA[ 0 ] + 1) * (boxA[ 3 ] - boxA[ 1 ] + 1)
    boxBArea = (boxB[ 2 ] - boxB[ 0 ] + 1) * (boxB[ 3 ] - boxB[ 1 ] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


## setting
box_nums_frame = [ 1024 ]

## load test set
annfile = "/home/xum/Documents/Git/AlphaNext/AlphaModel/AG_CapitalExclamation/Frost/instances_test2017.json"
dataset = json.load(open(annfile, 'r'))
# pre process
image_list = [ ]
for image_data in dataset[ "images" ]:
    image_list.append(image_data[ 'id' ])
image_num = len(image_list)

## load GT: gt_box
GT_boxes = [ [ ] for k in range(len(image_list)) ]
for anno in dataset[ "annotations" ]:
    idx = image_list.index(anno[ "image_id" ])
    GT_boxes[ idx ].append(anno[ "bbox" ])
print('gt_boxes shap:', len(GT_boxes), len(GT_boxes[ 0 ]), len(GT_boxes[ 0 ][ 0 ]))

## generate Sliding Window: ss_box
proposals_x1 = np.array([ 1.0 for k in range(1024) ])
proposals_y1 = np.array([ 1.0 for k in range(1024) ])
proposals_x2 = np.array([ 1.0 for k in range(1024) ])
proposals_y2 = np.array([ 1.0 for k in range(1024) ])
for sizexy in range(4):
    for k in range(16):
        for l in range(16):
            proposals_x1[ sizexy * 256 + k * 16 + l ] = ((k) / (16 + 0.5 * 2 ** sizexy))
            proposals_y1[ sizexy * 256 + k * 16 + l ] = ((l) / (16 + 0.5 * 2 ** sizexy))
            proposals_x2[ sizexy * 256 + k * 16 + l ] = ((k + 1 + 0.5 * 2 ** sizexy) / (16 + 0.5 * 2 ** sizexy))
            proposals_y2[ sizexy * 256 + k * 16 + l ] = ((l + 1 + 0.5 * 2 ** sizexy) / (16 + 0.5 * 2 ** sizexy))
prp_boxes = np.array([ proposals_x1 * 1920, proposals_y1 * 1080, (proposals_x2 - proposals_x1) * 1920,
                       (proposals_y2 - proposals_y1) * 1080 ])
ss_boxes = [ prp_boxes.transpose() ] * image_num
print('ss_boxes shap:', len(ss_boxes), len(ss_boxes[ 0 ]), len(ss_boxes[ 0 ][ 0 ]))
SS_BOXES = [ ss_boxes ]

## load rpn
RP_BOXES = [ ]
cls_threshold = 0.0
det_file = "/home/xum/Documents/Git/AlphaNext/AlphaModel/AG_CapitalExclamation/"+\
           "output/default/coco_2017_test/default/res101_faster_rcnn_iter_50000/"+\
           "detections_for_count.pkl"
data1 = pickle.load(open(det_file, 'rb'))  # cls*img*box*[x,y,h,w,s]
for box_num_frame in box_nums_frame:
    rp_boxes = [ ]
    for k in range(image_num):
        prp_boxes = [ ]
        for prp_boxes_cls in data1:
            prp_boxes = prp_boxes + [
                [ prp_box[ 0 ], prp_box[ 1 ], prp_box[ 2 ] - prp_box[ 0 ], prp_box[ 3 ] - prp_box[ 1 ],prp_box[-1] ] for prp_box in
                prp_boxes_cls[ k ] ]
        prp_boxes.sort(key=lambda x: -x[ -1 ])
        rp_boxes.append(prp_boxes[ 1:box_num_frame ])

    print('rp_boxes shap:', len(rp_boxes), len(rp_boxes[ 0 ]), len(rp_boxes[ 0 ][ 0 ]))
    RP_BOXES.append(rp_boxes)

## load eb
EB_BOXES = [ ]
path_prop = "/home/xum/Documents/Git/AlphaNext/AlphaModel/AG_CapitalExclamation/Preprocessing/boxes_eb_result.json"
with open(path_prop) as json_data:
    G_EB_PROP = json.load(json_data)
for box_num_frame in box_nums_frame:
    eb_boxes = [ ]
    for image_id in image_list:
        prp_boxes = G_EB_PROP[ image_id - 1 ][ 'detection_boxes' ]
        # prp_boxes = [ prp_box[ 0:-1 ] for prp_box in prp_boxes ][ 0:box_num_frame ]
        prp_boxes = [ [ prp_box[ 1 ], prp_box[ 0 ], prp_box[ 3 ] - prp_box[ 1 ], prp_box[ 2 ] - prp_box[ 0 ] ] for
                      prp_box in prp_boxes ][ 0:box_num_frame ]
        eb_boxes.append(prp_boxes)
    print('eb_boxes shap:', len(eb_boxes), len(eb_boxes[ 0 ]), len(eb_boxes[ 0 ][ 0 ]))
    EB_BOXES.append(eb_boxes)

## load Jean
jn_boxes = [ ]
path_prop = "/home/xum/Documents/Git/AlphaNext/AlphaModel/AG_CapitalExclamation/Preprocessing/boxes_result.json"
with open(path_prop) as json_data:
    G_BG_PROP = json.load(json_data)
    G_BG_PROP = G_BG_PROP[ 0 ]
for image_id in image_list:
    prp_boxes = [ [ ] ]
    try:
        prp_boxes = G_BG_PROP[ image_id - 1 ][ 'detection_boxes' ]
        prp_boxes = [ [ prp_box[ 0 ], prp_box[ 1 ], prp_box[ 2 ] - prp_box[ 0 ], prp_box[ 3 ] - prp_box[ 1 ] ] for
                      prp_box in prp_boxes ]
    except:
        print("find a invalid box by Jean.")
    finally:
        jn_boxes.append(prp_boxes)
print('jn_boxes shap:', len(jn_boxes), len(jn_boxes[ 0 ]), len(jn_boxes[ 0 ][ 0 ]))
JN_BOXES = [ jn_boxes ]

RPBS_BOXES = [[
                  [box for box in RP_BOXES[0][k] if box[-1]>0.1]+
                  JN_BOXES[0][k]+
                  [box for box in RP_BOXES[0][k] if box[-1]<0.1]
                  for k in range(41)]
              ]
RPEB_BOXES = [[RP_BOXES[0][k]+EB_BOXES[0][k] for k in range(41)]]
BSEB_BOXES = [[JN_BOXES[0][k]+EB_BOXES[0][k] for k in range(41)]]
## merge
proposals = SS_BOXES + RP_BOXES + JN_BOXES + EB_BOXES + RPBS_BOXES + RPEB_BOXES + BSEB_BOXES
print ('total ', len(proposals), 'propsals')

## analyse
''' format: [x,y,w,h] '''
iou_threshold = 0.4
mars = [ ]
# maps = [ ]
axis = range(5, 101)
for box_num in axis:
    recalls = [ ]
    precisions = [ ]
    for id in range(image_num):
        recall = [ ]
        precision = [ ]
        for PRE_boxes in proposals:
            gt_boxes = GT_boxes[ id ][ 0:box_num ]
            pre_boxes = PRE_boxes[ id ][ 0:box_num ]
            overlaps = box_overlaps(pre_boxes, gt_boxes)
            if 0:
                filename = '/home/xum/Desktop/Link to AG_CapitalExclamation/data/coco/images/test2017/COCO_test2017_000000000012.jpg'
                img = mpimg.imread(filename)
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111, aspect='equal')
                imgplot = ax1.imshow(img)
                for box in pre_boxes:
                    p = patches.Rectangle((box[ 0 ], box[ 1 ]), box[ 2 ], box[ 3 ], fill=False,
                                          edgecolor=colorA[ 1 ])
                    ax1.add_patch(p)
                for box in gt_boxes:
                    p = patches.Rectangle((box[ 0 ], box[ 1 ]), box[ 2 ], box[ 3 ], fill=False,
                                          edgecolor='black')
                    ax1.add_patch(p)
                patch = [
                    patches.Patch(color='black', label='Ground Truth'),
                    patches.Patch(color=colorA[ 1 ], label='Proposals')
                ]
                ax1.legend(handles=patch, loc=2)
                # ax1.axis([ 0, 1900, 0, 1080 ])
                plt.show()

            # gt_assignment = overlaps.argmax(axis=1)
            # max_overlaps = overlaps.max(axis=1)
            recall.append(sum(overlaps.max(axis=0) > iou_threshold) / float(len(overlaps.max(axis=0))))
            # precision.append(sum(overlaps.max(axis=1) > iou_threshold) / float(len(overlaps.max(axis=1))))
        recalls.append(recall)
        # precisions.append(precision)
        # print max_overlaps
    # print("precisions",precisions)
    mar = [ sum([ recalls0[ k ] for recalls0 in recalls ]) / float(len(recalls[ : ])) for k in range(len(recall)) ]
    # map = [sum([precisions0[k] for precisions0 in precisions]) / float(len(precisions[:]))  for k in range(len(precision))]
    mars.append(mar)
    # maps.append(map)
    print("average recall " + str(mar))
    # print("average precision " + str(map))

## figure
plt.title('Recall V.S. boxes number (IoU threshold = ' + str(iou_threshold) + ')')
for k in range(len(proposals)):
    lines = plt.plot(axis, [ mars0[ k ] for mars0 in mars ])
    plt.setp(lines, color=colorA[ k ], linewidth=1.0)
#SS_BOXES + RP_BOXES + JN_BOXES + EB_BOXES + RPBS_BOXES + RPEB_BOXES + BSEB_BOXES
patch = [
    patches.Patch(color=colorA[ 0 ], label='Exhaustive Search'),
    patches.Patch(color=colorA[ 1 ], label='RPN'),
    patches.Patch(color=colorA[ 2 ], label='Background Subtraction'),
    patches.Patch(color=colorA[ 3 ], label='Edge Box'),
    patches.Patch(color=colorA[ 4 ], label='RPN + BS'),
    patches.Patch(color=colorA[ 5 ], label='RPN + EB'),
    patches.Patch(color=colorA[ 6 ], label='BS + ES')

]
plt.xlabel('# of boxes', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.legend(handles=patch, loc=1)
plt.show()
