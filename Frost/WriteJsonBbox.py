from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import shutil

def ReadBoxFile(txtPath):
    # read file by line
    annTxt =  np.loadtxt(txtPath,delimiter=',')
    boxes = annTxt[:,0:-1]
    classes = annTxt[:,-1]
    print(boxes.shape,classes.shape)
    return boxes,classes

def WriteCOCOFile(bfile, filePath):
    # info
    info = {
       'contributor': "Frost M. Xu",
       'date_created':"2017-04-27",
       'description':"This is stable 1.0 version of the 2017 Zhong Xing Peng Yue data set.",
       'url':"https://opt.zju.edu.cn",
       'version':"1.0",
       'year':2017,
    }
    # licenses
    licenses = "not yet"
    # images move to anno
    images = []

    # categories
    category1 = {'id':0, 'name':'NotSure'}
    category2 = {'id':1, 'name':'Male'}
    category3 = {'id':2, 'name':'Female'}
    categories = [category1,category2,category3]
    cath1 = {'id': 0, 'name': " {'N.A.','N.A','NA'}"}
    cath2 = {'id': 1, 'name': "BLACK"}
    cath3 = {'id': 2, 'name': "BLUE"}
    cath4 = {'id': 3, 'name': "BROWN"}
    cath5 = {'id': 4, 'name': "GREEN"}
    cath6 = {'id': 5, 'name': "ORANGE"}
    cath7 = {'id': 6, 'name': "{'RED','ROSE'}"}
    cath8 = {'id': 7, 'name': "{'GREY','GRAY','SILVER'}"}
    cath9 = {'id': 8, 'name': "WHITE"}
    cath10 = {'id': 9, 'name': " {'YELLOW','TAN','GOLD','GOLDEN'}"}
    cath11 = {'id': 10, 'name': "TRANSPARENT"}
    cath12 = {'id': 11, 'name': "MULTI"}
    cath13 = {'id': 12, 'name': "UNKNOWN"}
    hat_categories = [cath1,cath2,cath3,cath4,cath5,cath6,cath7,cath8,cath9,cath10,cath11,cath12,cath13]
    glass_categories = [cath1,cath2,cath3,cath4,cath5,cath6,cath7,cath8,cath9,cath10,cath11,cath12,cath13]
    mask_categories = [cath1,cath2,cath3,cath4,cath5,cath6,cath7,cath8,cath9,cath10,cath11,cath12,cath13]

    # annotations
    annotations = []
    imageIdx = 0
    for line in bfile:
        string = line.split('"')
        clses = [int(string[0].split()[k]) for k in range(1,5)]
        cls = string[0].split()[1]
        names = [line.split('"')[k] for k in range(1, len(line.split('"')), 2)]
        bboxes = [line.split('"')[k] for k in range(2, len(line.split('"')), 2)]
        assert len(names) == len(bboxes)
        for k in range(len(names)):
            oldName = names[k]
            name = 'COCO_alpha2017_' +str(imageIdx).zfill(12) + '.png'
            newPath = '/media/xum/New Volume/data/WIDER FACE/cocoStyle/'+name
            shutil.copy('/media/xum/New Volume/data/WIDER FACE/'+oldName, newPath)
            box = [float(bbox) for bbox in bboxes[k].split()]
            bbox = [box[1], box[0], box[3], box[2]]
            if not ( box[0]>=0 and box[1] >= 0 and box[3]<=112 and box[2]<96):
                print('fingd one error data')
                print(bbox)
                continue

            image = {'date_captured': '2017-05-09', 'file_name': name, 'height': 112,
                     'id': imageIdx, 'license': 'None', 'url': 'None', 'width': 96, }
            images.append(image)
            annotation = {'bbox': bbox, 'category_id': int(cls), 'image_id': imageIdx,
                          'categpries_id':clses,
                          'id': imageIdx, 'area': bbox[2] * bbox[3],  'iscrowd': 0}
            imageIdx = imageIdx+1
            annotations.append(annotation)

    results = {
        'annotations':annotations,
        'categories':categories,
        'hat_categories':hat_categories,
        'glasses_categories':glass_categories,
        'maskcategories':mask_categories,
        'images':images,
        'info':info,
        'licenses':licenses,
        'type':'instances'
    }
    with open(filePath, 'w') as fid:
        json.dump(results, fid)
    print('Writing results json to {}'.format(filePath))
    return results

file = open("FaceResult.txt")
fakeDataset = WriteCOCOFile(file, "instances_alpha2017.json")
# print(fakeDataset)
import pprint
pprint.pprint(len(fakeDataset['annotations']))
print("result is in file: FaceBbox.json")
