import pprint, pickle
import numpy as np
det_file = "/home/xum/Documents/Git/AlphaNext/AlphaModel/AG_CapitalExclamation/output/default/coco_2017_train/default/res101_faster_rcnn_iter_50000/detections_for_count.pkl"


pkl_file = open(det_file, 'rb')

data1 = pickle.load(pkl_file) # cls*img*box*[x,y,h,w,s]

#GT_COUNT = [1, 1, 2, 3, 1, 1, 2, 1, 1, 1, 1, 4, 1, 2, 3, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 3, 5, 6, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 1, 3, 5, 1, 1, 1, 2, 3, 4, 5, 1, 1, 2, 3, 1, 4, 1, 1, 5, 1, 2, 3, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 4, 6, 4, 1, 2, 3, 1, 1, 3, 4, 1, 2, 7, 14, 3, 3, 12, 5, 1, 1, 1, 3, 1, 2, 4, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 3, 4, 3, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 3, 3, 1, 2, 2, 1, 2, 1, 1, 3, 1, 1, 3, 2, 3, 6, 7, 1, 2, 3, 1, 4, 5, 1, 2, 6, 7, 3, 3, 1, 3, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 3, 3, 1, 2, 4, 1, 1, 1, 2, 4, 4, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 3, 4, 1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 2, 3, 3, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2]
cout_file = "/home/xum/Documents/Git/AlphaNext/AlphaModel/AG_CapitalExclamation/Frost/instances_train2017_count.txt"
GT_COUNTS = pickle.load(open(cout_file, "r"))
GT_COUNT = GT_COUNTS[0,:]

xaxis = np.linspace(0.5,1,100)
Res = []
Counts = []
for thresh in xaxis:
  counts = np.zeros((len(data1), len(data1[0])))
  print('Calculate '+ str(thresh) +'...' )
  for k in range(len(data1[0])):
    for l in range(1,len(data1)):
        data = data1[l][k]
        scores = data[:,4]
        counts[l,k] = sum(scores>thresh)
  for k in range(len(data1[0])):
    counts[0, k] = sum(counts[:, k])
  #print(counts[0,:])
  #print(GT_COUNT)
  ## GT_COUNT[0] is background
  diff2 = map(lambda x,y: (x-y)**2, counts,GT_COUNTS[:,1:])

  # assert len(counts[0])==len(GT_COUNT)-1
  res = [np.sqrt(sum(v)/len(v)) for v in diff2]
  print res
  Res.append(res)
  Counts.append(sum(counts[0, :]))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

f, axarr = plt.subplots(2)
plt.suptitle('Counting Result v.s. Threshold', fontsize=20)
scope = 1
for k in range(4):
    lines = axarr[1].plot(xaxis, [res[k]*scope for res in Res])
    plt.setp(lines, color=color8[k], linewidth=2.0)
axarr[1].set_ylabel('Counting Error', fontsize=16)

lines = axarr[0].plot(xaxis, Counts)
plt.setp(lines, color=color8[6], linewidth=2.0)
lines =  axarr[0].plot(xaxis, [sum(GT_COUNT)]*len(xaxis))
plt.setp(lines, color=color8[7], linewidth=2.0)
maxy = [5,sum(GT_COUNT) * 2]
for k in range(2):
    axarr[k].axis([min(xaxis), max(xaxis), 0, maxy[1-k]])

plt.xlabel('Threshold', fontsize=16)

axarr[0].set_ylabel('Counting Result', fontsize=16)
patch = []
for k in range(4):
    patch.append( patches.Patch(color=color8[k], label='error (abusolute) for class '+ str(k)))#+', '+str(scope)+'X') )
axarr[1].legend(handles=patch)
patch = []
patch.append( patches.Patch(color=color8[6], label='total number (GT)') )
patch.append( patches.Patch(color=color8[7], label='total number (Predict)') )
axarr[0].legend(handles=patch)

plt.show()