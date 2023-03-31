import csv
import numpy as np
import pickle
import math
from tqdm import tqdm
import sys

# Usage: python make_video_detection.py path/to/csv path/to/save/pkl
# label dict
dict_label = {}
for i in range(22): # we use 1-22
    dict_label[i] = []

print(dict_label.keys())

input_csv, output_pkl = sys.argv[1], sys.argv[2]
with open(input_csv, newline='') as csvfile:
    rows = csv.reader(csvfile)
    print("in progress...")
    for row in tqdm(rows):
        video_name = row[0]
        video_type = video_name.split('/')[0] # video sport type
        frame_number = float(row[1])
        x1 = round(float(row[2]))
        y1 = round(float(row[3]))
        x2 = round(float(row[4]))
        y2 = round(float(row[5]))
        class_id = int(row[6]) - 1 # suppose to -1
        score = float(row[7])

        existed = False

        video_same_name = []

        for tube in list(dict_label[class_id]):
            # print(video_name)
            if video_name in tube:
                if tube[2][-1][0] +1 == frame_number: #frame number supposed to be continuous

                    x = list(tube)
                
                    existed = True

                    frame_cnt = 0
                    score_cnt = 0

                    for box in x[2]:
                        frame_cnt += 1
                        score_cnt += box[-1]
                    
                    #update frame_cnt and score_cnt
                    frame_cnt += 1
                    score_cnt += score

                    x[1] = score_cnt / frame_cnt

                    #update tube_boxes
                    x[2].append([frame_number,x1,y1,x2,y2,score])
                    # print(x[0], x[1])
                
                    dict_label[class_id].remove(tube)
                    dict_label[class_id].append(tuple(x))

                    break

        #if video name not exist
        if not existed:
            tube_v = video_name
            tube_score = score
            tube_boxes = []
            tube_boxes.append([frame_number,x1,y1,x2,y2,score])
            tube = (tube_v,tube_score,tube_boxes)
            dict_label[class_id].append(tube)

# change tube_boxes to numpy array
for i in range(22): # we use 1-22
    for tube in list(dict_label[i]):
        x = list(tube)
        x[2] = np.array(x[2],dtype = np.float32)
        dict_label[i].append(tuple(x))
        dict_label[i].remove(tube)

with open(output_pkl, 'wb') as handle:
    pickle.dump(dict_label, handle, protocol=2)

print("Done")
