import numpy as np
import tempfile
import os
from pprint import pformat
import csv
import time
from collections import defaultdict
from .pascal_evaluation import object_detection_evaluation, standard_fields
import operator


def save_jhmdb_results(dataset, predictions, output_folder, logger):
    logger.info("Preparing results for AVA format")
    ava_results = prepare_for_jhmdb_detection(predictions, dataset)
    logger.info("Evaluating predictions")
    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        if output_folder:
            file_path = os.path.join(output_folder, "result_jhmdb.csv")
        write_csv(ava_results, file_path, logger)
        write_files(ava_results, output_folder, logger)
        return
   

def make_image_key(video_id, timestamp):
    """Returns a unique identifier for a video id & timestamp."""
    return "%s,%04d" % (video_id, int(timestamp))


def decode_image_key(image_key):
    return image_key[:-5], image_key[-4:]

def prepare_for_jhmdb_detection(predictions, dataset):
    ava_results = {}
    score_thresh = dataset.action_thresh
    for video_id, prediction in enumerate(predictions):
        video_info = dataset.get_video_info(video_id)
        if len(prediction) == 0:
            continue
        video_width = video_info["width"]
        video_height = video_info["height"]
        prediction = prediction.resize((video_width, video_height))
        prediction = prediction.convert("xyxy")

        prediction = prediction.to('cpu')
        boxes = prediction.bbox.numpy()

        # No background class.
        scores = prediction.get_field("scores").numpy()
        box_ids, action_ids = np.where(scores >= score_thresh)
        boxes = boxes[box_ids, :]
        scores = scores[box_ids, action_ids]
        action_ids = action_ids + 1

        movie_name = video_info['movie']
        timestamp = video_info['timestamp']

        clip_key = make_image_key(movie_name, timestamp)

        ava_results[clip_key] = {
            "boxes": boxes,
            "scores": scores,
            "action_ids": action_ids
        }
    return ava_results

def testlist_to_dict(base_path):
    testlist_path = os.path.join(base_path, 'testlist.txt')
    with open(testlist_path) as f:
        data = f.readlines()
    data = [i.strip() for i in data]

    dict_data = {}
    for i in data:
        v, k = i.split('/')[:2]
        dict_data[k] = v
        
    return dict_data
    

def write_csv(ava_results, csv_result_file, logger):
    print(csv_result_file)
    dict_data = testlist_to_dict(csv_result_file.split('/')[0] + '/jhmdb/annotations')
    start = time.time()
    with open(csv_result_file, 'w') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=',')
        for clip_key in ava_results:
            movie_name, timestamp = decode_image_key(clip_key)
            cur_result = ava_results[clip_key]
            boxes = cur_result["boxes"]
            scores = cur_result["scores"]
            action_ids = cur_result["action_ids"]
            assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]
            for box, score, action_id in zip(boxes, scores, action_ids):
                box_str = ['{:.5f}'.format(cord) for cord in box]
                score_str = '{:.5f}'.format(score)
                movie_name_with_dir = dict_data[movie_name] + '/' + movie_name
                spamwriter.writerow([movie_name_with_dir, timestamp, ] + box_str + [action_id, score_str])
    print_time(logger, "write file " + csv_result_file, start)

def write_files(ava_results, csv_result_file, logger):
    start = time.time()
    if not os.path.exists(csv_result_file + '/detections'):
            os.mkdir(csv_result_file + '/detections')

    dict_data = testlist_to_dict(csv_result_file.split('/')[0] + '/jhmdb/annotations')

    for clip_key in ava_results:
        movie_name, timestamp = decode_image_key(clip_key)
        filename = dict_data[movie_name] + "_" + movie_name + "_" + timestamp.zfill(5) + ".txt"
        cur_result = ava_results[clip_key]
        boxes = cur_result["boxes"]
        scores = cur_result["scores"]
        action_ids = cur_result["action_ids"]
        assert boxes.shape[0] == scores.shape[0] == action_ids.shape[0]

        detection_path = os.path.join(csv_result_file + '/detections', filename)
        
        
        with open(detection_path, 'w+') as f_detect:
            for box, score, action_id in zip(boxes, scores, action_ids):
                if float(score) > 0:
                    box_str = [str(round(cord)) for cord in box]
                    score_str = '{:.5f}'.format(score)
                    f_detect.write(str(int(action_id)) + ' ' + score_str + ' ' + box_str[0] + ' ' + box_str[1] + 
                        ' ' + box_str[2] + ' ' + box_str[3] + '\n')

                
def print_time(logger, message, start):
    logger.info("==> %g seconds to %s", time.time() - start, message)


