## Data Preparation

### Easy Version

1. Download the annotations files from [[here]](https://drive.google.com/file/d/1izUZrVxA7HMZY2v5iIZSKSMI1XHgECkD/view?usp=sharing). 

2. Extract the file and put the annotations in `HIT/data/jhmdb/annotations`
3. Download the jhmdb videos from their official [website](https://jhmdb.is.tue.mpg.de/), format the `HIT/data/jhmdb/videos` directory as follows:

    ```bash
    {video_name}
        |_ 00001.png
        |_ 00002.png
    ...
    
    ```

### Prepare AVA dataset (Mostly taken from AlphAction) -- unfinished
1. **Download Annotations.** Donwload AVA Actions annotations from the 
[official dataset website](https://research.google.com/ava/download.html).
Organize those annotations file as following structure:

    ```
    AVA/
    |_ annotations/
    |  |_ ava_action_list_v2.2.pbtxt
    |  |_ ava_action_list_v2.2_for_activitynet_2019.pbtxt
    |  |_ ava_include_timestamps_v2.2.txt
    |  |_ ava_train_excluded_timestamps_v2.2.csv
    |  |_ ava_val_excluded_timestamps_v2.2.csv
    |  |_ ava_train_v2.2.csv
    |  |_ ava_val_v2.2.csv
    ```

2. **Download Videos.** Download the list of training/validation file names
from [CVDF repository](https://github.com/cvdfoundation/ava-dataset) and 
download all videos following those links provided there. Place 
the list file and video files as follows:

    ```
    AVA/
    |_ annotations/
    |  |_ ava_file_names_trainval_v2.1.txt
    |_ movies/
    |  |_ trainval/
    |  |  |_ <MOVIE-ID-1>.mp4
    |  |  |_ ...
    |  |  |_ <MOVIE-ID-N>.mp4
    ```
   
3. **Create Symbolic Link.** Create a symbolic link that 
references the AVA dataset directory by running following
commands.
    
    ```shell
    cd /path/to/HIT
    mkdir data
    ln -s /path/to/AVA data/AVA
    ```
   
4. **Preprocess Videos.** Running following commands to 
process raw movies.

    ```shell
    python tools/process_ava_videos.py \
    --movie_root data/AVA/movies/trainval \
    --clip_root data/AVA/clips/trainval \
    --kframe_root data/AVA/keyframes/trainval \
    --process_num $[`nproc`/2]
    ```
   
    This script extracts video clips and key frames from 
    those raw movies. Each video clip lasts exactly one 
    second and ranges from second 895 to second 1805. 
    All video clips are scaled such that the shortest side
    becomes no larger than 360 and transcoded to have fps 25.
    The first frame of each video clip is extracted as key 
    frame, which follows the definition in AVA dataset. 
    (Key frames are only used to detect persons and objects.)
    The output video clips and key frames will be saved as follows:

    ```
    AVA/
    |_ clips/
    |  |_ trainval/
    |  |  |_ <MOVIE-ID-1>
    |  |  |  |_ [895~1805].mp4
    |  |  |_ ...
    |  |  |_ <MOVIE-ID-N>
    |  |  |  |_ [895~1805].mp4
    |_ keyframes/
    |  |_ trainval/
    |  |  |_ <MOVIE-ID-1>
    |  |  |  |_ [895~1805].jpg
    |  |  |_ ...
    |  |  |_ <MOVIE-ID-N>
    |  |  |  |_ [895~1805].jpg
    ```
   
   This processing could take a long time, so we just provide the processed
   key frames and clips for downloading([keyframes](https://drive.google.com/open?id=18Tm-LBUHtkntWZ7rmKllYJz1ZrOqQ3Ez), 
   [clips](https://drive.google.com/open?id=1n2PuZrk3fD6r7gt_h-8CTWvWSYdXjBKu)). 

5. **Convert Annotations.** Our codes use COCO-style anntations, 
so we have to convert official csv annotations into COCO json format
by running following commands.

    ```shell
    python preprocess_data/ava/csv2COCO.py \
    --csv_path data/AVA/annotations/ava_train_v2.2.csv \
    --movie_list data/AVA/annotations/ava_file_names_trainval_v2.1.txt \
    --img_root data/AVA/keyframes/trainval
    python preprocess_data/ava/csv2COCO.py \
    --csv_path data/AVA/annotations/ava_val_v2.2.csv \
    --movie_list data/AVA/annotations/ava_file_names_trainval_v2.1.txt \
    --img_root data/AVA/keyframes/trainval
    ```
   
    The converted json files will be stored in `AVA/annotations` directory
    as follows, `*_min.json` means that the json file has no space indent.
    
    Alternatively, you could just download our json files 
    here([train](https://drive.google.com/open?id=1UaSaMm1b9SBVnBXqlgVOlP4P89TXXvqt), 
    [val](https://drive.google.com/open?id=1iYQKIsVTetjnVgxzP3QmMY3JqYBeO6uK)).
   
    ```
    AVA/
    |_ annotations/
    |  |_ ava_train_v2.2.json
    |  |_ ava_train_v2.2_min.json
    |  |_ ava_val_v2.2.json
    |  |_ ava_val_v2.2_min.json
    ```
   
6. **Detect Persons and Objects.** The predicted person boxes 
for AVA validation set can be donwloaded [[here]](https://drive.google.com/open?id=1hi84yVOWseALM3DadYLrj6ppWFu1dooX).
Note that we only use ground truth person boxes for training.
The object boxes files are also available for download([train](https://drive.google.com/open?id=1E2VKboLSS0vcZRECIcDNjNLIl5sdHmWa), 
[val](https://drive.google.com/open?id=10XorUJzUUyLJZ2h9tFfR8rNif1Itzkcx)).
These files should be placed at following locations.

    ```
    AVA/
    |_ boxes/
    |  |_ ava_val_det_person_bbox.json
    |  |_ ava_train_det_object_bbox.json
    |  |_ ava_val_det_object_bbox.json
    ```

    For person detector, we first trained it on MSCOCO 
    keypoint dataset and then fine-tuned it on AVA dataset. 
    The final model weight is available [[here]](https://drive.google.com/open?id=14eVMRes9bJwn7hN0CVDa_tjaUTfGWoem).
    
    For object detector, we use the model provided in 
    [maskrcnn-benchmark repository](https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth), 
    which is trained on MSCOCO dataset. Person boxes are removed
    from the predicted results.

7. **Detect Keypoints.**
We use [detectron2](https://github.com/facebookresearch/detectron2) as pose detector. Please install the project as indicated in their github repo.

You can use `python preprocess_data/ava/keypoints_detection.py` as a reference to write a script to perform inference on AVA or you can directly download our files [here](https://drive.google.com/drive/folders/1WCtFKnTOpd9jSyUZYXTRJsM_sGrWqweb?usp=share_link) and put them in `data/AVA/annotations/`.

8. **PS**
Some modifications to the project are needed to train on AVA (especially the files `hit/dataset/datasets/jhmdb.py` and `config_files/hitnet.yaml`). Please refer to [Alphaction](https://github.com/MVIG-SJTU/AlphAction) for inspiration on how to change them or open an issue.




