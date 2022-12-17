"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "data"
    DATASETS = {
        "jhmdb_train": {
            "video_root": "jhmdb/videos",
            "ann_file": "jhmdb/annotations/jhmdb_train_gt_min.json",
            "box_file": "",
            "eval_file_paths": {
                "labelmap_file": "",
            },
            "object_file": "jhmdb/annotations/train_object_detection.json",
            "keypoints_file": "jhmdb/annotations/jhmdb_train_person_bbox_kpts.json",
        },
        "jhmdb_val": {
            "video_root": "jhmdb/videos",
            "ann_file": "jhmdb/annotations/jhmdb_test_gt_min.json",
            "box_file": "jhmdb/annotations/jhmdb_test_yowo_det_score.json",
            "eval_file_paths": {
                "csv_gt_file": "",
                "labelmap_file": "",
            },
            "object_file": "jhmdb/annotations/test_object_detection.json",
            "keypoints_file": "jhmdb/annotations/jhmdb_test_person_bbox_kpts.json",
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        if attrs["box_file"]=="":
            box_file = ""
        else:
            box_file = os.path.join(data_dir, attrs["box_file"])
        args = dict(
            video_root=os.path.join(data_dir, attrs["video_root"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            box_file=box_file,
            eval_file_paths={key: os.path.join(data_dir, attrs["eval_file_paths"][key]) for key in
                                attrs["eval_file_paths"]},
            object_file=os.path.join(data_dir, attrs["object_file"]),
            keypoints_file=os.path.join(data_dir, attrs["keypoints_file"]),
        )
        return dict(
            factory="DatasetEngine",
            args=args
        )
        raise RuntimeError("Dataset not available: {}".format(name))
