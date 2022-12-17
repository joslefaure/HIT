import av
from PIL import Image
import numpy as np

def av_decode_video(video_path):
    with av.open(video_path) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_rgb().to_ndarray())
    return frames

def cv2_decode_video(video_path):
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_rgb().to_ndarray())
    return frames


def image_decode(video_path):
    frames = []
    try:
        with Image.open(video_path) as img:
            frames.append(np.array(img.convert('RGB')))
    except BaseException as e:
        raise RuntimeError('Caught "{}" when loading {}'.format(str(e), video_path))
    
    return frames