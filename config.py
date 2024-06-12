DEVICE = '0' # For GPU set device num which you want to use (or set 'cpu', but it's too slow)

# Our confidence for every person (bbox)
PERSON_CONF = 0.5

KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

# Our confidence for used keypoints
KPTS_CONF = 0.6

IOU_THRESHOLD = 0.1

# It's xMem original config, you can try to change this values for your task (check xMem article)
XMEM_CONFIG = {
    'top_k': 80,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 250,
    'min_mid_term_frames': 7,
    'max_mid_term_frames': 30,
    'max_long_term_elements': 10000,
}

# Max possible count of persons in video (if you has error, set bigger number)
MAX_OBJECT_CNT = 20

# Check new persons in frame every N frames
YOLO_EVERY = 40
# INFERENCE_SIZE = (360, 288)
# INFERENCE_SIZE = (516, 388)
# INFERENCE_SIZE = (1032, 776)
# INFERENCE_SIZE = (640, 360)
INFERENCE_SIZE = (960, 540)
# INFERENCE_SIZE = (480, 270)
# INFERENCE_SIZE = (1280, 720)
# INFERENCE_SIZE = (1920, 1080)

