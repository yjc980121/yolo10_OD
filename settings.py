from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
SOURCES_LIST = [IMAGE, VIDEO]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'image_1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'image_1_output.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'

VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH
}

def attempt_download_from_hub(repo_id, hf_token=None):
    # https://github.com/fcakyon/yolov5-pip/blob/main/yolov5/utils/downloads.py
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils._errors import RepositoryNotFoundError
    from huggingface_hub.utils._validators import HFValidationError
    try:
        repo_files = list_repo_files(repo_id=repo_id, repo_type='model', token=hf_token)
        model_file = [f for f in repo_files if f.endswith('.pt')][0]
        file = hf_hub_download(
            repo_id=repo_id,
            filename=model_file,
            repo_type='model',
            token=hf_token,
        )
        return file
    except (RepositoryNotFoundError, HFValidationError):
        return None

# ML Model config
#MODEL_DIR = ROOT / 'weights'
#DETECTION_MODEL = MODEL_DIR / 'yolov10x.pt'
MODEL_PATH = attempt_download_from_hub("kadirnar/yolov10m", hf_token="hf_token")
DETECTION_MODEL = MODEL_PATH