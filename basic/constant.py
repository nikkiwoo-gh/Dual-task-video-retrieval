import os
import logging


#ROOT_PATH=os.path.join(os.environ['HOME'], 'VisualSearch')
ROOT_PATH = '/vireo00/nikki/AVS_data'
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)
# logger.setLevel(logging.INFO)

