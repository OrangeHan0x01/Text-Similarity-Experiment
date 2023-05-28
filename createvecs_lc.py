from sim_utils import *
from paddlenlp import Taskflow

similarity = Taskflow("text_similarity")
dataset_transform_traintsv(similarity)