from .stl10 import STL10_boxes
from .tiny200 import Tiny200_boxes
from .imagenet_subset import ImageFolderSubset, ImageFolderSubsetCCrop
from .imagenet import ImageFolderCCrop
from torchvision.datasets import ImageFolder

from .BDD100K import BDD100KCCrop
from .build import build_dataset, build_dataset_ccrop


