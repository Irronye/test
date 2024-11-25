import torch
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset

class BDD100KCCrop(Dataset):
    def __init__(self, root, transform_rcrop, transform_ccrop, init_box=[0., 0., 1., 1.], **kwargs):
        super().__init__(root=root, **kwargs)
        self.samples = [os.path.join(root, fname) for fname in os.listdir(root) 
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))]
        if not self.samples:
            raise FileNotFoundError(f"No valid images found in {root}")
    
        self.root = root
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop
        self.boxes = torch.tensor(init_box).repeat(self.__len__(), 1)
        self.use_box = True
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.use_box:
            box = self.boxes[index].float().tolist()  # box=[h_min, w_min, h_max, w_max]
            sample = self.transform_ccrop([sample, box])
        else:
            sample = self.transform_rcrop(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
