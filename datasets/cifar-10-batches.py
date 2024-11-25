import pickle
import numpy as np
from torchvision import datasets
from PIL import Image
class CIFAR10Dataset:
    def __init__(self, root):
        self.data = []
        self.targets = []
        self.load_data(root)
    def load_data(self, root):
        for i in range(1, 6):
            with open(f'{root}/data_batch_{i}', 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                self.data.append(batch['data'])
                self.targets.extend(batch['labels'])

                self.data = np.concatenate(self.data)
                self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        target = self.targets[index]
        return img, target
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    dataset = CIFAR10Dataset('./cifar-10-batches-py')
    print(f'Number of samples: {len(dataset)}')
    img, label = dataset[0]
    img.show()  # This will display the first image
