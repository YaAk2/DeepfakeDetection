import pandas as pd
import torch
from torchvision import transforms, datasets
import torch.utils.data as data


class DFirt(data.Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = datasets.ImageFolder(path, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
        self.N = len(self.data)
        self.classes = self.data.classes
        self.num_classes = len(self.classes)
        
    def oversampling(self, batch_size=128): 
        counts = []
        df = pd.DataFrame(self.data.samples, columns=['Path', 'Label'])
        [counts.append(df['Label'].value_counts()[n]) for n in range(self.num_classes)]
        weights = 1./torch.Tensor(counts/max(counts))
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, self.N)
        pass    
        
    def class_distribution(self):
        df = pd.DataFrame(self.data.samples, columns=['Path', 'Label'])
        [print(self.classes[n], ': ', df['Label'].value_counts()[n]) for n in range(self.num_classes)]
    
    def data_loader(self, batch_size=128):
        data = torch.utils.data.DataLoader(self.data, batch_size=batch_size, shuffle=True, num_workers=0)
        return data