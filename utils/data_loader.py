import os
import torch
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np
import PIL
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(data.Dataset):
    def __init__(self, datadir, transforms, mode='train'):
        self.dataset = datadir
        self.transforms = transforms
        self.mode = mode
        if self.mode == 'train':
            self.images = self.find_files('training')
        elif self.mode == 'test':
            self.images = self.find_files('testing')
        elif self.mode == 'valid':
            self.images = self.find_files('valid')
        
    def find_files(self, mode):
        filenames = []
        for (path, _, files) in os.walk(self.dataset+mode+'/input'):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext.lower() in ['.jpg', '.jpeg', '.png', '.ppm']:
                    filenames.append(f'{path}/{filename}')
        return filenames
       
    def __getitem__(self, index):
        
        def get_crop_and_rotation(image, size=512):       
            np.random.seed(int(str(time.time())[11:]))
            width, height = image.size 
            left = np.random.randint(width - size)
            top = np.random.randint(height - size)
            right = left + size
            bottom = top + size
            angle = np.random.randint(360)
            aug_params = left, top, right, bottom, angle
            return aug_params

        def crop_and_rotate_image(image, aug_params):
            left, top, right, bottom, angle = aug_params
            image = image.rotate(angle, PIL.Image.NEAREST, expand = 0) 
            return image.crop((left, top, right, bottom))
        
        def load_tensor(input_name, output_name, transform=self.transforms):            
            input_image  = Image.open(input_name)
            output_image = Image.open(output_name)
            aug_params = get_crop_and_rotation(output_image)
            input_image  = crop_and_rotate_image(input_image, aug_params)
            output_image = crop_and_rotate_image(output_image, aug_params)
            return transform(input_image), transform(output_image)

        def indexerror(index, dataset):
            while index >= len(dataset):
                index -= len(dataset)
            return index
        
        index = indexerror(index, self.images)
        
        input_name  = self.images[index]
        output_name = input_name.replace('/input/', '/output/')
        if input_name.endswith('.jpg'):
            output_name = output_name.replace('.jpg', '.png')
        
        input_tensor, output_tensor = load_tensor(input_name, output_name, self.transforms)     
        
        return input_tensor, output_tensor
    
    def __len__(self):
        return len(self.images)

def get_loader(image_dir='./data/', batch_size=1, num_workers=1, mode='train'):
    
    def get_transform(centercrop=512):        
        transform = []
        transform.append(T.CenterCrop(centercrop))
        transform.append(T.ToTensor())
        return T.Compose(transform)

    transforms = get_transform()
    
    shuffle = (mode == 'train')
    
    data_loader = torch.utils.data.DataLoader(dataset=Dataset(image_dir, transforms, mode),
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    
    """
    ####### How to Use #######
    
    data_loader = get_loader(mode='train')
    data_iter = iter(data_loader)
    
    ...
    
    try:
        input_tensor, output_tensor = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        input_tensor, output_tensor = next(data_iter)
        
        
    or
    
    data_loader = get_loader(mode='train')
    for epoch in range(num_epochs):
        for i, (input_tensor, output_tensor) in enumerate(data_loader):
            ...
    
    """
    
    return data_loader
