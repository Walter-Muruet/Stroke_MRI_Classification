from torch import tensor
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL  import Image
from typing import List, Tuple

# Custom class to load image files. Inherits from the torch primitive Dataset
class StrokeImageData(Dataset):
    def __init__(self, img_fps:List[str], device:str, recipe:transforms.Compose = None) -> None:
        self._rawimages = img_fps
        self._class_encoder = {'Control':0, 'Acute Ischemic Stroke':1}
        self._device = device
        if recipe:
            self._transforms = recipe
        else:
            self._transforms = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self._rawimages)
    
    def __getitem__(self, ix:int) -> Tuple[tensor, tensor]:
        _file = self._rawimages[ix]
        _image = self._load_image(self._rawimages[ix])
        _class = self._class_encoder.get(_file.split('/')[-2])
        
        #return self._transforms(_image).to(self._device)[:1,:,:], tensor(_class).to(self._device)
        return self._transforms(_image).to(self._device), tensor(_class).to(self._device)
    
    def _load_image(self, file_path:str) -> Image.Image:
        """
        A function to read image files (jpg or png)
        Params:
            file_path: A string containing path/to/file
        Returns:
            An Image.Image object
        """
        return Image.open(file_path).convert('RGB') #Ensures png files have only three channels (RGB) and not four (RGBA)