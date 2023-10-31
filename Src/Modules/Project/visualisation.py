import torch
import numpy as np
from matplotlib import pyplot as plt

def display_image_samples(img_dataset: torch.utils.data.dataset.Dataset, r_seed: int = None) -> None:
    """
    Function to visualise a random sample of 12 images in the form of a matplotlib plot. It is meant to quickly assess image data is being loaded correctly.
    Params:
        img_dataset: A PyTorch Dataset class that returns an Image as PyTorch Tensor and a Tensor containing the Image Label
        r_seed: A integer used to ensure random state replicability
    Returns:
        None.
    """

    # Class labels
    _class_lbl = {0: 'Control', 1: 'Stroke'}
    
    # Initialise random seed
    if r_seed:
        np.random.seed(r_seed)

    # Randomly choose 12 images to display
    _img_idxs = np.random.choice(range(len(img_dataset)), size = 12, replace = False)

    # Initialise plot
    fig, axes = plt.subplots(4,3, figsize = (15,12.5))

    fig.suptitle("Sample of MRI Images\nwith class labels", size = 18)

    # Fill plot
    for _idx, ax in zip(_img_idxs, axes.ravel()):
        ## Fetch image and class
        _img, _class = img_dataset[_idx]
        
        ## Prepare image so it is compatible with matploblib pyplot
        _img_proc = _img.cpu().view(-1, 224, 224).permute(1,2,0)
        
        ## Populate plot
        ax.imshow(_img_proc, cmap = 'gray', )
        ax.axis('off')
        ax.grid('off')
        ax.set_title(f'Label: {_class_lbl.get(int(_class.cpu().numpy()))}')
    plt.show()