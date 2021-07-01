# %% import everything
import torch
import torchvision as tv
from PIL import Image
from matplotlib import pyplot as plt

from modules import *
from utils import *

# %% configure options
options = DEFAULT_OPTIONS

# %% inverse render net
irn = InverseRenderNet(options)

load_checkpoint(irn, 'pretrained/relight_model.npz')

# %% load images from folder 'timeLapse_imgs'
img1 = Image.open('images/1.png')
img2 = Image.open('images/2.png')
mask = Image.open('images/mask.png')

# %% preview the images
plt.title('img1')
plt.imshow(img1)
plt.show()
plt.title('img2')
plt.imshow(img2)
plt.show()
plt.title('mask')
plt.imshow(mask, cmap='gray')
plt.show()

# %% convert PIL images to PyTorch tensors
img1 = tv.transforms.functional.to_tensor(img1)
img2 = tv.transforms.functional.to_tensor(img2)
mask = tv.transforms.functional.to_tensor(mask)

# %% create batch data
img1 = img1.unsqueeze(0)
img2 = img2.unsqueeze(0)
mask = mask.unsqueeze(0)
images = torch.cat([img1, img2], dim=0)
masks = torch.cat([mask, mask], dim=0)

# %% forward into InverseRenderNet
am_out, nm_out, mask_out = irn(images)

# %% post-processing
albedo, normal, shadow, lighting = irn.postprocess(images, masks, am_out,
                                                   nm_out, mask_out)

# %%
lighting
# %%
