import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, models, transforms

from captum.attr import IntegratedGradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#name of the network that will be used in interpretation
model = torch.load('resnet_network_rgb')
model.to(device)
model.eval()

#normalize validation images
transform = transforms.Compose([
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ])

class ImageFolderWithPaths(datasets.ImageFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

#location of validation data       
data_dir = 'rgb/val/'

image_dataset = ImageFolderWithPaths(root=data_dir, transform=transform)
testloader = torch.utils.data.DataLoader(image_dataset,batch_size=198,shuffle=False, num_workers=0)

for inputs,labels, paths in testloader:
    inputs=inputs.to(device)
    labels=labels.to(device)
    print(labels, paths)

classes = ('high','low')

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(198)))

outputs = model(inputs)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(198)))

ind = 0
for ind in range(0,198):
    
    input = inputs[ind].unsqueeze(0).to(device)
    input.requires_grad = True
    
    def attribute_image_features(algorithm, input, **kwargs):
        model.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                  target=labels[ind],
                                                  **kwargs
                                                 )
        return tensor_attributions
    
    ig = IntegratedGradients(model)
    attr_ig, delta = attribute_image_features(ig, input, baselines=input * 0, return_convergence_delta=True)
    attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    print('Approximation delta: ', abs(delta))

    print('Original Image')
    print('Predicted:', classes[predicted[ind]], 
          ' Probability:', torch.max(F.softmax(outputs, 1)).item())
    
    original_image = np.transpose((inputs[ind].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    
    _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",show_colorbar=True, title="Overlayed Integrated Gradients")
     
    plt.savefig('Figures_rgb/fig'+str(ind)+'.png',bbox_inches='tight',pad_inches=0)
        
    ind = ind +1
