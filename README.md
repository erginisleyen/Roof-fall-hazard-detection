# Roof-fall-hazard-detection

The code for manuscript titled "Roof fall hazard detection with convolutional neural networks using transfer learning" by Ergin Isleyen

This repository is a part of manuscript submission to Computers & Geosciences Journal.

///"resnet.py"///

Trains a ResNet CNN for roof fall hazard detection. Images should be separated into "train" and "val" folders prior to using this script. Also, inside these folders, images should be placed under a folder titled with their actual class, e.g. "hazard" and "non-hazard".
It uses a transfer learning approach with a network trained on ImageNet dataset.
The script uses PyTorch library.

Default batch size is 16, and the default number of epochs is 25. 

It calculates and prints the confusion matrix. 

///deep learning interpretation.py//
Developed with captum library.
Uses the network train with the "resnet.py" script.
Uses integrated gradients technique.
Saves the results in image format (Blended heat map) to specified directory.

