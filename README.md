# cs6910_assignment2
##Part A
# CNN Image Classification with PyTorch Lightning

This code implements a Convolutional Neural Network (CNN) for image classification using PyTorch Lightning. It leverages the iNaturalist 12K dataset to train the model.

## CNN Architecture Features:

- **Five Convolutional Layers:** These layers are responsible for extracting image features.
- **Max Pooling:** Used for dimensionality reduction.
- **Batch Normalization:** Improves training stability by normalizing the inputs of each layer.
- **Dropout:** Prevents overfitting by randomly dropping out connections between neurons.
- **Final Fully Connected Layer:** Employs softmax activation for classification tasks.

## Flexibility and Hyperparameters:

The code offers flexibility by allowing the definition of key settings through command line arguments. These settings, also known as hyperparameters, include:

- **Activation Function Type:** Such as ReLU, LeakyReLU, etc.
- **Batch Normalization Usage:** Can be toggled on or off.
- **Data Augmentation Techniques:** Options include random cropping, flipping, etc.
- **Filter Organization:** Specifies the number of filters and kernel size within the convolutional layers.
- **Dropout Rate:** Controls the dropout probability.



Train and Fine-tune a Convolutional Neural Network with PyTorch Lightning

This code provides two parts:

Part A: Train a CNN from Scratch: This section implements a Convolutional Neural Network (CNN) from the ground up using PyTorch Lightning. It trains the model on the Inaturalist dataset, downloadable from [].
Part B: Fine-tune a Pre-trained Model (ResNet): This section demonstrates fine-tuning a pre-trained ResNet model on the same Inaturalist dataset using PyTorch Lightning again.
Beyond these core functionalities, the code allows you to:

Optimize Hyperparameters: Easily adjust various model settings using the Weights & Biases platform. Simply define the configurations you want to test, and the platform will explore them automatically, logging the results for your analysis.
Customize Model Architecture: Experiment with different CNN architectures by modifying the "Model1" class code. You can freely change layer structures, activation functions, and other architectural elements to find the optimal configuration for your task.
Enhance Data with Augmentation: The "DataModule" class empowers you to explore various data augmentation techniques. This can significantly improve the robustness and accuracy of your model.
WandB report link:
https://wandb.ai/ch22m009/DLASSIGN_2/reports/CS6910-Assignment-2--Vmlldzo3NTA3OTM5
References :
1) https://docs.wandb.ai/tutorials/lightning
2) https://www.kaggle.com/code/shivanandmn/cnn-pytorch-lightning-beginners-model
3) https://aayushmaan1306.medium.com/basics-of-convolutional-neural-networks-using-pytorch-lightning-474033093746
4) https://www.scaler.com/topics/pytorch/build-and-train-an-image-classification-model-with-pytorch-lightning/
5) https://forum.inaturalist.org/t/train-a-neural-network-using-inaturalist-photos-database/30864
