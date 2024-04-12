# cs6910_assignment2
This is an implementation of Convolutional Neural networks using Pytorch lightining and Fine Tuning.
The PartA of the code corresponds to training the CNN on the inaturalist data set which can be found on:
https://storage.googleapis.com/wandb_datasets/nature_12K.zip
For Fine-tuning your Deep Learning Model(PartB):
Again Pytorch lightining is used to fine tune ResNet on the inaturalist dataset
With the code you can furthermore:
Optimizing Hyperparameters: The Weights & Biases platform can help you streamline hyperparameter tuning. You set up the configurations you want to test, and the platform automatically explores those possibilities, recording the results.
Adapting the Model Architecture: You can experiment with different configurations for your convolutional neural network (CNN) by making changes to the code in the "Model1" class. This lets you try out various layer structures, activation functions, and other architectural elements.
Enhancing Data with Augmentation: The "DataModule" class allows you to test out different data augmentation techniques. This can improve the strength and accuracy of your model.




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
