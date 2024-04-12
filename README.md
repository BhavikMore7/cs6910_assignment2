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

### Dataset and Data Loaders

- **iNaturalist 12K Dataset:** The dataset is sourced from the `/kaggle/input/inaturalist12k/Data/inaturalist_12K/train` and `/kaggle/input/inaturalist12k/Data/inaturalist_12K/val` directories for training and testing, respectively (if utilizing Kaggle).
- **Data Augmentation Parameter:** Depending on the value of the `data_augmentation` parameter in the project configuration, either `transform` or `transform_augmented` is applied to the training set.
- **Testing Set:** The testing set always employs the `transform` function.
The dataset is split into training and validation sets using a ratio of 80:20. Data loader objects are created for both sets, with a batch size of 64 for both and the training set being shuffled.

### Data Transformations

- **Two Sets of Transformations:** Two sets of data transformations are specified: `transform` and `transform_augmented`.
- **Common Transformations:** Both transformations resize the images to 256x256 pixels and convert them into tensors.
- **Augmented Transformation:** `transform_augmented` incorporates additional data augmentation techniques such as random cropping, flipping, and rotating to enhance variation in the training data.
- **Normalization:** Both transformations normalize the images using mean and standard deviation values derived from the ImageNet dataset.




### Train and Fine-tune a Convolutional Neural Network with PyTorch Lightning

This code offers two main functionalities:

#### Part A: Train a CNN from Scratch
- **Implementation:** This section constructs a Convolutional Neural Network (CNN) from scratch using PyTorch Lightning.
- **Dataset:** It trains the model on the iNaturalist dataset, which is downloadable from [link].
  
#### Part B: Fine-tune a Pre-trained Model (ResNet)
- **Fine-tuning:** Demonstrates the process of fine-tuning a pre-trained ResNet model on the same iNaturalist dataset using PyTorch Lightning.
  
#### Additional Features:
- **Hyperparameter Optimization:** Easily adjust various model settings using the Weights & Biases platform. Define configurations to test, and the platform automatically explores them, logging results for analysis.
- **Customizable Architecture:** Experiment with different CNN architectures by modifying the "Model1" class code. You can freely alter layer structures, activation functions, and other architectural elements to find the best configuration for your task.
- **Data Augmentation:** The "DataModule" class allows exploration of various data augmentation techniques, enhancing the model's robustness and accuracy.

WandB report link:
https://wandb.ai/ch22m009/DLASSIGN_2/reports/CS6910-Assignment-2--Vmlldzo3NTA3OTM5
References :
1) https://docs.wandb.ai/tutorials/lightning
2) https://www.kaggle.com/code/shivanandmn/cnn-pytorch-lightning-beginners-model
3) https://aayushmaan1306.medium.com/basics-of-convolutional-neural-networks-using-pytorch-lightning-474033093746
4) https://www.scaler.com/topics/pytorch/build-and-train-an-image-classification-model-with-pytorch-lightning/
5) https://forum.inaturalist.org/t/train-a-neural-network-using-inaturalist-photos-database/30864
