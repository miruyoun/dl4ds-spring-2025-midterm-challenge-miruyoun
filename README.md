[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xnB1OI0j)
# DS542 Deep Learning for Data Science -- Spring 2025 Midterm Challenge

## AI Disclosure

This submission was created with the assistance of ChatGPT.

**AI Contributions:**
- Provided guidance on architecture design and PyTorch module usage.
- Advised on how to implement features such as early stopping and LR scheduler.
- Provided assistance on how CNN and ResNet work.
- Helped ensure the code aligned with assignment rubric requirements.

**My Contributions:**
- Wrote the convolutional layers and implemented the forward pass.
- Modified and finalized hyperparameter values.
- Modified and applied the data augmentation.



## Model Description

**Simple CNN:** A custom convolutional neural network from scratch using PyTorch.  
- Three convolutional layers with increasing filter counts (32, 64, 128).  
- Each convolutional layer is followed by ReLU activation and 2x2 max pooling.  
- After the final convolution, the feature map is flattened and passed through two fully connected layers.  

**Predefined ResNet Model:** A modified ResNet34 architecture. The model was modified to adapt to CIFAR-100's smaller input resolution and classification complexity  
- Replaced the first convolutional layer with a 3x3 kernel, stride 1, and padding 1.  
- Replaced max pool with adaptive average pooling.  
- Replaced the fully connected layer with a custom head  
      o Linear → ReLU → Dropout(0.3)  
      o Linear → ReLU → Dropout(0.3)  
      o Final Linear output layer with 100 output classes  

**Predefined ResNet Model with Pretrained Weights:**  
- Same architecture was kept however, I introduced default training weights.



## Hyperparameter Tuning

**Simple CNN:** Values were  
- Learning rate: 0.1  
- Batch size: 512 (determined empirically based on system memory)  
- Epoch: 5  

**Predefined ResNet Model:**  
- Learning rate: 0.1  
- Batch size: 128 (For speed and stability)  
- Epoch: 10  

**Predefined ResNet Model with Pretrained Weights:**  
- Learning rate: 0.001  
- Batch size: 128  
- Epoch: 50  



## Regularization Techniques

**Simple CNN:**  
- Weight decay was added to the optimizer.  
- Cosine Annealing scheduler helped lower LR over time.  

**Predefined ResNet Model:**  
- Same as Simple CNN however a dropout layer was added to improve generalization  

**Predefined ResNet Model with Pretrained Weights:**  
- Same as Predefined ResNet Model  



## Data Augmentation Strategy

**SimpleCNN:** Only minimal augmentation was applied:  
- ToTensor(), Convert images to PyTorch tensors  
- Normalize(mean=0.5, std=0.5) Centers pixel values  

**Predefined ResNet Model:**  
- RandomCrop(32, padding=4), Simulate zoom and translation  
- RandomHorizontalFlip(p=0.5), Horizontally flip images  
- RandomRotation(20), Random rotation  
- ToTensor(), Convert images to PyTorch tensors  
- Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), New normalization  

**Predefined ResNet Model with Pretrained Weights**  
- Same as Predefined ResNet Model  



## Results Analysis

The Simple CNN and predefined ResNet models were relatively fast to train and easy to tune; however, that came at the cost of having poor accuracy in both training, validation, and test datasets. With the predefined model and pretrained weights, I was able to beat the baseline however, with only about mid 40% accuracy on test datasets. Potential areas of improvements could be trying deeper models, better data augmentations and training on more epochs.



## Experiment Tracking Summary

All training and validation metrics were logged using Weights & Biases (wandb), including:  
- Training loss and accuracy  
- Validation loss and accuracy  
- Learning rate schedule

[Link to WANDB website](https://api.wandb.ai/links/miruyoun-boston-university/gpqtkp5g)
