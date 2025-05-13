# Deep Learning – Summer 2024 (NYCU)

This repository contains assignments, projects, and code implementations for the **Deep Learning** course offered at **National Yang Ming Chiao Tung University (NYCU)** during **Summer 2024**.


### Lab1:
- Only use Numpy and the python standard libraries to implement simple neural networks with forwarding pass and backpropagation.


### Lab2:
- Predict motor imagery task by training SCCNet model using deep learning techniques.

### Lab3:

- I design the ResNet, UNet architecture to recognize the animal picture.
And also preprocess the dataset like image flipping and image scaling.
- Finally use Dice score to evaluate model’s performance.

### Lab4:
Implemented a conditional VAE-based video prediction model with KL annealing and teacher forcing strategies, achieving strong reconstruction quality with a PSNR of over 29.

### Lab5:

- Implement MaskGIT for the inpainting task.
- Image contain gray regions indicating missing information, which we aim to restore using MaskGIT.
- **Focus on:** Multi-head attention, transformer training, and inference inpainting.

### Lab6:
I implemented a conditional DDPM using UNet2DModel and DDPMScheduler from Hugging Face, instead of directly using the default DDPMPipeline.

And this Conditional DDPM can generate images based on given one-hot encoded class labels.
And train the model to learn the denoising process (i.e how to recover clean images from noisy ones.)
During testing, generate new images by starting from pure noise and denoising step-by-step using the learned model, guided by the input labels.
Finally, the conditional DDPM achieved **an accuracy of 0.8594**, indicating correct object composition in the generated images.
