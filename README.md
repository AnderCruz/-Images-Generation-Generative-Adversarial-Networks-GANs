# Images Generation with GANs

This project implements a Generative Adversarial Network (GAN) to generate fashion images using the Fashion MNIST dataset. The notebook contains a complete implementation of both generator and discriminator models, training loop, and visualization of generated images.

## Project Overview

The project demonstrates:
- Loading and preprocessing the Fashion MNIST dataset
- Building a DCGAN (Deep Convolutional Generative Adversarial Network) architecture
- Training the GAN with proper loss functions and optimizers
- Generating and visualizing synthetic fashion images
- Saving model checkpoints during training

## Model Architecture

### Generator
The generator uses a deep convolutional architecture that transforms random noise into 28x28 fashion images:
- Input: 100-dimensional noise vector
- Dense layer → Reshape to 7x7x256
- Transposed convolution layers with batch normalization and LeakyReLU
- Output: 28x28x1 image with tanh activation

### Discriminator
The discriminator classifies images as real or fake:
- Input: 28x28x1 grayscale image
- Convolutional layers with LeakyReLU and dropout
- Flatten → Dense output layer
- Output: Binary classification (real/fake)

## Usage

### Prerequisites
- TensorFlow 2.x
- Matplotlib
- TensorFlow Datasets
- Jupyter Notebook

### Training
```python
# Run the training loop
train(train_ds, epochs=50)
```

The training process:
1. Alternates between training generator and discriminator
2. Saves checkpoints every 15 epochs
3. Generates sample images throughout training
4. Displays progress and timing information

### Generating Images
```python
# Create noise vector
noise = tf.random.normal([num_examples, noise_dimension])

# Generate images
generated_images = generator(noise, training=False)

# Display results
plt.imshow((generated_images[0] * 127.5 + 127.5).numpy(), cmap='gray')
```

## Configuration

- **Batch Size**: 256
- **Noise Dimension**: 100
- **Training Epochs**: 50
- **Optimizer**: Adam (learning rate 1e-4)
- **Loss Function**: Binary Cross-Entropy

## Results

The model generates realistic fashion item images after training. The training process includes visualization of generated images at different epochs to monitor progress.

## Checkpoints

Model checkpoints are saved in the `./training_checkpoints` directory, allowing for training resumption and model reuse.

## Tags

`GAN` `TensorFlow` `Deep-Learning` `Computer-Vision` `Image-Generation` `Fashion-MNIST` `Generative-Models` `DCGAN` `Machine-Learning` `Neural-Networks`

## Note

This implementation follows DCGAN best practices including:
- Use of LeakyReLU activations
- Batch normalization in generator
- Dropout in discriminator
- Proper weight initialization
- Separate optimizers for generator and discriminator

The project serves as an educational example of GAN implementation and can be extended for more complex image generation tasks.
