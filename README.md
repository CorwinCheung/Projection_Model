# Vision Models Collection

This repository hosts a collection of advanced computer vision models designed to address various tasks in 3D modeling, style transformation, and view synthesis.

## 3D CAD to 2D Projection and Reconstruction

These involve the conversion of 3D CAD models into 2D projections and the subsequent reconstruction of 3D objects from these 2D views. The primary objectives are:

- **3D to 2D Projection**: Converting detailed 3D CAD models into 2D representations from multiple perspectives.
- **2D to 3D Reconstruction**: Using neural networks to accurately reconstruct 3D objects from limited 2D views.

### Goals

- Compatibility with various 3D CAD formats.
- Accurate reconstruction with minimal input views.
- Utilization of sophisticated neural network architectures for precise 3D modeling.

## Style Transfer GAN

This implementation of a Style Transfer Generative Adversarial Network (GAN) transforms images into different artistic styles, focusing particularly on stained glass aesthetics. The main components are:

- **Generator Network**: Transforms input images into a specified artistic style.
- **Discriminator Network**: Evaluates the quality of the generated images to ensure style fidelity.

### Goals

- High-quality style transfer with a focus on stained glass effects.
- Adjustable style parameters for custom outputs.

## Neural Radiance Fields (NeRF)

An implementation of Neural Radiance Fields (NeRF) model generates 2D views of objects from different angles, offering a sophisticated approach to view synthesis. The model features:

- **Scene Representation**: Encodes 3D scenes into a continuous volumetric format.
- **View Synthesis**: Generates novel views by querying the neural network from various perspectives.

### Goals

- High-fidelity 2D view synthesis from limited input images.
- Supports complex scene rendering with realistic lighting and textures.

