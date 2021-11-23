# Speech-To-Speech-Generative-Model
Speech Generation with Generative modeling for MNIST digits using CNN VAE and conditional GAN.

**Melspectogram is used for audio feature extraction.**

The Generated Spectrogram from latent Space could easily be distinguised between VAE and GAN Architecture. The Spectrogram generated from GAN contains more details and has better sound quality, However VAE gives better explicit density estimation of voices on latent space.

| Original| VAE Generated      |    GAN Generated|           
|------------|-------------|--------------|
| <img src="https://github.com/ashwani-adu3839/Speech-To-Speech-Generative-Model/blob/main/Img/Original.png" width="400"> | <img src="https://github.com/ashwani-adu3839/Speech-To-Speech-Generative-Model/blob/main/Img/VAE.png" width="400"> |<img src="https://github.com/ashwani-adu3839/Speech-To-Speech-Generative-Model/blob/main/Img/GAN.png" width="400"> |

The t-SNE  plot of latent space in VAE model. The different positions of same digits corresponds to voices of different speakers.

<img src="https://github.com/ashwani-adu3839/Speech-To-Speech-Generative-Model/blob/main/Img/tsnePLot.png" width="600"> 



# Link
[Tensorflow](https://www.tensorflow.org/api_docs) - _Tensorflow_

[Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) - _A simple audio/speech dataset consisting of recordings of spoken digits_

[Conditional GAN](https://arxiv.org/abs/1411.1784) - _Conditional Generative Adversarial Nets_
[VAE](https://arxiv.org/abs/1312.6114) - _Auto-Encoding Variational Bayes_
