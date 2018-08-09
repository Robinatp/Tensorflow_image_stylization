# Tensorflow_image_stylization
  This is a repository of image stylization by Tensorflow implementation , include conditional instance normalization,conditional weight normalization,
conditional stylized-params normalization,adaptive instance normalization and perceptural losses


# Reference

1,perceptural losses:[fast-neural-style-tensorflow](https://github.com/Robinatp/Tensorflow_image_stylization/tree/master/fast-neural-style-tensorflow)

[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155), which first proposes an image transformation network
to transform input image into a pastiche outpur image and a loss network pretrained for image classification to define perceptual loss function

1,conditional instance normalization,conditional weight normalization:[image_stylization](https://github.com/Robinatp/Tensorflow_image_stylization/tree/master/image_stylization)
[A Learned Representation for Artistic Style](https://arxiv.org/abs/1610.07629), which proposes conditional instance normalization,conditional weight normalization
to get multiple styles per module and combile the arbitrary styles by weight

2,conditional stylized-params normalization:[arbitrary_image_stylization](https://github.com/Robinatp/Tensorflow_image_stylization/tree/master/arbitrary_image_stylization)
[Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/abs/1705.06830), which proposes a new style prediction
network that directly predicts an embedding vector S from an input style image for the style transfer network



4,adaptive instance normalization:[Tensorflow-Style-Transfer-with-Adain](https://github.com/Robinatp/Tensorflow_image_stylization/tree/master/Tensorflow-Style-Transfer-with-Adain)
[Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868), which propoces an AdaIn layer using to perform
style transfer in the feature sapce

All of method above has a fixed weight of a loss network pretrained for image classification in the train session,however in the inference there only a 
style transfer network which build a encoder-decoder network except arbitrary_image_stylization which specially include style prediction network additionaly
