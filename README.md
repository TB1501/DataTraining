# DataTraining

This is an ML model for recognizing handwritten digits.

The adopted library for building the model is OpenCV.

The model is trained on the MNIST dataset. The MNIST dataset is a set of 60000 images of handwritten digits from 0-9 and coresponding labels.

The adopted architecture was the Multi-Layer Perceptron (MLP) and for data training the backpropagation algorithm was selected.

The build of the model followed these steps:
1st step:

- in the 1st step the images were loaded and transformed to a format used in OpenCv (cv::Mat)
- the image data was normalized to values betweeen 0 and 1 in order to enchance model performence

2nd step:

- in the 2nd step the model was generated
- for the model architecture the MLP was selected which is a format of deep learning
- the activation function was set as Sygmoid
- the model is built with 3 layers (input, gidden and output layer)

3rd step:

- in the 3rd step the data was prepared for training
- the container with matrix data was transformed to 1D vector for further processing

4th step:

- in the 4th step the model was trained
- the criteria for model training were set as maximum iterations and desired accuracy
- the training methode was set as backpropagation
