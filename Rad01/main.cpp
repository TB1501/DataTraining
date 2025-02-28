#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include "EndianConversion.h"
#include"ImageReader.h"
#include"LabelReader.h"



int main() {

    /*1ST STEP: INITIALIZE THE MODEL. CONVERT THE DATA TO OPENCV FORMAT*/

    std::string fileNameImages = "C:\\Users\\Tin\\Desktop\\archive (1)\\train-images-idx3-ubyte\\train-images-idx3-ubyte";
    std::string fileNameLabels = "C:\\Users\\Tin\\Desktop\\archive (1)\\train-labels-idx1-ubyte\\train-labels-idx1-ubyte";

    std::vector<std::vector<unsigned char>> images = readImages(fileNameImages);
    std::vector<std::vector<unsigned char>> labels = readLabels(fileNameLabels);

    //These are the containers for images and labels
    std::vector<cv::Mat> imagesData;
    std::vector<int> labelsData;

    if (!images.empty()) {
        std::cout << "Successfully read " << images.size() << " images." << std::endl;
    }
    else {
        std::cout << "Failed to read images." << std::endl;
    }

    if (!labels.empty()) {
        std::cout << "Successfully read " << labels.size() << " labels." << std::endl;
    }
    else {
        std::cout << "Failed to read labels." << std::endl;
    }

    //In this part we are extracting the images from vector to a vector of cv::Mat type (OpenCv matrix format)

    for (auto i = 0u; i < 10; ++i)
    {
        //Initializing the cv::Mat object with the size of 28x28 and setting the pixel values to 0
        cv::Mat m(28, 28, CV_8UC1, cv::Scalar(0));
        for (int row = 0; row < m.rows; ++row)
            for (int col = 0; col < m.cols; ++col)
                m.at<uchar>(row, col) = images[i][row * m.cols + col];

        cv::Mat m_normalized;
        cv::normalize(m, m_normalized, 0, 1, cv::NORM_MINMAX);
        imagesData.push_back(std::move(m_normalized));
        labelsData.push_back(labels[i].front());
    }

    //Checking the size of the images and labels
    std::cout << "ImagesData size: " << imagesData.size()<<"   " << "LabelsData size: " << labelsData.size() << std::endl;


    //___________________________________________________________________________________________________________________________________________________

    /*2ND STEP: INITIALIZING THE NEURAL NETWORK MODEL. HERE WE ARE USING THE MLP (MULTILAYER PERCEPTRON) MODEL */

    //Initializing the MLP model
    cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();

    //Setting the activation function for the neurons in the hidden layer. In this case we are using the sigmoid function and setting the alpha and beta values to 1
    mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);

    //Here we are defining the number of input, hidden and output layers
    //Input layers: equals the number of features (number of pixels for 1 picture)
    int inputLayerSize = imagesData[0].total();
    if (inputLayerSize > std::numeric_limits<int>::max()) {
        throw std::overflow_error("inputLayerSize exceeds the maximum value for int");
    }
    //Hidden layer: empirical value
    int hiddenLayerSize = 100;
    //Output layer: equals the number of classes (0-9)
    int outputLayerSize = 10;

    //Creating the layers matrix for the NN
    cv::Mat layers = (cv::Mat_<int>(3, 1)<<inputLayerSize, hiddenLayerSize, outputLayerSize);

    //Assigning the layers to the model
    mlp->setLayerSizes(layers);

    //___________________________________________________________________________________________________________________________________________________

    /*3RD STEP: PREPARING THE TRAINING DATA*/

    int numSamples = imagesData.size();

    //Creating the training data and label data matrices. 
    cv::Mat trainingData(numSamples, inputLayerSize, CV_32F);
    cv::Mat labelData(numSamples, outputLayerSize, CV_32F);

    //Here we are converting the images and labels to the training and label data matrices
    for (int i = 0; i < numSamples; i++) {

        cv::Mat image = imagesData[i].reshape(1, 1);
        image.convertTo(trainingData.row(i), CV_32F);

        cv::Mat label = cv::Mat::zeros(1, outputLayerSize, CV_32F);
        label.at<float>(0, labelsData[i]) = 1.0;
        label.copyTo(labelData.row(i));

    }

    //Checking the size of the training and label data
    std::cout << "3rd step: " << trainingData.size() << "  " << labelData.size() << std::endl;


    //___________________________________________________________________________________________________________________________________________________
    
    /*4TH STEP: TRAINING THE MODEL*/

    //Setting the termination criteria for the training. Maximum iteration is set to 10000 and epsilon to 0.001 (minimum value for the error)
    cv::TermCriteria termCrit(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 0.001);
    mlp->setTermCriteria(termCrit);

    //Setting the training method. In this case we are using the backpropagation method with the learning rate of 0.001 and momentum of 0.1
    mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001, 0.1);

    //Training the model
    mlp->train(trainingData, cv::ml::ROW_SAMPLE, labelData);

    mlp->save("C:\\Users\\Tin\\Desktop\\Vsite\\3 godina\\Zavrsni\\DataTraining\\Trained_model\\trainedModel_6.xml");
    return 0;



}
