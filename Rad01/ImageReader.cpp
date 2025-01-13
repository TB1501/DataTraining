#include "ImageReader.h"
#include "EndianConversion.h"
#include <iostream>
#include <fstream>
#include <vector>

std::vector<std::vector<unsigned char>> readImages(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    // Read the magic number, number of images, rows, and columns
    char magicBytes[4], numImagesBytes[4], numRowsBytes[4], numColumnsBytes[4];
    file.read(magicBytes, 4);
    file.read(numImagesBytes, 4);
    file.read(numRowsBytes, 4);
    file.read(numColumnsBytes, 4);

    // Convert the images from big-endian to integers
    int numImages = convertBigEndianToInt(numImagesBytes);
    int numRows = convertBigEndianToInt(numRowsBytes);
    int numColumns = convertBigEndianToInt(numColumnsBytes);

    //Print the number of images, rows, and columns - CHECK IF THE CONVERSION IS CORRECT
    std::cout << "Images: " << numImages << ", Rows: " << numRows << ", Columns: " << numColumns << std::endl;

    // Allocate a vector to hold all images
    std::vector<std::vector<unsigned char>> results;

    for (int i = 0; i < numImages; i++) {

        // Allocate a vector to hold the current image
        std::vector<unsigned char> image(numRows * numColumns);

        // Read the current image
        file.read((char*)(image.data()), numRows * numColumns);

        // Add the current image to the results
        results.push_back(image);

    }

    file.close();
    return results;
}