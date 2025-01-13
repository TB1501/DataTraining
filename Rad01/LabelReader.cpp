#include "LabelReader.h"
#include "EndianConversion.h"
#include <iostream>
#include <fstream>
#include <vector>

std::vector<std::vector<unsigned char>> readLabels(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return {};
    }

    // Read the magic number, number of labels
    char magicBytes[4], numLabelBytes[4];
    file.read(magicBytes, 4);
    file.read(numLabelBytes, 4);


    int numLabels = convertBigEndianToInt(numLabelBytes);

    std::cout << "Labels: " << numLabels << std::endl;

    // Allocate a vector to hold all labels
    std::vector<std::vector<unsigned char>> results;

    for (int i = 0; i < numLabels; i++) {

        // Allocate a vector to hold the current label
        std::vector<unsigned char> label(1);

        // Read the current label
        file.read((char*)(label.data()), 1);

        // Add the current label to the results
        results.push_back(label);

    }

    file.close();
    return results;
}