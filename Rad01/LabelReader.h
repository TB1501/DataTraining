#pragma once
#include <fstream>
#include <vector>


// Reading training labels from an IDX file. In our case the MNIST dataset.
std::vector<std::vector<unsigned char>> readLabels(const std::string& filename);