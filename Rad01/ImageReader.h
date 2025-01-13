#pragma once
#include <fstream>
#include <vector>

// This is a function that reads images from an IDX file. In our case the MNIST dataset.
std::vector<std::vector<unsigned char>> readImages(const std::string& filename);