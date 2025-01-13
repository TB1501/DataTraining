#include "EndianConversion.h"

int convertBigEndianToInt(const char* bytes) {
    return (static_cast<unsigned char>(bytes[0]) << 24) |
        (static_cast<unsigned char>(bytes[1]) << 16) |
        (static_cast<unsigned char>(bytes[2]) << 8) |
        static_cast<unsigned char>(bytes[3]);
}