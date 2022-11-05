//
// Created by ricardo on 5/11/22.
//

#ifndef CUDAKNN_DATASET_H
#define CUDAKNN_DATASET_H

#include <string>
#include <fstream>
#include <vector>

class Dataset {
private:
    long long int totalPoints;
    long long int totalDimensions;
    std::vector<std::vector<long double>> points;

public:
    Dataset(std::string filePath) {
        std::ifstream file(filePath);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filePath << std::endl;
            exit(EXIT_FAILURE);
        }

        file >> totalPoints;

        std::string line;
        while (std::getline(file, line)) {
            std::cout << line << std::endl;
        }
        file.close();

    }

    long long int getTotalPoints() {
        return totalPoints;
    }

    long long int getTotalDimensions() {
        return totalDimensions;
    }
};

#endif //CUDAKNN_DATASET_H
