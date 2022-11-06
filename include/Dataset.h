//
// Created by ricardo on 5/11/22.
//

#ifndef CUDAKNN_DATASET_H
#define CUDAKNN_DATASET_H

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include "Point.h"

class Dataset {
private:
    Point *points;
    size_t nPoints{};

public:
    explicit Dataset(const std::string& path) {
        std::ifstream file(path);

        if (!file.is_open()) {
            fprintf(stderr, "Error opening file %s", path.c_str());
            exit(EXIT_FAILURE);
        }

        file >> this->nPoints;

        this->points = (Point *) malloc(nPoints * sizeof(Point));

        for (size_t index = 0; index < this->nPoints; ++index) {
            file >> this->points[index];
            this->points[index].setId((int) index);
        }
    }

    [[nodiscard]] size_t getNPoints() const {
        return nPoints;
    }

    [[nodiscard]] Point *getPoints() const {
        return points;
    }

};

#endif //CUDAKNN_DATASET_H
