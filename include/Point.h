//
// Created by ricardo on 6/11/22.
//

#ifndef CUDAKNN_POINT_H
#define CUDAKNN_POINT_H

typedef struct NodeThrust {
    double x;
    double y;
    double z;
    int id;
    size_t label;
    double distance;

    __host__ __device__ bool operator<(const NodeThrust &p) const {
        return distance < p.distance;
    }

} NodeThrust;

struct GPUPoint {
    float x;
    float y;
    float z;
    int id;
};

enum DistanceType {
    EUCLIDEAN,
    MANHATTAN,
    CHEBYSHEV
};

class Point {
public:

    int id;
    DistanceType distanceType;
    double x;
    double y;
    double z;
    size_t label;
    double distance;

    Point(size_t id, double x, double y, double z, size_t label) : id(id), x(x), y(y), z(z), label(label),
                                                                   distance(0.0),
                                                                   distanceType(EUCLIDEAN) {}

    Point() : id(-1), x(0), y(0), z(0), label(0), distance(0.0), distanceType(EUCLIDEAN) {}

    __host__ __device__ bool operator<(const Point &p) const {
        return distance < p.distance;
    }

    [[nodiscard]] double computeDistance(const Point &p, DistanceType type) const {

        double distance = 0.0;

        switch (type) {
            case EUCLIDEAN:
                distance = sqrt(pow(x - p.x, 2) + pow(y - p.y, 2) + pow(z - p.z, 2));
            case MANHATTAN:
                break;
            case CHEBYSHEV:
                break;
            default:
                return sqrt(pow(x - p.x, 2) + pow(y - p.y, 2) + pow(z - p.z, 2));
        }

        return distance;
    }

    [[nodiscard]] size_t getId() const {
        return id;
    }

    [[nodiscard]] DistanceType getDistanceType() const {
        return distanceType;
    }

    [[nodiscard]] double getX() const {
        return x;
    }

    [[nodiscard]] double getY() const {
        return y;
    }

    [[nodiscard]] double getZ() const {
        return z;
    }

    [[nodiscard]] size_t getLabel() const {
        return label;
    }

    void setId(int id_) { this->id = id_; }

    friend std::ostream &operator<<(std::ostream &os, const Point &p) {
        os << p.x << " " << p.y << " " << p.z;
        return os;
    }

    friend std::istream &operator>>(std::istream &is, Point &p) {
        is >> p.x >> p.y >> p.z;
        return is;
    }

};

#endif //CUDAKNN_POINT_H
