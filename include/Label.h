//
// Created by ricardo on 7/11/22.
//

#ifndef CUDAKNN_LABEL_H
#define CUDAKNN_LABEL_H

class Label {
public:
    size_t label;
    int frequency;

    __host__ __device__ bool operator<(const Label &l) const {
        return frequency < l.frequency;
    }
};

#endif //CUDAKNN_LABEL_H
