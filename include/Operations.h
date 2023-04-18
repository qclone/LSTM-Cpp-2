#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <random>
#include <cmath>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>

using namespace std;

class Operations
{
public:
    struct dimension
    {
        int d1;
        int d2;
    };

    static vector<vector<float>> fillDiagonalVector(vector<vector<float>>, dimension, float);
    static vector<float> multiplyWithWeights(vector<vector<float>>, vector<float>, dimension);
    static vector<vector<float>> subWeights(vector<vector<float>>, vector<float>, int);

    static void display2DVector(vector<vector<float>>, int, int);
    static void display1DVector(vector<float>, int);

    static vector<float> sigmoid(vector<float>, int);
    static vector<float> invsigmoid(vector<float>, int);
    static vector<float> tanh_vector(vector<float>, int);
    static vector<float> invtanh_vector(vector<float>, int);
    static vector<float> softmax(vector<float>, int);

    //static vector<vector<float>> multiplyVectors(vector<float>, int, vector<float>, int);
    static vector<float> multiplyVectors(vector<float>, int, vector<float>, int);
    static vector<float> sumVectors(vector<float>, vector<float>, int);
    static vector<float> subVectors(vector<float>, vector<float>, int);
    static vector<float> squareVector(vector<float>, int);

    static void save2DVector(vector<vector<float>>, int, int, string);
    static void save1DVector(vector<float>, int, string);

    static vector<float> standardiseVector(vector<float>, int);
    static vector<float> standardiseVector(vector<float>, int, float);
    static float mean(vector<float>, int);
    static float stdev(vector<float>, int);
    static float getMaximum(vector<float>, int);
};

#endif // OPERATIONS_H
