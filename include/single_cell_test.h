#ifndef SINGLE_CELL_TEST_H
#define SINGLE_CELL_TEST_H

#include "Cell.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class single_cell_test
{
    public:
        void test(void);

        int _input_length = 20;
        float _weight_range = 1;
        float _learning_rate = 0.1;

        int ROUNDS = 55;

        vector<float> _c_tp =   {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        vector<float> _x_t =    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
        vector<float> _y_t =    {1,2,3,4,5,6,9,3,4,5,4,2,3,1,4,4,4,1,1,1};
        vector<float> _h_tp =   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
};

#endif // SINGLE_CELL_TEST_H
