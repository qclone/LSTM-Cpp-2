#ifndef NETWORK_TEST_H
#define NETWORK_TEST_H

#include "Cell.h"
#include "Network.h"
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class network_test
{
    public:
        void test(void);

        int _input_length = 2;
        float _weight_range = 1;
        float _learning_rate = 1;
        int _network_size = 32;

        Network* net;

        int ROUNDS = 2000;
};

#endif // NETWORK_TEST_H
