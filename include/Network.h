#ifndef NETWORK_H
#define NETWORK_H


#include <iostream>
#include <vector>
#include "Cell.h"
#include "Operations.h"

using namespace std;

class Network
{
    public:
        Network(int, float, float, int);
        void feedNetwork(void);
        void trainNetwork(int);
        void validateNetwork(void);
        void testNetwork(vector<float>);

        vector<vector<float>> targets;
        vector<vector<float>> predictions;

        vector<float> x0;
        vector<float> c0;
        vector<float> h0;


        int network_size;

    protected:

    private:
        vector<Cell*> net;

};

#endif // NETWORK_H
