#ifndef NETWORK_H
#define NETWORK_H


#include <iostream>
#include "Cell.h"
#include "LSTMCell.h"
#include "ForwardPropagationCell.h"
#include "BackPropagationCell.h"
#include "Operations.h"

using namespace std;

class Network
{
    public:
        Network();

    protected:

    private:
        vector<Cell> net;

};

#endif // NETWORK_H
