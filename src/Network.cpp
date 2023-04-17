#include "Network.h"

Network::Network()
{
    int _output_length = 4;
    int _input_length = 4;
    float _weight_range = 1;
    float _learning_rate = 1;
    float _learning_momentum = 1;

    Network::net = vector<Cell>(10);
    cout << "If 10 unique LSTM cells are created we should see 10 unique numbers:" << endl;
    // network size should be configured later...
    for(int i = 0; i < 10; i++)
    {
        Network::net.at(i) = Cell(_input_length,
                                  _weight_range,
                                  _learning_rate);
        cout << Network::net.at(i).id_number << endl;
    }
}
