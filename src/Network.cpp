#include "Network.h"

Network::Network(int _input_length,
                 float _weight_range,
                 float _learning_rate,
                 int _network_size)
{
    Network::targets = vector<vector<float>>(_network_size, vector<float>(_input_length));
    Network::predictions = vector<vector<float>>(_network_size, vector<float>(_input_length));

    Network::network_size = _network_size;
    Network::net = vector<Cell*>(Network::network_size);
    for(int i = 0; i < Network::network_size; i++)
    {
        Network::net.at(i) = new Cell(_input_length,
                                  _weight_range,
                                  _learning_rate);
    }
}

void Network::feedNetwork(void)
{
    Network::net.at(0)->setY(Network::targets.at(0));
    Network::net.at(0)->initialiseState(Network::c0, Network::x0, Network::h0);
    Network::net.at(0)->forwardPropagation();

    for(int i = 1; i < Network::network_size; i++)
    {
        Network::net.at(i)->setY(Network::targets.at(i));
        Network::net.at(i)->initialiseState(Network::net.at(i-1)->getct(),
                                            Network::x0,
                                            Network::net.at(i-1)->getht());
        Network::net.at(i)->forwardPropagation();
        Network::predictions.at(i) = Network::net.at(i)->getct();
    }
}

void Network::trainNetwork(int ROUNDS)
{
    for(int i = 1; i < Network::network_size; i++)
        Network::predictions.at(i) = {0,0};
    for(int i = 0; i < ROUNDS; i++)
    {
        Network::net.at(0)->setY(Network::targets.at(0));
        Network::net.at(0)->initialiseState(Network::c0, Network::x0, Network::h0);
        Network::net.at(0)->forwardPropagation();
        Network::predictions.at(0) = Network::net.at(0)->getct();
        Network::net.at(0)->gradientCalculation();
        Network::net.at(0)->updateWeights();

        for(int i = 1; i < Network::network_size; i++)
        {
            Network::net.at(i)->setY(Network::targets.at(i));
            Network::net.at(i)->initialiseState(Network::net.at(i-1)->getct(),
                                                Network::x0,
                                                Network::net.at(i-1)->getht());
            Network::net.at(i)->forwardPropagation();
            Network::predictions.at(i) = Network::net.at(i)->getct();
            Network::net.at(i)->gradientCalculation();
            Network::net.at(i)->updateWeights();
        }
    }
}

void Network::validateNetwork()
{
    Network::net.at(0)->initialiseState(Network::c0, Network::x0, Network::h0);
    Network::net.at(0)->forwardPropagation();
    Network::feedNetwork();
}

void Network::testNetwork(vector<float> _xt)
{
    Network::x0 = _xt;
    Network::feedNetwork();
}
