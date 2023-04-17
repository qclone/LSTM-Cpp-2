#include "Cell.h"

Cell::Cell()
{

}

Cell::Cell(int _input_length,
           float _weight_range,
           float _learning_rate)
{
    Cell::id_number = rand() % 1000;

    Cell::input_length = _input_length;
    Cell::weight_range = _weight_range;
    Cell::learning_rate = vector<float>(_input_length);
    for(int i = 0; i < _input_length; i++)
        Cell::learning_rate.at(i) = _learning_rate;
    // Initialise weights of weight matrices
    Cell::initialiseWeights();
    Cell::initialiseBias();

    Cell::ctm1 = vector<float>(Cell::input_length);
    Cell::htm1 = vector<float>(Cell::input_length);
    Cell::xt = vector<float>(Cell::input_length);
}

void Cell::initialiseBias()
{
    Cell::bi = vector<float>(Cell::input_length);
    Cell::bg = vector<float>(Cell::input_length);
    Cell::bf = vector<float>(Cell::input_length);
    Cell::bo = vector<float>(Cell::input_length);
}

void Cell::initialiseState(vector<float> _c_tp, vector<float> _x_t, vector<float> _h_tp)
{
    Cell::ctm1 = _c_tp;
    Cell::htm1 = _h_tp;
    Cell::xt = _x_t;
}

void Cell::initialiseWeights()
{
    Operations::dimension wd;
    wd.d1 = Cell::input_length;
    wd.d2 = Cell::input_length;

    Cell::Wxi = Operations::fillDiagonalVector(Cell::Wxi, wd, Cell::weight_range);
    Cell::Whi = Operations::fillDiagonalVector(Cell::Whi, wd, Cell::weight_range);

    Cell::Wxg = Operations::fillDiagonalVector(Cell::Wxg, wd, Cell::weight_range);
    Cell::Whg = Operations::fillDiagonalVector(Cell::Whg, wd, Cell::weight_range);

    Cell::Wxf = Operations::fillDiagonalVector(Cell::Wxf, wd, Cell::weight_range);
    Cell::Whf = Operations::fillDiagonalVector(Cell::Whf, wd, Cell::weight_range);

    Cell::Wxo = Operations::fillDiagonalVector(Cell::Wxo, wd, Cell::weight_range);
    Cell::Who = Operations::fillDiagonalVector(Cell::Who, wd, Cell::weight_range);
}

void Cell::setY(vector<float> y)
{
    Cell::yt = y;
}

/** forward propagation functions **/
void Cell::forwardPropagation(void)
{
    Operations::dimension wd;
    wd.d1 = Cell::input_length;
    wd.d2 = Cell::input_length;

    // input gate
    Cell::Zg = Operations::multiplyWithWeights(Cell::Wxg, Cell::xt, wd);
    Cell::Zg = Operations::sumVectors(Cell::Zg, Operations::multiplyWithWeights(Cell::Whg, Cell::htm1, wd), Cell::input_length);
    Cell::Zg = Operations::sumVectors(Cell::Zg, Cell::bg, Cell::input_length);
    Cell::gt = Operations::tanh_vector(Cell::Zg, Cell::input_length);

    Cell::Zi = Operations::multiplyWithWeights(Cell::Wxi, Cell::xt, wd);
    Cell::Zi = Operations::sumVectors(Cell::Zi, Operations::multiplyWithWeights(Cell::Whi, Cell::htm1, wd), Cell::input_length);
    Cell::Zi = Operations::sumVectors(Cell::Zi, Cell::bi, Cell::input_length);
    Cell::it = Operations::sigmoid(Cell::Zi, Cell::input_length);

    Cell::i_gate = Operations::multiplyVectors(Cell::gt, Cell::input_length, Cell::it, Cell::input_length);

    // forget fate
    Cell::Zf = Operations::multiplyWithWeights(Cell::Wxf, Cell::xt, wd);
    Cell::Zf = Operations::sumVectors(Cell::Zf, Operations::multiplyWithWeights(Cell::Whf, Cell::htm1, wd), Cell::input_length);
    Cell::Zf = Operations::sumVectors(Cell::Zf, Cell::bf, Cell::input_length);
    Cell::ft = Operations::sigmoid(Cell::Zf, Cell::input_length);

    Cell::f_gate = Cell::ft;

    // output gate
    Cell::Zo = Operations::multiplyWithWeights(Cell::Wxo, Cell::xt, wd);
    Cell::Zo = Operations::sumVectors(Cell::Zo, Operations::multiplyWithWeights(Cell::Who, Cell::htm1, wd), Cell::input_length);
    Cell::Zo = Operations::sumVectors(Cell::Zo, Cell::bg, Cell::input_length);
    Cell::ot = Operations::sigmoid(Cell::Zo, Cell::input_length);

    Cell::o_gate = Cell::ot;

    // current cell state
    Cell::ct = Operations::multiplyVectors(Cell::ctm1, Cell::input_length, Cell::f_gate, Cell::input_length);
    Cell::ct = Operations::sumVectors(Cell::ct, Cell::i_gate, Cell::input_length);

    // prdiction state
    Cell::ht = Operations::tanh_vector(Cell::ct, Cell::input_length);
    Cell::ht = Operations::multiplyVectors(Cell::ht, Cell::input_length, Cell::o_gate, Cell::input_length);
}

/** gradient calculation functions **/
void Cell::gradientCalculation(void)
{
    //cout << "Calculating errors of the cell" << endl;
    // Error of cell
    Cell::dE = Operations::subVectors(Cell::yt, Cell::ct, Cell::input_length);
    cout << "dE = ";
    Operations::display1DVector(Cell::dE, Cell::input_length);

    //cout << "Calculating gradient with respect to gates" << endl;
    /*Gradient with respect to gates*/
    Cell::dEdot = Operations::tanh_vector(Cell::ct, Cell::input_length);
    Cell::dEdot = Operations::multiplyVectors(Cell::dEdot, Cell::input_length, Cell::dE, Cell::input_length);

    Cell::dEdct = Operations::invtanh_vector(Cell::ct, Cell::input_length);
    Cell::dEdct = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdct = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::dE, Cell::input_length);

    Cell::dEdit = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::it, Cell::input_length);

    Cell::dEdft = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::ctm1, Cell::input_length);

    Cell::dEdctm1 = Operations::multiplyVectors(Cell::dEdct, Cell::input_length, Cell::ft, Cell::input_length);

    //cout << "Calculating gradient with respect to output weights" << endl;
    /*Gradient with respect to output weights*/
    Cell::dEdbo = Operations::multiplyVectors(Cell::dE, Cell::input_length, Operations::tanh_vector(Cell::ct, Cell::input_length), Cell::input_length);
    Cell::dEdbo = Operations::multiplyVectors(Cell::dEdbo, Cell::input_length, Operations::sigmoid(Cell::Zo, Cell::input_length), Cell::input_length);
    Cell::dEdbo = Operations::multiplyVectors(Cell::dEdbo, Cell::input_length, Operations::invsigmoid(Cell::Zo, Cell::input_length), Cell::input_length);

    Cell::dEdWxo = Operations::multiplyVectors(Cell::dEdbo, Cell::input_length, Cell::xt, Cell::input_length);
    Cell::dEdWho = Operations::multiplyVectors(Cell::dEdbo, Cell::input_length, Cell::htm1, Cell::input_length);

    //cout << "Calculating gradient with respect to forget weights" << endl;
    /*Gradient with respect to forget weights*/
    Cell::dEdbf = Operations::multiplyVectors(Cell::dE, Cell::input_length, Operations::invtanh_vector(Cell::ct, Cell::input_length), Cell::input_length);
    Cell::dEdbf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdbf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::ctm1, Cell::input_length);
    Cell::dEdbf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Operations::sigmoid(Cell::Zf, Cell::input_length), Cell::input_length);
    Cell::dEdbf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Operations::invsigmoid(Cell::Zf, Cell::input_length), Cell::input_length);

    Cell::dEdWxf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::xt, Cell::input_length);
    Cell::dEdWhf = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::htm1, Cell::input_length);

    //cout << "Calculating gradient with respect to input weights" << endl;
    /*Gradient with respect to input weights*/
    Cell::dEdbi = Operations::multiplyVectors(Cell::dE, Cell::input_length, Operations::invtanh_vector(Cell::ct, Cell::input_length), Cell::input_length);
    Cell::dEdbi = Operations::multiplyVectors(Cell::dEdbi, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdbi = Operations::multiplyVectors(Cell::dEdbi, Cell::input_length, Cell::gt, Cell::input_length);
    Cell::dEdbi = Operations::multiplyVectors(Cell::dEdbi, Cell::input_length, Operations::sigmoid(Cell::Zi, Cell::input_length), Cell::input_length);
    Cell::dEdbi = Operations::multiplyVectors(Cell::dEdbi, Cell::input_length, Operations::invsigmoid(Cell::Zi, Cell::input_length), Cell::input_length);

    Cell::dEdWxi = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::xt, Cell::input_length);
    Cell::dEdWhi = Operations::multiplyVectors(Cell::dEdbf, Cell::input_length, Cell::htm1, Cell::input_length);

    //cout << "Calculating gradient with respect to c and g states" << endl;
    /*Gradient with respect to c and g states*/
    Cell::dEdbg = Operations::multiplyVectors(Cell::dE, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdbg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Operations::invtanh_vector(Cell::ct, Cell::input_length), Cell::input_length);
    Cell::dEdbg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Cell::ot, Cell::input_length);
    Cell::dEdbg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Cell::it, Cell::input_length);
    Cell::dEdbg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Operations::invtanh_vector(Cell::Zg, Cell::input_length), Cell::input_length);

    Cell::dEdWhg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Cell::htm1, Cell::input_length);
    Cell::dEdWxg = Operations::multiplyVectors(Cell::dEdbg, Cell::input_length, Cell::xt, Cell::input_length);
}

/** Weight adjustment functions **/
void Cell::updateWeights(void)
{
    //cout << "Cell weights being updated" << endl;
    Cell::Wxo = Operations::subWeights(Cell::Wxo, Operations::multiplyVectors(Cell::dEdWxo, Cell::input_length, Cell::learning_rate, Cell::input_length), Cell::input_length);
    Cell::Who = Operations::subWeights(Cell::Who, Operations::multiplyVectors(Cell::dEdWho, Cell::input_length, Cell::learning_rate, Cell::input_length), Cell::input_length);
    Cell::bo = Operations::sumVectors(Cell::bo, Cell::dEdbo, Cell::input_length);

    Cell::Wxf = Operations::subWeights(Cell::Wxf, Operations::multiplyVectors(Cell::dEdWxf, Cell::input_length, Cell::learning_rate, Cell::input_length), Cell::input_length);
    Cell::Whf = Operations::subWeights(Cell::Whf, Operations::multiplyVectors(Cell::dEdWhf, Cell::input_length, Cell::learning_rate, Cell::input_length), Cell::input_length);
    Cell::bf = Operations::sumVectors(Cell::bf, Cell::dEdbf, Cell::input_length);

    Cell::Wxi = Operations::subWeights(Cell::Wxi, Operations::multiplyVectors(Cell::dEdWxi, Cell::input_length, Cell::learning_rate, Cell::input_length), Cell::input_length);
    Cell::Whi = Operations::subWeights(Cell::Whi, Operations::multiplyVectors(Cell::dEdWhi, Cell::input_length, Cell::learning_rate, Cell::input_length), Cell::input_length);
    Cell::bi = Operations::sumVectors(Cell::bi, Cell::dEdbi, Cell::input_length);

    Cell::Wxg = Operations::subWeights(Cell::Wxg, Operations::multiplyVectors(Cell::dEdWxg, Cell::input_length, Cell::learning_rate, Cell::input_length), Cell::input_length);
    Cell::Whg = Operations::subWeights(Cell::Whg, Operations::multiplyVectors(Cell::dEdWhg, Cell::input_length, Cell::learning_rate, Cell::input_length), Cell::input_length);
    Cell::bg = Operations::sumVectors(Cell::bg, Cell::dEdbg, Cell::input_length);
}

vector<float> Cell::getht(void)
{
    return Cell::ht;
}

vector<float> Cell::getct(void)
{
    return Cell::ct;
}



