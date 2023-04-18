#ifndef CELL_H
#define CELL_H

#include "Operations.h"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

class Cell
{
    public:
    /** Initialisating functions **/
        // "unique" (random) number for LSTM cell.
        // for debugging.
        int id_number;

        Cell();
        // this constructor sets the output length, input length,
        // weight_range, learning_rate, learning_momenum,
        Cell(int, float, float);

        // this initialises the weight matrices.
        // values are from 0 to Cell::weight_range.
        void initialiseWeights();
        // this initialises the bias vectors.
        // bias vectors are from 0 to Cell::weight_range.
        void initialiseBias();
        // this initialises the state of the LSTM (ct-1, ht-1, xt).
        void initialiseState(vector<float>, vector<float>, vector<float>);

        string toString(void);

        /** forward propagation functions **/
        void forwardPropagation(void);

        /** gradient calculation functions **/
        void gradientCalculation(void);

        /** Weight adjustment functions **/
        void updateWeights(void);

        vector<float> getht(void);
        vector<float> getct(void);
        void setY(vector<float>);

        /** LSTM properties **/
        int input_length;
        float weight_range;
        vector<float> learning_rate;
    private:
    /** Cell state and gate vectors **/
        // cell state
        vector<float> ct;
        vector<float> ctm1;
        // prediction
        vector<float> ht;
        vector<float> htm1;
        // input/output values
        vector<float> xt;
        vector<float> yt; // expected value
        vector<float> ot;
        vector<float> ft;
        vector<float> gt;
        vector<float> it;
        // input/output gates
        vector<float> i_gate;
        vector<float> o_gate;
        vector<float> f_gate;
    /** Weight multiplier vector midpoints* **/
    // * the output of multiplying the weights with inputs/states
    // before sigmoid or tanh is applied.
        vector<float> Zf;
        vector<float> Zg;
        vector<float> Zo;
        vector<float> Zi;
    /** Private weight matrices and bias vectors **/
        // input weights
        vector<vector<float>> Wxi;
        vector<vector<float>> Whi;
        vector<float> bi;
        // candidate weights
        vector<vector<float>> Wxg;
        vector<vector<float>> Whg;
        vector<float> bg;
        // forget weights
        vector<vector<float>> Wxf;
        vector<vector<float>> Whf;
        vector<float> bf;
        // output weights
        vector<vector<float>> Wxo;
        vector<vector<float>> Who;
        vector<float> bo;

    /** Error **/
    vector<float> dE; // = y-ht (mean square error)
    /** Gate gardients **/
    vector<float> dEdot;
    vector<float> dEdct;
    vector<float> dEdit;
    vector<float> dEdft;
    vector<float> dEdctm1;
    /** Output weight gradients **/
    vector<float> dEdWxo;
    vector<float> dEdWho;
    vector<float> dEdbo;
    /** Forget weight gradients **/
    vector<float> dEdWxf;
    vector<float> dEdWhf;
    vector<float> dEdbf;
    /** Input weight gradients **/
    vector<float> dEdWxi;
    vector<float> dEdWhi;
    vector<float> dEdbi;
    /** Gradients of candidate **/
    vector<float> dEdWxg;
    vector<float> dEdWhg;
    vector<float> dEdbg;
};

#endif // CELL_H
