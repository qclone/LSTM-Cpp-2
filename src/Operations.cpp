#include "Operations.h"

vector<float> Operations::multiplyWithWeights(vector<vector<float>> W, vector<float> X, dimension wd)
{
    // initialise Y vector to the row dimension of matrix
    vector<float> Y(wd.d1);
    // multiply X vector with the column dimension of matrix
    for(int i = 0; i < wd.d1; i++)
    {
        float row_sum = 0;
        for(int j = 0; j < wd.d2; j++)
        {
            row_sum += (X.at(j) * W.at(i).at(j));
        }
        Y.at(i) = row_sum;
    }
    return Y;
}

void Operations::display2DVector(vector<vector<float>> vec, int d1, int d2)
{
    cout << endl;
    for(int i = 0; i < d1; i++)
    {
        for(int j = 0; j < d2; j++)
        {
            cout << "" << vec.at(i).at(j) << ",";
            if(vec.at(i).at(j) == 0){ cout << "\t\t"; }
            else { cout << "\t"; }
        }
        cout << endl;
    }
}

void Operations::display1DVector(vector<float> vec, int d1)
{
    cout << endl;
    for(int i = 0; i < d1; i++)
    {
        cout << "" << vec.at(i) << ",";
        if(vec.at(i) == 0){ cout << "\t\t"; }
        else { cout << "\t"; }
    }
    cout << endl;
}

vector<vector<float>> Operations::fillDiagonalVector(vector<vector<float>> vec, dimension wp, float w)
{
    vector<float> row(wp.d1);
    vector<vector<float>> matrix(wp.d2, row);
    vec = matrix;
    for(int i = 0; i < wp.d1; i++)
        for(int j = 0; j < wp.d2; j++)
        {
            vec.at(i).at(j) = 0;
            if(i == j) // diagonal position
                vec.at(i).at(j) = static_cast <float> (rand() / static_cast <float> (RAND_MAX/w));
        }
    return vec;
}

vector<float> Operations::invsigmoid(vector<float> vec, int d1)
{
    vector<float> vec_sigmoid(d1);
    for(int i = 0; i < d1; i++)
    {
        vec_sigmoid.at(i) = 1 - (1 / (1 + exp(-vec.at(i))));
    }

    return vec_sigmoid;
}

vector<float> Operations::sigmoid(vector<float> vec, int d1)
{
    vector<float> vec_sigmoid(d1);
    for(int i = 0; i < d1; i++)
    {
        vec_sigmoid.at(i) = (1 / (1 + exp(-vec.at(i))));
    }

    return vec_sigmoid;
}

vector<float> Operations::invtanh_vector(vector<float> vec, int d1)
{
    vector<float> vec_invtanh(d1);
    for(int i = 0; i < d1; i++)
    {
        vec_invtanh.at(i) = (1 - ((tanh(vec.at(i)))*(tanh(vec.at(i)))));
    }
    return vec_invtanh;
}

vector<float> Operations::tanh_vector(vector<float> vec, int d1)
{
    vector<float> vec_tanh(d1);
    for(int i = 0; i < d1; i++)
    {
        vec_tanh.at(i) = tanh(vec.at(i));
    }

    return vec_tanh;
}

/*vector<vector<float>> Operations::multiplyVectors(vector<float> vec1, int d1, vector<float> vec2, int d2)
{
    vector<float> rows(d1);
    vector<vector<float>> Y(d2, rows);
    for(int i = 0; i < d1; i++)
        for(int j = 0; j < d2; j++)
        {
            Y.at(i).at(j) = vec1.at(i) * vec2.at(j);
        }
    return Y;
}*/

/*This version of vector multiply is like ./ in matlab*/
vector<float> Operations::multiplyVectors(vector<float> vec1, int d1, vector<float> vec2, int d2)
{
    vector<float> Y(d1);
    for(int i = 0; i < d1; i++)
            Y.at(i) = vec1.at(i) * vec2.at(i);
    return Y;
}

vector<float> Operations::subVectors(vector<float> _i, vector<float> _j, int d1)
{
    vector<float> _k(d1);
    for(int i = 0; i < d1; i++)
    {
        _k.at(i) = _i.at(i) - _j.at(i);
    }
    return _k;
}

vector<vector<float>> Operations::subWeights(vector<vector<float>> _i, vector<float> _j, int d1)
{
    vector<float> row(d1);
    vector<vector<float>> _k(d1, row);
    for(int i = 0; i < d1; i++)
        for(int j = 0; j < d1; j++)
            if(i == j)
                _k.at(i).at(j) = _i.at(i).at(j) + _j.at(i);
    return _k;
}

vector<float> Operations::sumVectors(vector<float> _i, vector<float> _j, int d1)
{
    vector<float> _k(d1);
    for(int i = 0; i < d1; i++)
    {
        _k.at(i) = _i.at(i) + _j.at(i);
    }
    return _k;
}

vector<float> Operations::softmax(vector<float> vec, int d1)
{
    float maximum = vec.at(0);
    for(int i = 1; i < d1; i++)
        if(vec.at(i) > maximum)
            maximum = vec.at(i);

    float sum = 0;
    for(int i = 0; i < d1; i++)
        sum += expf(vec.at(i)-maximum);

    for(int i = 0; i < d1; i++)
        vec.at(i) = expf(vec.at(i) - maximum - log(sum));


    return vec;
}

vector<float> Operations::squareVector(vector<float> vec, int d1)
{
    vector<float> vec2(d1);
    for(int i = 0; i < d1; i++)
    {
        vec2.at(i) = vec.at(i) * vec.at(i);
    }
    return vec2;
}

void Operations::save2DVector(vector<vector<float>> vec, int d1, int d2, string file)
{
    ofstream File(file);
    for(int i = 0; i < d1; i++)
    {
        for(int j = 0; j < d2; j++)
        {
            File << vec.at(i).at(j);
            if(j != d2-1)
                File << ",";
        }
        File << "\n";
    }
    File.close();
}


void Operations::save1DVector(vector<float> vec, int d1, string file)
{
    ofstream File(file);
    for(int i = 0; i < d1; i++)
    {
        File << vec.at(i) << ", ";
    }
    File << "\n";
    File.close();
}

vector<float> Operations::standardiseVector(vector<float> vec, int d1)
{
    /*
    float mean = Operations::mean(vec, d1);
    float stdeviation = Operations::stdev(vec, d1);
    for(int i = 0; i < d1; i++)
        vec.at(i) = (vec.at(i) - mean) / stdeviation;
    */
    float maximum = Operations::getMaximum(vec, d1);
    for(int i = 0; i < d1; i++)
        vec.at(i) = (vec.at(i)) / maximum;
    cout << "Maximum " << maximum << endl;
    return vec;
}

float Operations::getMaximum(vector<float> vec, int d1)
{
    float maximum = vec.at(0);
    for(int i = 0; i < d1; i++)
        if(abs(vec.at(i)) > maximum)
            maximum = abs(vec.at(i));
    return maximum;
}

vector<float> Operations::standardiseVector(vector<float> vec, int d1, float maximum)
{
    for(int i = 0; i < d1; i++)
        vec.at(i) = (vec.at(i)) / maximum;
    return vec;
}


float Operations::mean(vector<float> vec, int d1)
{
    float sum = 0;
    for(int i = 0; i < d1; i++)
        sum += vec.at(i);
    return sum/d1;
}

float Operations::stdev(vector<float> vec, int d1)
{
    float mean  = Operations::mean(vec, d1);
    float deviation = 0;
    for(int i = 0; i < d1; i++)
        deviation += ((vec.at(i) - mean) * (vec.at(i) - mean));

    deviation /= (d1-1);
    deviation = sqrt(deviation);
    return deviation;
}
