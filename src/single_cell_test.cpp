#include "single_cell_test.h"

void single_cell_test::test(void)
{
    cout << "Training a single LSTM Cell to output a target sequence from input" << endl;

    Cell* cell = new Cell(single_cell_test::_input_length,
                          single_cell_test::_weight_range,
                          single_cell_test::_learning_rate);

    cell->initialiseState(single_cell_test::_c_tp, single_cell_test::_x_t, single_cell_test::_h_tp);
    cell->initialiseBias();
    cell->initialiseWeights();
    cell->setY(single_cell_test::_y_t);

    vector<float> row(single_cell_test::_input_length);
    vector<vector<float>> ct_temporal(single_cell_test::ROUNDS, row);
    vector<vector<float>> ht_temporal(single_cell_test::ROUNDS, row);

    for(int i = 0; i < single_cell_test::ROUNDS; i++)
    {
        cell->setY(single_cell_test::_y_t);
        /** Testing forward propagation **/
        cell->forwardPropagation();

        ct_temporal.at(i) = cell->getct();
        ht_temporal.at(i) = cell->getht();

        /** Testing gradient calculation **/
        cell->gradientCalculation();
        /** Testing updaring weights and bias **/
        cell->updateWeights();

        cell->initialiseState(cell->getct(), single_cell_test::_x_t, cell->getht());
    }

    Operations::display2DVector(ct_temporal, single_cell_test::ROUNDS, single_cell_test::_input_length);

    Operations::save2DVector(ct_temporal, single_cell_test::ROUNDS, single_cell_test::_input_length);
}
