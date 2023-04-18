#include "network_test.h"
/**
    This test is intended to train a LSTM network on the input/output of a rayleigh channel.
    (real values of channel).
**/
void network_test::test(void)
{
    network_test::net = new Network(network_test::_input_length,
                                  network_test::_weight_range,
                                  network_test::_learning_rate,
                                  network_test::_network_size);

    /** (REAL) LS estimate of channel **/
    vector<float> LS_EST = {    -1.51120275010916000,	-2.04978774403636000,
                                -1.86959755782467000,   -1.28733708601674000,
                                -0.61971122315524300,	-0.14377093011389100,
                                0.022048566909344800,	-0.13848543119575400,
                                -0.39141124986415700,   -0.49386791995177700,
                                -0.41759166767354000,	-0.21386033162764900,
                                -0.03752100458637800,	-0.02388244855140030,
                                -0.12653084826790100,   -0.25309853150552400,
                                -0.30912497538824300,	-0.23405155789833900,
                                -0.10502011109699700,	-0.01996145279267770,
                                -0.02226188970404040,   -0.11725570681897900,
                                -0.21261361329212200,	-0.21794283471779400,
                                -0.14826023828236900,	-0.05017635940264710,
                                0.008733235174919100,   -0.02211203113601590,
                                -0.10434774295170800,	-0.17853723838995900,
                                -0.18989986154485700,	-0.10970779864006500,
                                -0.00878859844129037,   0.032703922350595900,
                                -0.00313376216390587,	-0.10040833092472600,
                                -0.17028936583489300,	-0.13543326525558300,
                                -0.03738079041746790,   0.053770532780937300,
                                0.072580482639246600,	-0.01476863074222770,
                                -0.12026167425850800,	-0.14697029531602000,
                                -0.08414608637146880,   0.041043082876727500,
                                0.135944178813579000,	0.119007848453417000,
                                0.018584625244420800,	-0.09979946704941600,
                                -0.13181632983469800,   -0.00657875501899657,
                                0.174234828656850000,	0.279889815067819000,
                                0.235992525385426000,	0.032137831277234700,
                                -0.14246612090153200,   -0.09543037611829610,
                                0.178743980681268000,	0.600595040676894000,
                                0.937707639528192000,	0.919428299014739000,
                                0.275103540916114000,   -1.26592011298811000};

    float maximum = Operations::getMaximum(LS_EST, 64);
    LS_EST = Operations::standardiseVector(LS_EST, 64, maximum);
    cout << "Maximum of the LS Estimation is " << maximum << endl;

    Operations::save1DVector(LS_EST, 64, "LS_VECTOR.csv");

    vector<vector<float>> targets(32, vector<float>(2));
    targets.at(0)  = Operations::standardiseVector({-1.51120275010916000,  -2.04978774403636000}, 2, maximum);
    targets.at(1)  = Operations::standardiseVector({-1.86959755782467000,  -1.28733708601674000}, 2, maximum);
    targets.at(2)  = Operations::standardiseVector({-0.61971122315524300,  -0.14377093011389100}, 2, maximum);
    targets.at(3)  = Operations::standardiseVector({0.022048566909344800,  -0.13848543119575400}, 2, maximum);
    targets.at(4)  = Operations::standardiseVector({-0.39141124986415700,  -0.49386791995177700}, 2, maximum);
    targets.at(5)  = Operations::standardiseVector({-0.41759166767354000,  -0.21386033162764900}, 2, maximum);
    targets.at(6)  = Operations::standardiseVector({-0.03752100458637800,  -0.02388244855140030}, 2, maximum);
    targets.at(7)  = Operations::standardiseVector({-0.12653084826790100,  -0.25309853150552400}, 2, maximum);
    targets.at(8)  = Operations::standardiseVector({-0.30912497538824300,  -0.23405155789833900}, 2, maximum);
    targets.at(9)  = Operations::standardiseVector({-0.10502011109699700,  -0.01996145279267770}, 2, maximum);
    targets.at(10) = Operations::standardiseVector({-0.02226188970404040,  -0.11725570681897900}, 2, maximum);
    targets.at(11) = Operations::standardiseVector({-0.21261361329212200,  -0.21794283471779400}, 2, maximum);
    targets.at(12) = Operations::standardiseVector({-0.14826023828236900,  -0.05017635940264710}, 2, maximum);
    targets.at(13) = Operations::standardiseVector({0.008733235174919100,  -0.02211203113601590}, 2, maximum);
    targets.at(14) = Operations::standardiseVector({-0.10434774295170800,  -0.17853723838995900}, 2, maximum);
    targets.at(15) = Operations::standardiseVector({-0.18989986154485700,  -0.10970779864006500}, 2, maximum);
    targets.at(16) = Operations::standardiseVector({-0.00878859844129037,  0.032703922350595900}, 2, maximum);
    targets.at(17) = Operations::standardiseVector({-0.00313376216390587,  -0.10040833092472600}, 2, maximum);
    targets.at(18) = Operations::standardiseVector({-0.17028936583489300,  -0.13543326525558300}, 2, maximum);
    targets.at(19) = Operations::standardiseVector({-0.03738079041746790,  0.053770532780937300}, 2, maximum);
    targets.at(20) = Operations::standardiseVector({0.072580482639246600,  -0.01476863074222770}, 2, maximum);
    targets.at(21) = Operations::standardiseVector({-0.12026167425850800,  -0.14697029531602000}, 2, maximum);
    targets.at(22) = Operations::standardiseVector({-0.08414608637146880,  0.041043082876727500}, 2, maximum);
    targets.at(23) = Operations::standardiseVector({0.135944178813579000,  0.119007848453417000}, 2, maximum);
    targets.at(24) = Operations::standardiseVector({0.018584625244420800,  -0.09979946704941600}, 2, maximum);
    targets.at(25) = Operations::standardiseVector({-0.13181632983469800,  -0.00657875501899657}, 2, maximum);
    targets.at(26) = Operations::standardiseVector({0.174234828656850000,  0.279889815067819000}, 2, maximum);
    targets.at(27) = Operations::standardiseVector({0.235992525385426000,  0.032137831277234700}, 2, maximum);
    targets.at(28) = Operations::standardiseVector({-0.14246612090153200,  -0.09543037611829610}, 2, maximum);
    targets.at(29) = Operations::standardiseVector({0.178743980681268000,  0.600595040676894000}, 2, maximum);
    targets.at(30) = Operations::standardiseVector({0.937707639528192000,  0.919428299014739000}, 2, maximum);
    targets.at(31) = Operations::standardiseVector({0.275103540916114000,  -1.26592011298811000}, 2, maximum);

    network_test::net->x0 = {0,0};
    network_test::net->h0 = {0,0};
    network_test::net->c0 = {0,0};
    network_test::net->targets = targets;

    /** Train network **/
    network_test::net->trainNetwork(network_test::ROUNDS);

    Operations::save2DVector(network_test::net->predictions, 32, 2, "predictions_VECTOR.csv");

    /** Validate network **/
    network_test::net->validateNetwork();
    Operations::save2DVector(network_test::net->predictions, 32, 2, "validation_VECTOR.csv");

    /** Test network **/
    network_test::net->testNetwork({0,0});
    Operations::save2DVector(network_test::net->predictions, 32, 2, "testing_VECTOR.csv");

    /** Now network changes slightly **/
    vector<float> LS_EST2 = {   -1.51120275010916000,	-2.04978774403636000,//1
                                -1.86959755782467000,   -1.28733708601674000,//2
                                -0.61971122315524300,	-0.14377093011389100,//3
                                0.022048566909344800,	-0.13848543119575400,//4
                                -0.39141124986415700,   -0.49386791995177700,//5
                                -0.41759166767354000,	-0.21386033162764900,//6
                                -0.03752100458637800,	-0.02388244855140030,//7
                                -0.12653084826790100,   -0.25309853150552400,//8
                                -0.30912497538824300,	-0.23405155789833900,//9
                                -0.10502011109699700,	-0.01996145279267770,//10
                                -0.02226188970404040,   -0.11725570681897900,//11
                                -0.21261361329212200,	-0.21794283471779400,//12
                                -0.14826023828236900,	-0.05017635940264710,//13
                                0.008733235174919100,   -0.02211203113601590,//14
                                -0.10434774295170800,	-0.17853723838995900,//15
                                -0.18989986154485700,	-0.10970779864006500,//16
                                -0.00878859844129037,   0.032703922350595900,//17
                                -0.00313376216390587,	-0.10040833092472600,//17
                                -0.17028936583489300,	-0.13543326525558300,//19
                                -0.03738079041746790,   0.053770532780937300,//20
                                -0.13181632983469800,   -0.00657875501899657,//21
                                0.174234828656850000,	0.279889815067819000,//22
                                0.235992525385426000,	0.032137831277234700,//23
                                -0.14246612090153200,   -0.09543037611829610,//24
                                0.178743980681268000,	0.600595040676894000,//25
                                0.937707639528192000,	0.919428299014739000,//26
                                0.275103540916114000,   -1.26592011298811000,//27
                                0.072580482639246600,	-0.01476863074222770,//28
                                -0.12026167425850800,	-0.14697029531602000,//29
                                -0.08414608637146880,   0.041043082876727500,//30
                                0.135944178813579000,	0.119007848453417000,//31
                                0.018584625244420800,	-0.09979946704941600};//32

    LS_EST2 = Operations::standardiseVector(LS_EST2, 64, maximum);
    Operations::save1DVector(LS_EST2, 64, "LS2_VECTOR.csv");

    targets.at(0)  = Operations::standardiseVector({-1.51120275010916000,  -2.04978774403636000}, 2, maximum);//1
    targets.at(1)  = Operations::standardiseVector({-1.86959755782467000,  -1.28733708601674000}, 2, maximum);//2
    targets.at(2)  = Operations::standardiseVector({-0.61971122315524300,  -0.14377093011389100}, 2, maximum);//3
    targets.at(3)  = Operations::standardiseVector({0.022048566909344800,  -0.13848543119575400}, 2, maximum);//4
    targets.at(4)  = Operations::standardiseVector({-0.39141124986415700,  -0.49386791995177700}, 2, maximum);//5
    targets.at(5)  = Operations::standardiseVector({-0.41759166767354000,  -0.21386033162764900}, 2, maximum);//6
    targets.at(6)  = Operations::standardiseVector({-0.03752100458637800,  -0.02388244855140030}, 2, maximum);//7
    targets.at(7)  = Operations::standardiseVector({-0.12653084826790100,  -0.25309853150552400}, 2, maximum);//8
    targets.at(8)  = Operations::standardiseVector({-0.30912497538824300,  -0.23405155789833900}, 2, maximum);//9
    targets.at(9)  = Operations::standardiseVector({-0.10502011109699700,  -0.01996145279267770}, 2, maximum);//10
    targets.at(10) = Operations::standardiseVector({-0.02226188970404040,  -0.11725570681897900}, 2, maximum);//11
    targets.at(11) = Operations::standardiseVector({-0.21261361329212200,  -0.21794283471779400}, 2, maximum);//12
    targets.at(12) = Operations::standardiseVector({-0.14826023828236900,  -0.05017635940264710}, 2, maximum);//13
    targets.at(13) = Operations::standardiseVector({0.008733235174919100,  -0.02211203113601590}, 2, maximum);//14
    targets.at(14) = Operations::standardiseVector({-0.10434774295170800,  -0.17853723838995900}, 2, maximum);//15
    targets.at(15) = Operations::standardiseVector({-0.18989986154485700,  -0.10970779864006500}, 2, maximum);//16
    targets.at(16) = Operations::standardiseVector({-0.00878859844129037,  0.032703922350595900}, 2, maximum);//17
    targets.at(17) = Operations::standardiseVector({-0.00313376216390587,  -0.10040833092472600}, 2, maximum);//18
    targets.at(18) = Operations::standardiseVector({-0.17028936583489300,  -0.13543326525558300}, 2, maximum);//19
    targets.at(19) = Operations::standardiseVector({-0.03738079041746790,  0.053770532780937300}, 2, maximum);//20
    targets.at(20) = Operations::standardiseVector({-0.13181632983469800,  -0.00657875501899657}, 2, maximum);//21
    targets.at(21) = Operations::standardiseVector({0.174234828656850000,  0.279889815067819000}, 2, maximum);//22
    targets.at(22) = Operations::standardiseVector({0.235992525385426000,  0.032137831277234700}, 2, maximum);//23
    targets.at(23) = Operations::standardiseVector({-0.14246612090153200,  -0.09543037611829610}, 2, maximum);//24
    targets.at(24) = Operations::standardiseVector({0.178743980681268000,  0.600595040676894000}, 2, maximum);//25
    targets.at(25) = Operations::standardiseVector({0.937707639528192000,  0.919428299014739000}, 2, maximum);//26
    targets.at(26) = Operations::standardiseVector({0.275103540916114000,  -1.26592011298811000}, 2, maximum);//27
    targets.at(27) = Operations::standardiseVector({0.072580482639246600,  -0.01476863074222770}, 2, maximum);//28
    targets.at(28) = Operations::standardiseVector({-0.12026167425850800,  -0.14697029531602000}, 2, maximum);//29
    targets.at(29) = Operations::standardiseVector({-0.08414608637146880,  0.041043082876727500}, 2, maximum);//30
    targets.at(30) = Operations::standardiseVector({0.135944178813579000,  0.119007848453417000}, 2, maximum);//31
    targets.at(31) = Operations::standardiseVector({0.018584625244420800,  -0.09979946704941600}, 2, maximum);//32

     /** Train network **/
    network_test::net->targets = targets;
    network_test::net->trainNetwork(50);
    Operations::save2DVector(network_test::net->predictions, 32, 2, "predictions2_VECTOR.csv");

}
