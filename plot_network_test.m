clear all;
close all;


predictions = csvread('predictions_VECTOR.csv'); 
predictions = predictions(:,1);

LS = csvread('LS_VECTOR.csv'); 
LS = LS(1:2:64);

LS2 = csvread('LS2_VECTOR.csv'); 
LS2 = LS2(1:2:64);

validation = csvread('validation_VECTOR.csv'); 
validation = validation(:,1);

testing = csvread('testing_VECTOR.csv'); 
testing = testing(:,1);

prediction2 = csvread('predictions2_VECTOR.csv'); 
prediction2 = prediction2(:,1);

figure();
plot(predictions, '--');
hold on;
plot(LS, 'o');
plot(LS2, '-s');
plot(validation, '-x');
plot(testing, '-^');
plot(prediction2, '-s');
legend( "Trained output", ...
        "LS Estimation", ...
        "LS Estimation 2",...
        "Validation", ...
        "Testing", ...
        "Prediction 2");