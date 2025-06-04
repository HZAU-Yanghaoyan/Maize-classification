% Clear workspace and command window
clear; clc;

%% Part1: Create Layer Graph
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([224 224 3],"Name","data","Normalization","zscore")
    convolution2dLayer([7 7],64,"Name","conv1","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1")
    reluLayer("Name","conv1_relu")
    maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2a_branch2a")
    reluLayer("Name","res2a_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2a_branch2b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res2a")
    reluLayer("Name","res2a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2b_branch2a")
    reluLayer("Name","res2b_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2b_branch2b_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","res2b");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv_2_6_4_2","BiasLearnRateFactor",0,"Padding","same")
    sigmoidLayer("Name","sigmoid_1_6_4_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_1_6_4")
    convolution2dLayer([1 1],4,"Name","conv_1_6_4","BiasLearnRateFactor",0,"Padding","same")
    reluLayer("Name","relu_1_6_4")
    convolution2dLayer([1 1],64,"Name","conv_2_6_4_1","BiasLearnRateFactor",0,"Padding","same")
    sigmoidLayer("Name","sigmoid_1_6_4_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = reluLayer("Name","res2b_relu_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a")
    reluLayer("Name","res3a_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3a_branch2b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res3a")
    reluLayer("Name","res3a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","res3b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3b_branch2a")
    reluLayer("Name","res3b_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn3b_branch2b_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","res3b");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_1_6_3")
    convolution2dLayer([1 1],8,"Name","conv_1_6_3","BiasLearnRateFactor",0,"Padding","same")
    reluLayer("Name","relu_1_6_3")
    convolution2dLayer([1 1],128,"Name","conv_2_6_3_1","BiasLearnRateFactor",0,"Padding","same")
    sigmoidLayer("Name","sigmoid_1_6_3_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = reluLayer("Name","res3b_relu");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a")
    reluLayer("Name","res4a_branch2a_relu_1")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4a_branch2b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","conv_2_6_3_2","BiasLearnRateFactor",0,"Padding","same")
    sigmoidLayer("Name","sigmoid_1_6_3_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_3_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res4a")
    reluLayer("Name","res4a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","res4b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b_branch2a")
    reluLayer("Name","res4b_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4b_branch2b_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","res4b");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_1_6_2")
    convolution2dLayer([1 1],16,"Name","conv_1_6_2","BiasLearnRateFactor",0,"Padding","same")
    reluLayer("Name","relu_1_6_2")
    convolution2dLayer([1 1],256,"Name","conv_2_6_2_1","BiasLearnRateFactor",0,"Padding","same")
    sigmoidLayer("Name","sigmoid_1_6_2_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_2_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = reluLayer("Name","res4b_relu");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch2a")
    reluLayer("Name","res5a_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5a_branch2b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_3_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_4")
    reluLayer("Name","relu_1_6_1_4")
    convolution2dLayer([1 1],512,"Name","conv_2_6_1_4","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b_2_3")
    reluLayer("Name","res4a_branch2a_relu_2")
    globalAveragePooling2dLayer("Name","gapool_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","conv_2_6_2_2","BiasLearnRateFactor",0,"Padding","same")
    sigmoidLayer("Name","sigmoid_1_6_2_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_2_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    reluLayer("Name","relu_1_6_1_3")
    convolution2dLayer([1 1],512,"Name","conv_2_6_1_3","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b_2_2")
    reluLayer("Name","res5a_relu_2")
    globalAveragePooling2dLayer("Name","gapool_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_4_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_4_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_5")
    reluLayer("Name","relu_1_6_1_5")
    convolution2dLayer([1 1],512,"Name","conv_2_6_1_5","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b_2_4")
    reluLayer("Name","res2b_relu_2")
    globalAveragePooling2dLayer("Name","gapool_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res5a")
    reluLayer("Name","res5a_relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","res5b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5b_branch2a")
    reluLayer("Name","res5b_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn5b_branch2b_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","res5b");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","conv_1_6_1_2","BiasLearnRateFactor",0,"Padding","same")
    sigmoidLayer("Name","sigmoid_1_6_1_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_1_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_1_6_1")
    convolution2dLayer([1 1],32,"Name","conv_1_6_1_1","BiasLearnRateFactor",0,"Padding","same")
    reluLayer("Name","relu_1_6_1_1")
    convolution2dLayer([1 1],512,"Name","conv_2_6_1_1","BiasLearnRateFactor",0,"Padding","same")
    sigmoidLayer("Name","sigmoid_1_6_1_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_6_1_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    reluLayer("Name","relu_1_6_1_2")
    convolution2dLayer([1 1],512,"Name","conv_2_6_1_2","BiasLearnRateFactor",0,"Padding","same")
    batchNormalizationLayer("Name","bn5b_branch2b_2_1")
    reluLayer("Name","res5b_relu_2")
    globalAveragePooling2dLayer("Name","gapool_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(4,"Name","addition_1")
    reluLayer("Name","res5b_relu_1")
    globalAveragePooling2dLayer("Name","pool5")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","prob")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

clear tempLayers;

lgraph = connectLayers(lgraph,"pool1","res2a_branch2a");
lgraph = connectLayers(lgraph,"pool1","res2a/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2b","res2a/in1");
lgraph = connectLayers(lgraph,"res2a_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"res2a_relu","res2b/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2b_1","res2b/in1");
lgraph = connectLayers(lgraph,"res2b","conv_2_6_4_2");
lgraph = connectLayers(lgraph,"res2b","gapool_1_6_4");
lgraph = connectLayers(lgraph,"res2b","res2b_relu_1");
lgraph = connectLayers(lgraph,"res2b","multiplication_6_4_2/in2");
lgraph = connectLayers(lgraph,"res2b","multiplication_6_4_1/in2");
lgraph = connectLayers(lgraph,"res2b_relu_1","res3a_branch1");
lgraph = connectLayers(lgraph,"res2b_relu_1","res3a_branch2a");
lgraph = connectLayers(lgraph,"bn3a_branch1","res3a/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_4_2","multiplication_6_4_2/in1");
lgraph = connectLayers(lgraph,"bn3a_branch2b","res3a/in1");
lgraph = connectLayers(lgraph,"res3a_relu","res3b_branch2a");
lgraph = connectLayers(lgraph,"res3a_relu","res3b/in2");
lgraph = connectLayers(lgraph,"bn3b_branch2b_1","res3b/in1");
lgraph = connectLayers(lgraph,"res3b","gapool_1_6_3");
lgraph = connectLayers(lgraph,"res3b","res3b_relu");
lgraph = connectLayers(lgraph,"res3b","conv_2_6_3_2");
lgraph = connectLayers(lgraph,"res3b","multiplication_6_3_2/in2");
lgraph = connectLayers(lgraph,"res3b","multiplication_6_3_1/in2");
lgraph = connectLayers(lgraph,"res3b_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"res3b_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"bn4a_branch1","res4a/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_3_2","multiplication_6_3_2/in1");
lgraph = connectLayers(lgraph,"multiplication_6_3_2","addition_4/in2");
lgraph = connectLayers(lgraph,"bn4a_branch2b","res4a/in1");
lgraph = connectLayers(lgraph,"res4a_relu","res4b_branch2a");
lgraph = connectLayers(lgraph,"res4a_relu","res4b/in2");
lgraph = connectLayers(lgraph,"bn4b_branch2b_1","res4b/in1");
lgraph = connectLayers(lgraph,"res4b","gapool_1_6_2");
lgraph = connectLayers(lgraph,"res4b","multiplication_6_2_1/in2");
lgraph = connectLayers(lgraph,"res4b","res4b_relu");
lgraph = connectLayers(lgraph,"res4b","conv_2_6_2_2");
lgraph = connectLayers(lgraph,"res4b","multiplication_6_2_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_2_1","multiplication_6_2_1/in1");
lgraph = connectLayers(lgraph,"multiplication_6_2_1","addition_3/in1");
lgraph = connectLayers(lgraph,"res4b_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"res4b_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"bn5a_branch1","res5a/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_3_1","multiplication_6_3_1/in1");
lgraph = connectLayers(lgraph,"multiplication_6_3_1","addition_4/in1");
lgraph = connectLayers(lgraph,"sigmoid_1_6_2_2","multiplication_6_2_2/in1");
lgraph = connectLayers(lgraph,"multiplication_6_2_2","addition_3/in2");
lgraph = connectLayers(lgraph,"gapool_2","addition_1/in2");
lgraph = connectLayers(lgraph,"gapool_3","addition_1/in3");
lgraph = connectLayers(lgraph,"multiplication_6_4_2","addition_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_4_1","multiplication_6_4_1/in1");
lgraph = connectLayers(lgraph,"multiplication_6_4_1","addition_5/in1");
lgraph = connectLayers(lgraph,"gapool_4","addition_1/in4");
lgraph = connectLayers(lgraph,"bn5a_branch2b","res5a/in1");
lgraph = connectLayers(lgraph,"res5a_relu_1","res5b_branch2a");
lgraph = connectLayers(lgraph,"res5a_relu_1","res5b/in2");
lgraph = connectLayers(lgraph,"bn5b_branch2b_1","res5b/in1");
lgraph = connectLayers(lgraph,"res5b","conv_1_6_1_2");
lgraph = connectLayers(lgraph,"res5b","multiplication_6_1_2/in2");
lgraph = connectLayers(lgraph,"res5b","gapool_1_6_1");
lgraph = connectLayers(lgraph,"res5b","multiplication_6_1_1/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_1_2","multiplication_6_1_2/in1");
lgraph = connectLayers(lgraph,"multiplication_6_1_2","addition_2/in2");
lgraph = connectLayers(lgraph,"sigmoid_1_6_1_1","multiplication_6_1_1/in1");
lgraph = connectLayers(lgraph,"multiplication_6_1_1","addition_2/in1");
lgraph = connectLayers(lgraph,"gapool_1","addition_1/in1");

plot(lgraph);

%% Part2: Image classification

% Load a CNN model designed by the author
net = lgraph;

% Optional: Visualize the custom network architecture
analyzeNetwork(net);

% Specify the dataset folder (use relative path to avoid exposing personal info)
Location = './dataset';  % Replace with your actual dataset folder

% Create an image datastore with images organized in subfolders by class
allImages = imageDatastore(Location, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the dataset into training (80%), validation (10%), and testing (10%) sets
[training_set, validation_set, testing_set] = splitEachLabel(allImages, 0.80, 0.10, 0.10);

% Define the input image size expected by the network
imageInputSize = [224 224 3];

% Define data augmentation settings for the training set
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3]);

% Apply augmentation to the training set
augmented_training_set = augmentedImageDatastore(imageInputSize, training_set, ...
    'DataAugmentation', imageAugmenter);

% Resize validation and testing sets (no augmentation)
resized_validation_set = augmentedImageDatastore(imageInputSize, validation_set);
resized_testing_set = augmentedImageDatastore(imageInputSize, testing_set);

% Define training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', resized_validation_set, ...
    'ValidationFrequency', 12, ...
    'Verbose', false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'gpu');  % Use 'cpu' if GPU is not available

% Train the custom network using the augmented training set
[net, info] = trainNetwork(augmented_training_set, lgraph, options);

% Classify the test set using the trained model
[predLabels, predScores] = classify(net, resized_testing_set);

% Plot confusion matrix comparing predicted and actual labels
plotconfusion(testing_set.Labels, predLabels);

% Save results to a .mat file
save result

%% Part3: Evaluate_Model_Performance

% Extract true labels and predicted scores from the testing set
trueLabels = testing_set.Labels;
scores = predScores;

% === Evaluate performance when class 1 (e.g., "HN") is considered the positive class ===
classNames = categories(trueLabels);
disp(['Order of actual class labels: ', classNames{1}, ' (column 1) and ', classNames{2}, ' (column 2)']);

% Compute ROC curve and AUC for class 1
[fprHN, tprHN, ~, aucHN] = perfcurve(trueLabels, scores(:,1), classNames{1}); 

positiveClass = classNames{1};
y_true = trueLabels == positiveClass;
y_pred = predLabels == positiveClass;

% Compute confusion matrix components
TP = sum(y_true & y_pred);
FP = sum(~y_true & y_pred);
TN = sum(~y_true & ~y_pred);
FN = sum(y_true & ~y_pred);

% Calculate common evaluation metrics
Accuracy    = 100 * (TP + TN) / (TP + TN + FP + FN);
Precision   = 100 * TP / (TP + FP);
Recall      = 100 * TP / (TP + FN);         % Also known as Sensitivity
Specificity = 100 * TN / (TN + FP);
F1_score    = 2 * (Precision * Recall) / (Precision + Recall);
AUC_HN      = aucHN;

% Display performance metrics
fprintf('Evaluation for class "%s" (as Positive):\n', positiveClass);
fprintf('Accuracy: %.15f\n', Accuracy);
fprintf('Precision: %.15f\n', Precision);
fprintf('Recall (Sensitivity): %.15f\n', Recall);
fprintf('Specificity: %.15f\n', Specificity);
fprintf('F1-score: %.15f\n', F1_score);
fprintf('AUC: %.15f\n\n', AUC_HN);

% === Evaluate performance when class 2 (e.g., "LN") is considered the positive class ===
[fprLN, tprLN, ~, aucLN] = perfcurve(trueLabels, scores(:,2), classNames{2}); 

positiveClass = classNames{2};
y_true = trueLabels == positiveClass;
y_pred = predLabels == positiveClass;

% Compute confusion matrix components
TP = sum(y_true & y_pred);
FP = sum(~y_true & y_pred);
TN = sum(~y_true & ~y_pred);
FN = sum(y_true & ~y_pred);

% Calculate evaluation metrics
Accuracy    = 100 * (TP + TN) / (TP + TN + FP + FN);
Precision   = 100 * TP / (TP + FP);
Recall      = 100 * TP / (TP + FN);
Specificity = 100 * TN / (TN + FP);
F1_score    = 2 * (Precision * Recall) / (Precision + Recall);
AUC_LN      = aucLN;

% Display performance metrics
fprintf('Evaluation for class "%s" (as Positive):\n', positiveClass);
fprintf('Precision: %.15f\n', Precision);
fprintf('Recall (Sensitivity): %.15f\n', Recall);
fprintf('Specificity: %.15f\n', Specificity);
fprintf('F1-score: %.15f\n', F1_score);
fprintf('AUC: %.15f\n', AUC_LN);

%% Part4: Deep_Phenotype_Extraction

% Define image folder (use relative or anonymized path)
imageFolder = './dataset';

% Load images with folder names as labels
allImages = imageDatastore(imageFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Load model (assumes variable `net` is a function handle or model object)
net = net();

% Prepare image preprocessing for input size and color conversion
inputSize = net.Layers(1).InputSize;
augmentedData = augmentedImageDatastore(inputSize, allImages, ...
    'ColorPreprocessing', 'gray2rgb');

% Define the layers to extract features from (default setup for OurModel, customizable for other models)
layerMap = {
    'res2a_branch2a', 'Conv1';
    'res2a_branch2b', 'Conv2';
    'res2b_branch2a', 'Conv3';
    'res2b_branch2b', 'Conv4';
    'res3a_branch2a', 'Conv5';
    'conv_2_6_4_2',   'Conv6';
    'conv_1_6_4',     'Conv7';
    'res3a_branch1',  'Conv8';
    'res3a_branch2b', 'Conv9';
    'res3b_branch2a', 'Conv10';
    'res3b_branch2b', 'Conv11';
    'res4a_branch2a', 'Conv12';
    'res4a_branch2b', 'Conv13';
    'res4a_branch1',  'Conv14';
    'conv_1_6_3',     'Conv15';
    'conv_2_6_3_1',   'Conv16';
    'conv_2_6_3_2',   'Conv17';
    'res4b_branch2a', 'Conv18';
    'res4b_branch2b', 'Conv19';
    'conv_1_6_2',     'Conv20';
    'conv_2_6_2_2',   'Conv21';
    'res5a_branch1',  'Conv22';
    'res5a_branch2a', 'Conv23';
    'res5a_branch2b', 'Conv24';
    'res5b_branch2a', 'Conv25';
    'conv_2_6_2_1',   'Conv26';
    'res5b_branch2b', 'Conv27';
    'conv_2_6_1_3',   'Conv28';
    'conv_1_6_1_2',   'Conv29';
    'conv_1_6_1_1',   'Conv30';
    'conv_2_6_1_1',   'Conv31';
    'conv_2_6_1_2',   'Conv32';
    'conv_2_6_4_1',   'Conv33';
    'fc',             'FC'
};

% Initialize container for t-SNE results
tsneResults = [];

fprintf('Processing %d layers...\n', size(layerMap, 1));
tic;

for i = 1:size(layerMap, 1)
    layerName = layerMap{i, 1};
    outputName = layerMap{i, 2};

    % Extract deep features from the specified layer
    features = activations(net, augmentedData, layerName, ...
        'MiniBatchSize', 32, ...
        'OutputAs', 'rows');

    % Apply t-SNE for 2D dimensionality reduction
    Y = tsne(features);

    % Append result to combined table
    if isempty(tsneResults)
        tsneResults = array2table(Y, 'VariableNames', {'Var1', 'Var2'});
    else
        tsneResults = [tsneResults, ...
            array2table(Y, 'VariableNames', ...
            {['Var' num2str(i*2-1)], ['Var' num2str(i*2)]})];
    end

    fprintf('Layer %d (%s) completed in %.2f seconds\n', i, outputName, toc);
    tic;
end

% Save all t-SNE features to Excel
writetable(tsneResults, 'Deep_Phenotype.xlsx');