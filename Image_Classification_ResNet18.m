% Clear workspace and command window
clear; clc;

%% Part1: Image classification

% Load a pre-trained network (e.g., ResNet18, AlexNet, VGG, GoogLeNet, etc.)
net = resnet18;

% Optional: Visualize the network architecture
analyzeNetwork(net);

% Specify the dataset folder (use a relative path instead of personal directory)
Location = './dataset';  % Replace this with your actual dataset folder

% Create an image datastore with images organized in subfolders by class
allImages = imageDatastore(Location, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split the dataset into training (80%), validation (10%), and testing (10%) sets
[training_set, validation_set, testing_set] = splitEachLabel(allImages, 0.80, 0.10, 0.10);

% Convert the network to layer graph for modification
if isa(net, 'SeriesNetwork')
    lgraph = layerGraph(net.Layers);
else
    lgraph = layerGraph(net);
end

% Find the classification and fully connected (or conv) layers to replace
[learnableLayer, classLayer] = findLayersToReplace(lgraph);
[learnableLayer, classLayer];  % Optional line to display info

% Count the number of classes in the training set
categories(training_set.Labels);
numClasses = numel(categories(training_set.Labels));

% Replace the learnable layer with a new one for transfer learning
if isa(learnableLayer, 'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name', 'new_fc', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);

elseif isa(learnableLayer, 'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1, numClasses, ...
        'Name', 'new_conv', ...
        'WeightLearnRateFactor', 10, ...
        'BiasLearnRateFactor', 10);
end

% Replace the layers in the network graph
lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

% Define the input image size for ResNet-18
imageInputSize = [224 224 3];

% Define image augmentation settings for training
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
    'ExecutionEnvironment', 'gpu');  % Use 'cpu' if no GPU is available

% Train the network using the modified architecture and augmented dataset
[net, info] = trainNetwork(augmented_training_set, lgraph, options);

% Use the trained model to classify the test set
[predLabels, predScores] = classify(net, resized_testing_set);

% Plot a confusion matrix comparing predicted and actual labels
plotconfusion(testing_set.Labels, predLabels);

% Save results to workspace
save result

%% Part2: Evaluate_Model_Performance

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

%% Part3: Deep_Phenotype_Extraction

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

% Define the layers to extract features from (default setup for ResNet18, customizable for other models)
layerMap = {
    'res2a_branch2a', 'Conv1';
    'res2a_branch2b', 'Conv2';
    'res2b_branch2a', 'Conv3';
    'res2b_branch2b', 'Conv4';
    'res3a_branch2a', 'Conv5';
    'res3a_branch2b', 'Conv6';
    'res3a_branch1',  'Conv7';
    'res3b_branch2a', 'Conv8';
    'res3b_branch2b', 'Conv9';
    'res4a_branch2a', 'Conv10';
    'res4a_branch2b', 'Conv11';
    'res4a_branch1',  'Conv12';
    'res4b_branch2a', 'Conv13';
    'res4b_branch2b', 'Conv14';
    'res5a_branch2a', 'Conv15';
    'res5a_branch2b', 'Conv16';
    'res5a_branch1',  'Conv17';
    'res5b_branch2a', 'Conv18';
    'res5b_branch2b', 'Conv19';
    'new_fc',             'FC'
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