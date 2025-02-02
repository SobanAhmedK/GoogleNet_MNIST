
net = googlenet;



digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');


[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');


inputSize = [224 224]; % GoogLeNet input size
augmentedTrainingData = augmentedImageDatastore(inputSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augmentedValidationData = augmentedImageDatastore(inputSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');

lgraph = layerGraph(net);


fcLayer = lgraph.Layers(end-2); % Fully connected layer
outputLayer = lgraph.Layers(end); % Output layer

newFCLayer = fullyConnectedLayer(10, 'Name', 'new_fc');
lgraph = replaceLayer(lgraph, fcLayer.Name, newFCLayer);

newOutputLayer = classificationLayer('Name', 'new_output');
lgraph = replaceLayer(lgraph, outputLayer.Name, newOutputLayer);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 30, ...         % Adjust batch size as needed
    'MaxEpochs', 15, ...             % Keep epochs at 10
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedValidationData, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto'); % Use GPU if available


netTransfer = trainNetwork(augmentedTrainingData, lgraph, options);


save('GoogLeNet_MNIST.mat', 'netTransfer');

YPred = classify(netTransfer, augmentedValidationData);


YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
disp(['Validation Accuracy: ', num2str(accuracy * 100), '%']);

