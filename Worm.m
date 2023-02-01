% Imports the images and stores them
imds = imageDatastore("WormImages");

% Imports the CSV file containing the file names and the labels
groundtruth = readtable("WormData.csv");
% Extracts the labels from the CSV file and stores them as the labels for
% the images in a catagorical array
imds.Labels = categorical(groundtruth.Status);

%imshow(readimage(imds,1))
%imshow(readimage(imds,2))
%imshow(readimage(imds,3))

% Splits the images into training data and test data with 60% of the images
% placed in the training data
[trainImgs,testImgs] = splitEachLabel(imds,0.6,"randomized");

% Crops the training data and test data to the correct size (224 by 224) 
trainds = augmentedImageDatastore([224 224],trainImgs,"ColorPreprocessing","gray2rgb");
testds = augmentedImageDatastore([224 224],testImgs,"ColorPreprocessing","gray2rgb");

% Imports the net we are going to modify in the case googLeNet
net = googlenet;
% Stores the layers of the network to modify later on
lgraph = layerGraph(net);

% Creates a new fully connected layer which gives 2 outputs (Dead or Alive)
newFc = fullyConnectedLayer(2,"Name","new_fc")
% Replace the old layer with the new one we created
lgraph = replaceLayer(lgraph,"loss3-classifier",newFc)
% Creates a new output layer 
newOut = classificationLayer("Name","new_out")
% Replaces the new output layer
lgraph = replaceLayer(lgraph,"output",newOut)

% Network training set up
options = trainingOptions("sgdm","InitialLearnRate", 0.001, "ValidationFrequency",30, "Plots","training-progress")

% Begin training on the modified network now named wormsnet
wormsnet = trainNetwork(trainds,lgraph,options)

% After training we test the network with the test data
preds = classify(wormsnet,testds);

% Gets all the correct values
truetest = testImgs.Labels;
% Calculates number of correct guesses as a decimal
nnz(preds == truetest)/numel(preds)

% Creates the confusion chart for the classifications
confusionchart(truetest,preds);

% Finds all cases in which the predicted value is incorrect
idx = find(preds~=truetest)
if ~isempty(idx)
    imshow(readimage(testImgs,idx(1)))
    title(truetest(idx(1)))
end
