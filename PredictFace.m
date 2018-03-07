function predictedLabel = PredictFace(test_image,classifier,imSize)
cellSize = [2 2];
img = imresize(test_image,imSize);
% extract HOG features
features = extractHOGFeatures(img,'CellSize',cellSize);
% predict using the trained classifier
predictedLabel = predict(classifier, features);
end