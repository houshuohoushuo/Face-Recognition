clear;
maleFolder = './faces/male';
maleImgSetVector = imageSet(maleFolder,'recursive');
maleCount = 0;
for set = maleImgSetVector
    maleCount = maleCount + set.Count;
end

femaleFolder = './faces/female';
femaleImgSetVector = imageSet(femaleFolder,'recursive');
femaleCount = 0;
for set = femaleImgSetVector
    femaleCount = femaleCount + set.Count;
end

img = imread(set.ImageLocation{1});

faces = zeros(size(img,1),size(img,2),maleCount+femaleCount);
labels = zeros(maleCount+femaleCount,1);

counter = 1;
for set = maleImgSetVector
    for i = 1 : set.Count
        img = imread(set.ImageLocation{i});
        gray = (size(img,3) == 1);
        if ~gray
            img = rgb2gray(img);
        end
        img = double(img);
        faces(:,:,counter) = img;
        labels(counter) = 0;
        counter = counter + 1;
    end
end

for set = femaleImgSetVector
    for i = 1 : set.Count
        img = imread(set.ImageLocation{i});
        gray = (size(img,3) == 1);
        if ~gray
            img = rgb2gray(img);
        end
        img = double(img);
        faces(:,:,counter) = img;
        labels(counter) = 1;
        counter = counter + 1;
    end
end


cellSize = [2 2];
img = faces(:,:,1);
[featureVector,hogVisualization] = extractHOGFeatures(img,'CellSize',cellSize);
hogFeatureSize = length(featureVector);

% figure;
% imagesc(img);
% hold on;
% plot(hogVisualization);

numTrainingImages = maleCount + femaleCount;
features  = zeros(numTrainingImages,hogFeatureSize,'single');

% Extract HOG features from each training image.
for i = 1:numTrainingImages
    img = faces(:,:,i);
    features(i,:) = extractHOGFeatures(img,'CellSize',cellSize);
end

% Train a multiclass classifier
classifier = fitcecoc(features, labels);
imSize = size(faces(:,:,1));
save('classifier.mat','classifier','imSize','hogFeatureSize');