clear;
load('classifier.mat');
maleFolder = './faces/test/male';
maleImgSetVector = imageSet(maleFolder,'recursive');
maleCount = 0;
for set = maleImgSetVector
    maleCount = maleCount + set.Count;
end

femaleFolder = './faces/test/female';
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

numImages = maleCount + femaleCount;
predictedLabels = zeros(numImages,1);

for i = 1:numImages
    img = faces(:,:,i);
    predictedLabels(i) = PredictFace(img,classifier,imSize);
end

accuracy = sum(labels == predictedLabels)/numImages;
fprintf('accuracy = %f \n',accuracy);