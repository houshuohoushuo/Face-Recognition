clear;
gender = {'male','female'};
load('classifier.mat');

faceDetector = vision.CascadeObjectDetector();

boxTracker = MultiObjectTrackerKLT;

v = VideoReader('Parliament.mp4');

videoFrame = read(v,1);
frameSize = size(videoFrame);


% % Create the webcam object.
% cam = webcam();
% % Capture one frame to get its size.
% videoFrame = snapshot(cam);
% frameSize = size(videoFrame);

% Create the video player object.
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);

runLoop = true;
frameNum = 0;

v = VideoReader('Parliament.mp4');

while hasFrame(v) && runLoop
%     Get the next frame.
    videoFrame = imresize(readFrame(v),0.5);
%     videoFrame = snapshot(cam);
    videoFrameGray = rgb2gray(videoFrame);
    
    if mod(frameNum, 10) == 0
        % detect new face ecery 10 frames
        % down sample video frame for faster speed
        bbox = 2 * faceDetector.step(imresize(videoFrameGray, 0.5));        
        if ~isempty(bbox)
            boxTracker.addDetections(videoFrameGray,bbox);
        end
        
    else
        % track boxes
        if size(boxTracker.Bboxes) ~= 0
            boxTracker.track(videoFrameGray);
        end
    end
    
    if size(boxTracker.Bboxes) ~= 0
        genders = zeros(size(boxTracker.Bboxes,1),1);
        for i = 1 : size(boxTracker.Bboxes,1)
            y = round(boxTracker.Bboxes(i,2));
            x = round(boxTracker.Bboxes(i,1));
            w = round(boxTracker.Bboxes(i,3));
            h = round(boxTracker.Bboxes(i,4));
            %get face patches and predict their gender
            patch = double(videoFrameGray(y:y+h,x:x+w));
            genders(i) = PredictFace(patch,classifier,imSize);
        end
        %display bounding boxes and their corresponding genders
        displayFrame = insertObjectAnnotation(videoFrame, 'rectangle',...
            boxTracker.Bboxes, gender(genders(:)+1));
        videoPlayer.step(displayFrame);
    else
        videoPlayer.step(videoFrame);
    end
    
    runLoop = isOpen(videoPlayer);
    
    frameNum = frameNum + 1;
end

clear;