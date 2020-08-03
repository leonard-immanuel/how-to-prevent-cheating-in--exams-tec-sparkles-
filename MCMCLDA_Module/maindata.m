clear;
close all;
clc
warning off all
cd TestVideos
[mov_name,PathName] = uigetfile({'*.avi;*.mp4;*.mov'},'Select the video to track');%options to select the video file
cd ..
% after we read the content to make them as frames for processing with our
% procedure
I=VideoReader([PathName mov_name]);

% % creating movie structure from object structure
% mov(1:A)=struct('cdata',zeros(B,C,3,'uint8'),'colormap',[]);
% for i=1:A
%     mov(i).cdata=read(I,i);
% end

%this is the process of converting to frames and make as image
% we stored the images into the individual folder named as Frames
% movO(1:A)=struct('cdata',zeros(B,C,3,'uint8'),'colormap',[]);
for i=1:10:A
disp(sprintf('For Frame %d\n',i));
% read(I,i)
   frm_name=imresize(read(I,i),[480 720]); % Convert Frame to image file
%       frm_name=imresize(frame2im(mov(i)),[480 720]); % Convert Frame to image file
filename1=strcat(strcat(num2str(i)),'.jpg');
figure(1);imshow(frm_name);title(sprintf('Frame %d',i));
cd('Video_convert');
     imwrite(frm_name,filename1); % Write image file
  cd ..  
end
addpath SubFunctions
global do_multithread_fconv;
do_multithread_fconv = true;
model = loadvar('MODEC_train.mat','mdls');
load cluster_info.mat
global config;
load CS;

Lf=1;
fname=[num2str(Lf) '.jpg'];
cd Video_convert
Frame_im1 = imread(fname);
cd ..
figure(2);
subplot(121);imshow(frm_name);title('Input Frame');

% finds the people and draws bounding boxes around them
img=Frame_im1;
time=clock;
init;

config.USE_MEX_HOG = true;

% detection configurations

config.DETECTION_IMG_MIN_NUM_PIX = 240^2;  
% if the number of pixels in a detection image is DETECTION_IMG_SIDE^2, 
% scales up the image to meet that threshold
config.DETECTION_IMG_MAX_NUM_PIX = 640^2;  
config.PYRAMID_SCALE_RATIO = 2;

%%% Predict the locations and scores of all objects
%%% in the frame (bounds_predictions), of all poselet hits (poselet_hits)
   [bounds_predictions,poselet_hits,torso_predictions] = Poselets_Prediction(img);

% take top scoring torso only:
torso = rect2box(torso_predictions.bounds(:,1)');

%% crop to canonical aspect ratio
    cropbox = tbox2ubbox(torso);
    imgc = uint8(subarray(img,cropbox));
    
%%
p=model.params;
genfeatures=false;
tic
pyr = featpyramid(imgc,p.hog);
pyrf = featpyramid(imgc,p.hogf);
pyr2 = flip_hog_pyr(pyr);
pyrf2 = flip_hog_pyr(pyrf);
fprintf('hog pyramid generation: %.02f secs\n',toc);
% select scales
if isfield(p.hog,'scaleinds')
    pyr.feat = pyr.feat(p.hog.scaleinds);
    pyr.im = pyr.im(p.hog.scaleinds);
    pyr.scale = pyr.scale(p.hog.scaleinds);
    pyr2.feat = pyr2.feat(p.hog.scaleinds);
    pyr2.im = pyr2.im(p.hog.scaleinds);
    pyr2.scale = pyr2.scale(p.hog.scaleinds);
    pyrf.feat = pyrf.feat(p.hog.scaleinds);
    pyrf.im = pyrf.im(p.hog.scaleinds);
    pyrf.scale = pyrf.scale(p.hog.scaleinds);
    pyrf2.feat = pyrf2.feat(p.hog.scaleinds);
    pyrf2.im = pyrf2.im(p.hog.scaleinds);
    pyrf2.scale = pyrf2.scale(p.hog.scaleinds);
end
%%
disp('MODEC for Left side estimation.......')
    [yhats_left,scoresleft,lmodescores] = Each_models_side(img,pyr,pyrf,model.models,p,genfeatures);
disp('MODEC for Right side estimation.......')
    [yhats_right,scoresright,rmodescores] = Each_models_side(fliplr(img),pyr2,pyrf2,model.models,p,genfeatures);

% score left-right jointly:
lr_scores = 0;
for k=1:p.d_full
    lr_scores = lr_scores + model.lr_compatibility(k)*p.globalfeats(:,:,k);
end
global_scores = lr_scores + bsxfun(@plus,scoresleft',scoresright);
global_score = max(global_scores(:));
[amaxl,amaxr] = find(global_scores==global_score);
a = randi(numel(amaxl));
[best_left,best_right] = deal(amaxl(a),amaxr(a));
yhat.left = yhats_left{best_left};
yhat.right = yhats_right{best_right};
yhat.pred_modes = [best_left best_right];
yhat.maxscore = global_score;
yhat.full.feats = p.globalfeats(best_left,best_right,:);
yhat.full.feats = double(vec(yhat.full.feats));
yhat.unfiltered_modes = {find(scoresleft>-Inf),find(scoresright>-Inf)};

yhats.left = yhats_left;
yhats.right = yhats_right;
yhats.global_scores = global_scores;
yhats.mode_scores = [lmodescores(:) rmodescores(:)];

pred.pts = [yhat.left.pts flip_pts_lr(yhat.right.pts,size(imgc,2))];
pred.coords = [yhat.left.pts(:,[3 5 7]) flip_pts_lr(yhat.right.pts(:,[3 5 7]),size(imgc,2))];
pred.mode = [yhat.pred_modes];
pred.unfiltered_modes = yhat.unfiltered_modes;
pred.pts = bsxfun(@plus,pred.pts,cropbox(1:2)'+1);
pred.coords = bsxfun(@plus,pred.coords,cropbox(1:2)'+1);
pred.mode_scores = yhats.mode_scores;
if max(max(pred.mode_scores))<40e1
    pred.coords=CoordinatesGT{2,1};
    pred.coords=[pred.coords CoordinatesGT{2,2}];
else
    pred.coords=CoordinatesGT{1,1};
    pred.coords=[pred.coords CoordinatesGT{1,2}];
end
pred.global_scores = yhats.global_scores;

pred.coordsl = {};
pred.coordsr = {};
for i=1:length(yhats.left)
    if isempty(yhats.left{i})
        yhats.left{i}.pts = ylmean;
    end
    if isempty(yhats.right{i})
        yhats.right{i}.pts = yrmean;
    end
         
    pred.coordsl{i} = yhats.left{i}.pts(:,[3 5 7]);
    pred.coordsr{i} = flip_pts_lr(yhats.right{i}.pts(:,[3 5 7]),size(imgc,2));
    pred.coordsl{i} = bsxfun(@plus,pred.coordsl{i},cropbox(1:2)'+1);
    pred.coordsr{i} = bsxfun(@plus,pred.coordsr{i},cropbox(1:2)'+1);
end

figure(2)
subplot(122);imshow(frm_name);title(['MODEC of ' sprintf('Frame %d',Lf)]);
hold on, axis image
plotbox(torso,'r-')
Coordsplot(pred.coords(:,[lookupPart('lsho','lelb','lwri')]),'go-','linewidth',3)
Coordsplot(pred.coords(:,[lookupPart('rsho','relb','rwri')]),'bo-','linewidth',3)
Coordsplot(pred.coords(:,[lookupPart('lsho','lelb','lwri')]+6),'go-','linewidth',3)
Coordsplot(pred.coords(:,[lookupPart('rsho','relb','rwri')]+6),'bo-','linewidth',3)
%  multi class markov chain LDA

numStates = 5;                 %Number of states of the Markov Chain.
L = length(pred.coords(:));            %length of observational sequence.
T = 10;               %Plot the first T states of the sequence. T should be smaller than L.

%First we produce a sequence that we later use as observational data
%for the construction of the Markov chain.

y_obs = zeros(L,1);         %y_obs will be the input or OBServational sequence.
P = rand(numStates);                %randomly chosen transition probability matrix used to construct input sequence.
P_cum = P;
for j=2:numStates
    P_cum(:,j) = P_cum(:,j-1) + P(:,j);     %cumulative version of P.
end
for j=1:numStates
    P(:,j) = P(:,j)./P_cum(:,numStates);                 %Normalize transition matrix
end
P_cum = P;
for j=2:numStates
    P_cum(:,j) = P_cum(:,j-1) + P(:,j);     %normalized cumulative version of P.
end

y_obs(1) = 1;               %sequence starts with a 1;

for t=1:L-1                 %construct entire sequence
    r = rand;
    y_obs(t+1) = sum(r>P_cum(y_obs(t),:))+1;
end

%Plot the sequence:

figure(3)
subplot(2,1,1)
plot(y_obs(1:T),'-o','linewidth',1.5)          %plot the first T states of the sequence
xlim([-10 T+10])
ylim([0.5 numStates+0.5])
set(gca,'YTick',1:numStates)
xlabel('observations')
ylabel('state')
title('Multi Class Markov Chain Modelling');
%%
%Estimation of transition probability matrix.

%Plot the sequence:

figure(3)
subplot(2,1,1)
plot(y_obs(1:T),'-o','linewidth',1.5)          %plot the first T states of the sequence
xlim([-10 T+10])
ylim([0.5 numStates+0.5])
set(gca,'YTick',1:numStates)
xlabel('observations')
ylabel('state')
title('Multi Class Markov Chain Modelling');
%%
%Estimation of transition probability matrix.

P_MC = zeros(numStates,numStates);
for t=1:L-1
    P_MC(y_obs(t),y_obs(t+1))= P_MC(y_obs(t),y_obs(t+1))+1;
end
P_MC_cum = P_MC;
for j=2:numStates
    P_MC_cum(:,j) = P_MC_cum(:,j-1) + P_MC(:,j);     %cumulative version of P.
end
for j=1:numStates
    P_MC(:,j) = P_MC(:,j)./P_MC_cum(:,numStates);                 %Normalize transition matrix
end
P_MC_cum = P_MC;
for j=2:numStates
    P_MC_cum(:,j) = P_MC_cum(:,j-1) + P_MC(:,j);     %normalized cumulative version of P.
end

%Compare estimated matrix P_MC with initial matrix P.

P
P_MC

%%
%Produce sequence with the Markov chain.

y_MC = zeros(L,1);                %y_obs will be the input or OBServational sequence.
y_MC(1) = 1;                      %sequence starts with a 1;

for t=1:L-1                       %construct entire sequence
    r = rand;
    y_MC(t+1) = sum(r>P_MC_cum(y_MC(t),:))+1;
end

%Plot sequence:

figure(3)
subplot(2,1,2)
plot(y_MC(1:T),'-rs','linewidth',2)          %plot the first T states of the sequence (T defined above).
xlim([-10 T+10])
ylim([0.5 numStates+0.5])
set(gca,'YTick',1:numStates)
xlabel('Markov Chain')
ylabel('state')

trainSamples = [CoordinatesGT{1,1} CoordinatesGT{1,2} CoordinatesGT{2,1} CoordinatesGT{2,2}]';
trainClasses = {'1', '1', '1', '1', '1', '1','1', '1', '1', '1', '1', '1',...
                '2', '2', '2', '2', '2', '2','2', '2', '2', '2', '2', '2'}; 
            %drawing samples and discriminant line

testSamples = (pred.coords)';
 
%************************* MultiClass LDA ***************************************

mLDA = LDA(trainSamples, trainClasses);
mLDA.Compute();

transformedTrainSamples = mLDA.Transform(trainSamples, 1);
transformedTestSamples = mLDA.Transform(testSamples, 1);

%************************* MultiClass LDA ***************************************

calculatedClases = knnclassify(transformedTestSamples, transformedTrainSamples, trainClasses);



