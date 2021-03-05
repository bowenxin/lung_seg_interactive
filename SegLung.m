function [mask, probabilities] = SegLung(lungScan, lungScanSeed)
%% this is the function for target lung tumor segmentation
%% 1: organize raw data
%% 2: generate seeds images: 
%     a: find the slices you want to segment. 
%     b: for each slice, save it to a bmp image. 
%     c: open bmp image using paint
%     d: draw foreground and background seeds using R (255,0,0),G(0,255,0),
%     or B(0,0,255). save it to Seeds folder. e.g.
%     C:\Users\hcui7511\Desktop\lungCTsegmentation\274290\527_Seeds\
%%  Note: 1. this is 2D segmentation. So you need to generate seeds for all the slices you want to segment. 2. you can write another script to run all the slices automatically. 
%% find images
% imageroot='C:\Users\bxin3444\Desktop\47252';
% caseNumber='2';
% onlyname='FILE33';
%{
CT_name=fullfile(imageroot,caseNumber,strcat(onlyname,'.dcm'));
ref_name=fullfile(imageroot,strcat(caseNumber,'_seeds'),strcat(onlyname,'.bmp'));
OutputDir=fullfile(imageroot,strcat(caseNumber,'_output'));
if exist(OutputDir,'dir') == 0
    mkdir(OutputDir);
end
%}
% CT_name=strcat(imageroot,'527\',onlyname,'.dcm');
% ref_name=strcat(imageroot,'527_Seeds\',onlyname,'.bmp');
% OutputDir=strcat(imageroot,'527_output\');mkdir(OutputDir);
%% segmentation
%img=dicomread(CT_name);
img=lungScan;
ref_name=lungScanSeed;
if isa(img,'uint32')
    img=int16(img);
end
if isa(img,'uint16')
    % reduce redundant pixel and scale
    mx=65536;
    img=img-min(img(:));
    low_in = double(min(img(:)))/mx;
    high_in =double(max(img(:)))/mx;
    img = imadjust(double(img)./mx,[low_in; high_in],[]);
end

[~, labels, seeds] = seed_generation(lungScanSeed);
[mask,probabilities] = random_walker(img,seeds,labels);

%% save results
%{
if isempty(probabilities)
    f_name=[OutputRoot,'Empty_Output_RW.txt'];
    trouble_case=fopen(f_name,'a');
    fprintf(trouble_case,'\n seeds = %3.3f\n',thres,'case = %s\t',case_name,'Slice = %s\n',only_name);
    probabilities=ones(size(img_CT));
    mask=ones(size(img_CT));
end
save_2results(img,OutputDir,onlyname,probabilities,mask);
%}
end
%% support functions


function save_2results(img,OutputDir,name,probabilities,mask)
% work with the resutls of RW code
% Save the probability maps to 2 directories seperately
% draw the contours on PET and CT seperately. Function: segoutput_c.m. (revised version of original segoutput.m to make the contour red)
% save the output mask for further DSC comparison
% By Hui Cui 4/6/2014
K=size(probabilities,3);
outputDir_p=[OutputDir '\prop\'];
if exist(outputDir_p,'dir') == 0
    mkdir(outputDir_p);
end

for k=1:K
    outputDir_k=[outputDir_p '\' int2str(k) '\'];
    if exist(outputDir_k, 'dir') == 0
        mkdir(outputDir_k);
    end
    % save prob map for further evaluation
    if K~=1
        prob_map=probabilities(:,:,k);
        prob_img = sc(probabilities(:,:,k),'prob_jet');
    else
        prob_map=probabilities;
        prob_img = sc(probabilities,'prob_jet');
    end
    %     save([outputDir_k name '.mat'], 'prob_map');
    % save prob map in color version
    imwrite(prob_img, [outputDir_k name '.bmp']);
    clear prob_img;clear prob_map;
end;
% save contours on CT
[imgMasks,segOutline,imgMarkup]=segoutput_c(img,mask);
% --outputDir_c=[OutputDir '\contours\CT\'];
outputDir_c=[OutputDir '\contours\'];
if exist(outputDir_c, 'dir') == 0
    mkdir(outputDir_c);
end
imwrite(imgMarkup, [outputDir_c name '.bmp']);
% % save contours on PT
% [imgMasks,segOutline,imgMarkup]=segoutput_c(img2,mask);
% outputDir_c2=[OutputDir '\contours\PT\'];
%
%     mkdir(outputDir_c2);
%
% imwrite(imgMarkup, [outputDir_c2 name '.bmp']);
% save mask
outputDir_m=[OutputDir '\mask\'];
%newly add
if exist(outputDir_m,'dir') == 0
    mkdir(outputDir_m);
end
label_img = mask;
label_img(label_img==1)=255;
label_img=uint8(label_img);% double image should be converted to int8 for writing
imwrite(label_img, [outputDir_m name '.bmp']);
end
function [imgMasks,segOutline,imgMarkup]=segoutput_c(img,solution,bi)

if nargin < 3,
    bi = 0;
end;

%Inputs
[X Y Z]=size(img);
nlabels = max(solution(:));
bg_area = find(solution(:)==nlabels);

%Build outputs
imgMasks=reshape(solution,X,Y);

%Outline segments
imgSeg=imgMasks;
if X*Y == 1
    [fx,fy]=deal([]);
elseif X == 1
    fx=[];
    fy=gradient(imgSeg);
elseif Y == 1
    fx=gradient(imgSeg);
    fy=[];
else
    [fx,fy]=gradient(imgSeg);
end
[xcont_i xcont_j]=find(fx);
[ycont_i ycont_j]=find(fy);
xcont = find(fx);
ycont = find(fy);

segOutline=ones(X,Y);
segOutline(xcont)=0;
segOutline(ycont)=0;

imgMarkup=img(:,:,1);
%==== added by Hui
imgMarkup(xcont)=255;
imgMarkup(ycont)=255;
%====


if Z == 3
    imgTmp2=img(:,:,2); val2 = 1;
    if bi == 1, imgTmp2(bg_area)=imgTmp2(bg_area)/3; end;
    imgTmp2(xcont)=val2;
    if xcont_i+1 <= X, imgTmp2(xcont_i+1+X*(xcont_j-1))=val2; end;
    if xcont_i-1  > 0, imgTmp2(xcont_i-1+X*(xcont_j-1))=val2; end;
    if xcont_j-1  > 0, imgTmp2(xcont_i+X*(xcont_j-2))  =val2; end;
    if xcont_j+1 <= Y, imgTmp2(xcont_i+X*(xcont_j))    =val2; end;
    
    if xcont_i+1 <= X, if xcont_j-1  > 0, imgTmp2(xcont_i+1+X*(xcont_j-2))  =val2; end; end;
    if xcont_i+1 <= X, if xcont_j+1 <= Y, imgTmp2(xcont_i+1+X*(xcont_j))    =val2; end; end;
    if xcont_i-1  > 0, if xcont_j-1  > 0, imgTmp2(xcont_i-1+X*(xcont_j-2))  =val2; end; end;
    if xcont_i-1  > 0, if xcont_j+1 <= Y, imgTmp2(xcont_i-1+X*(xcont_j))    =val2; end; end;
    
    imgTmp2(ycont)=val2;
    if ycont_i+1 <= X, imgTmp2(ycont_i+1+X*(ycont_j-1))=val2; end;
    if ycont_i-1  > 0, imgTmp2(ycont_i-1+X*(ycont_j-1))=val2; end;
    if ycont_j-1  > 0, imgTmp2(ycont_i+X*(ycont_j-2))  =val2; end;
    if ycont_j+1 <= Y, imgTmp2(ycont_i+X*(ycont_j))    =val2; end;
    
    if ycont_i+1 <= X, if ycont_j-1  > 0, imgTmp2(ycont_i+1+X*(ycont_j-2))  =val2; end; end;
    if ycont_i+1 <= X, if ycont_j+1 <= Y, imgTmp2(ycont_i+1+X*(ycont_j))    =val2; end; end;
    if ycont_i-1  > 0, if ycont_j-1  > 0, imgTmp2(ycont_i-1+X*(ycont_j-2))  =val2; end; end;
    if ycont_i-1  > 0, if ycont_j+1 <= Y, imgTmp2(ycont_i-1+X*(ycont_j))    =val2; end; end;
    imgMarkup(:,:,2)=imgTmp2;
    
    imgTmp3=img(:,:,3); val3 = 0;
    if bi == 1, imgTmp3(bg_area)=imgTmp3(bg_area)/3; end;
    imgTmp3(xcont)=val3;
    if xcont_i+1 <= X, imgTmp3(xcont_i+1+X*(xcont_j-1))=val3; end;
    if xcont_i-1  > 0, imgTmp3(xcont_i-1+X*(xcont_j-1))=val3; end;
    if xcont_j-1  > 0, imgTmp3(xcont_i+X*(xcont_j-2))  =val3; end;
    if xcont_j+1 <= Y, imgTmp3(xcont_i+X*(xcont_j))    =val3; end;
    
    if xcont_i+1 <= X, if xcont_j-1  > 0, imgTmp3(xcont_i+1+X*(xcont_j-2))  =val3; end; end;
    if xcont_i+1 <= X, if xcont_j+1 <= Y, imgTmp3(xcont_i+1+X*(xcont_j))    =val3; end; end;
    if xcont_i-1  > 0, if xcont_j-1  > 0, imgTmp3(xcont_i-1+X*(xcont_j-2))  =val3; end; end;
    if xcont_i-1  > 0, if xcont_j+1 <= Y, imgTmp3(xcont_i-1+X*(xcont_j))    =val3; end; end;
    
    imgTmp3(ycont)=val3;
    if ycont_i+1 <= X, imgTmp3(ycont_i+1+X*(ycont_j-1))=val3; end;
    if ycont_i-1  > 0, imgTmp3(ycont_i-1+X*(ycont_j-1))=val3; end;
    if ycont_j-1  > 0, imgTmp3(ycont_i+X*(ycont_j-2))  =val3; end;
    if ycont_j+1 <= Y, imgTmp3(ycont_i+X*(ycont_j))    =val3; end;
    
    if ycont_i+1 <= X, if ycont_j-1  > 0, imgTmp3(ycont_i+1+X*(ycont_j-2))  =val3; end; end;
    if ycont_i+1 <= X, if ycont_j+1 <= Y, imgTmp3(ycont_i+1+X*(ycont_j))    =val3; end; end;
    if ycont_i-1  > 0, if ycont_j-1  > 0, imgTmp3(ycont_i-1+X*(ycont_j-2))  =val3; end; end;
    if ycont_i-1  > 0, if ycont_j+1 <= Y, imgTmp3(ycont_i-1+X*(ycont_j))    =val3; end; end;
    imgMarkup(:,:,3)=imgTmp3;
else
    imgTmp1=img(:,:,1);
    imgTmp1(xcont)=0;
    imgTmp1(ycont)=0;
    imgMarkup(:,:,2)=imgTmp1;
    imgMarkup(:,:,3)=imgTmp1;
end
end


% labels 1 or 2 represents red or green
% inx is the position
function [nlabels, labels, idx] = seed_generation(lungScanSeed,scale)

if nargin<2, scale = 1; end


ref=im2double(lungScanSeed); ref = imresize(ref,scale);

% if size(ref_judge,3)~=3
%     ref_rgb=ind2rgb(ref_judge,map);
%     ref=im2double(ref_rgb);
%     
%     ref = imresize(ref,scale);
% else
%     ref = imresize(ref,scale);%% original code
% end

% get RGB index from image for each channel

L{1} = find(ref(:,:,1)==1.0 & ref(:,:,2)==0.0 & ref(:,:,3)==0.0); % R
L{2} = find(ref(:,:,1)==0.0 & ref(:,:,2)==1.0 & ref(:,:,3)==0.0); % G
L{3} = find(ref(:,:,1)==0.0 & ref(:,:,2)==0.0 & ref(:,:,3)==1.0); % B

num = 0; % total seeds number
nlabels = 0; % number of color channelss
% labels refers to seed label, 1 or 2
% index pairs with label, 1 or 2

for i=1:3
    nL = size(L{i},1); % number of R,G,B pixel, e.g. 5 red seeds
    if nL > 0
        nlabels = nlabels + 1;
        labels(num+1:num+nL) = nlabels;
        idx(num+1:num+nL) = L{i}; % index of red pixel in image
        num = num + nL;
    end
end
end
function [mask,probabilities] = random_walker(img,seeds,labels,beta)
%Function [mask,probabilities] = random_walker(img,seeds,labels,beta) uses the 
%random walker segmentation algorithm to produce a segmentation given a 2D 
%image, input seeds and seed labels.
%
%Inputs: img - The image to be segmented
%        seeds - The input seed locations (given as image indices, i.e., 
%           as produced by sub2ind)
%        labels - Integer object labels for each seed.  The labels 
%           vector should be the same size as the seeds vector.
%        beta - Optional weighting parameter (Default beta = 90)
%
%Output: mask - A labeling of each pixel with values 1-K, indicating the
%           object membership of each pixel
%        probabilities - Pixel (i,j) belongs to label 'k' with probability
%           equal to probabilities(i,j,k)
%
%
%10/31/05 - Leo Grady
%Based on the paper:
%Leo Grady, "Random Walks for Image Segmentation", IEEE Trans. on Pattern 
%Analysis and Machine Intelligence, Vol. 28, No. 11, pp. 1768-1783, 
%Nov., 2006.
%
%Available at: http://www.cns.bu.edu/~lgrady/grady2006random.pdf
%
%Note: Requires installation of the Graph Analysis Toolbox available at:
%http://eslab.bu.edu/software/graphanalysis/

%Read inputs
if nargin < 4
    beta = 90;
end

%Find image size
img=im2double(img);
[X Y Z]=size(img);

%Error catches
exitFlag=0;
if((Z~=1) && (Z~=3)) %Check number of image channels
    disp('ERROR: Image must have one (grayscale) or three (color) channels.')
    exitFlag=1;
end 
if(sum(isnan(img(:))) || sum(isinf(img(:)))) %Check for NaN/Inf image values
    disp('ERROR: Image contains NaN or Inf values - Do not know how to handle.')
    exitFlag=1;
end
%Check seed locations argument
if(sum(seeds<1) || sum(seeds>size(img,1)*size(img,2)) || (sum(isnan(seeds)))) 
    disp('ERROR: All seed locations must be within image.')
    disp('The location is the index of the seed, as if the image is a matrix.')
    disp('i.e., 1 <= seeds <= size(img,1)*size(img,2)')
    exitFlag=1;
end
if(sum(diff(sort(seeds))==0)) %Check for duplicate seeds
    disp('ERROR: Duplicate seeds detected.')
    disp('Include only one entry per seed in the "seeds" and "labels" inputs.')
    exitFlag=1;
end
TolInt=0.01*sqrt(eps);
if(length(labels) - sum(abs(labels-round(labels)) < TolInt)) %Check seed labels argument
    disp('ERROR: Labels must be integer valued.');
    exitFlag=1;
end
if(length(beta)~=1) %Check beta argument
    disp('ERROR: The "beta" argument should contain only one value.');
    exitFlag=1;
end
if(exitFlag)
    disp('Exiting...')
    [mask,probabilities]=deal([]);
    return
end

%Build graph
[points edges]=lattice(X,Y);

%Generate weights and Laplacian matrix
if(Z > 1) %Color images
    tmp=img(:,:,1);
    imgVals=tmp(:);
    tmp=img(:,:,2);
    imgVals(:,2)=tmp(:);
    tmp=img(:,:,3);
    imgVals(:,3)=tmp(:);
else
    imgVals=img(:);
end
weights=makeweights(edges,imgVals,beta);
L=laplacian(edges,weights);
%L=laplacian(edges,weights,length(points),1);

%Determine which label values have been used
label_adjust=min(labels); labels=labels-label_adjust+1; %Adjust labels to be > 0
labels_record(labels)=1;
labels_present=find(labels_record);
number_labels=length(labels_present);

%Set up Dirichlet problem
boundary=zeros(length(seeds),number_labels);
for k=1:number_labels
    boundary(:,k)=(labels(:)==labels_present(k));
end

%Solve for random walker probabilities by solving combinatorial Dirichlet
%problem
probabilities=dirichletboundary(L,seeds(:),boundary);

%Generate mask
[dummy mask]=max(probabilities,[],2);
mask=labels_present(mask)+label_adjust-1; %Assign original labels to mask
mask=reshape(mask,[X Y]);
probabilities=reshape(probabilities,[X Y number_labels]);
end
