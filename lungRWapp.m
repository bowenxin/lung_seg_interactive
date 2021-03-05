function varargout = lungRWapp(varargin)
%LUNGRWAPP MATLAB code file for lungRWapp.fig
%      LUNGRWAPP, by itself, creates a new LUNGRWAPP or raises the existing
%      singleton*.
%
%      H = LUNGRWAPP returns the handle to a new LUNGRWAPP or the handle to
%      the existing singleton*.
%
%      LUNGRWAPP('Property','Value',...) creates a new LUNGRWAPP using the
%      given property value pairs. Unrecognized properties are passed via
%      varargin to lungRWapp_OpeningFcn.  This calling syntax produces a
%      warning when there is an existing singleton*.
%
%      LUNGRWAPP('CALLBACK') and LUNGRWAPP('CALLBACK',hObject,...) call the
%      local function named CALLBACK in LUNGRWAPP.M with the given input
%      arguments.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help lungRWapp

% Last Modified by GUIDE v2.5 02-Feb-2018 17:06:46

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @lungRWapp_OpeningFcn, ...
                   'gui_OutputFcn',  @lungRWapp_OutputFcn, ...
                   'gui_LayoutFcn',  [], ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
   gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


%%-----------------------------------------------------------------------------------    
% -------------------- SELF DEFINED INITIALISATION FROM HERE ------------------------
% -----------------------------------------------------------------------------------

% --- Executes just before lungRWapp is made visible.
function lungRWapp_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   unrecognized PropertyName/PropertyValue pairs from the
%            command line (see VARARGIN)

% Choose default command line output for lungRWapp
handles.output = hObject;

   
% setup, akzoom, pointer
set(handles.radiobutton_mediastinum, 'Value', 1);
set(handles.togglebutton_draw_view, 'String', 'Draw');
set(hObject, 'WindowKeyPressFcn', @keyPressCallback);
set(hObject,'WindowButtonMotionFcn','');
% set(hObject, 'WindowButtonDownFcn',@draw_mouse_down); % set draw callback
% akZoom('all_linked'); % pan, zoom and reset with mouse

% Add path 
addpath('./sc');
addpath('./natsortfiles');
addpath('./graphAnalysisToolbox-1.0');

% Update handles structure
guidata(hObject, handles);






% UIWAIT makes lungRWapp wait for user response (see UIRESUME)
% uiwait(handles.figure_lungRWapp);

%%-----------------------------------------------------------------------------------    
% -------------------------- SELF DEFINED SUPPORT FROM HERE ------------------------
% -----------------------------------------------------------------------------------

function update_axes_images(handles)
    % setup CT window for axis original and seeds
    curSlice = handles.volume.curSlice;
    curCTwindow = handles.volume.curCTwindow;
    displayRange = getDisplayRange(curCTwindow);
    % setup seeds for axis seeds
    curSeedsLabels = handles.draw.seedsLabels{curSlice};
    curSeedsIdx = handles.draw.seedsIdx{curSlice};
    sizeX = handles.volume.sizeX;
    [seedsX, seedsY] = getSeedsXY(curSeedsIdx, sizeX);
    redSeedsIdx = find(curSeedsLabels == 1);
    greenSeedsIdx = find(curSeedsLabels == 2);
    % setup output contour for axis contour
    curoutImageContour = handles.outImage.outImageContour{curSlice};
    
    % display axis original
    axes(handles.axes_original);
    imshow(handles.volume.imageOriginal{curSlice},displayRange);
    % display axis seeds
    axes(handles.axes_seeds);
    hold on;
    imshow(handles.volume.imageOriginal{curSlice},displayRange);
    plot(seedsY(redSeedsIdx), seedsX(redSeedsIdx), '.r'); 
    plot(seedsY(greenSeedsIdx), seedsX(greenSeedsIdx), '.g'); 
    hold off;
    axis tight;
    % display axis contour
    axes(handles.axes_contour);
    if isempty(curoutImageContour)
        imshow(handles.volume.imageContour{curSlice});
    else
        imshow(curoutImageContour);
    end
%     impixelinfo;

    

% seeds generation
function [seedsLabelNum, seedsLabels, seedsIdx] = seedsGenerate(imageSeeds)
    % input original seeded image, all slices with most image empty
    % output seedsLabelNum, kinds of label, 2 currently
    %        seedsLabel, label of seeds, double; 1 or 2
    %       seedsIdx, idx of seeds, double; 0 - 512*512
    for cnt = 1:numel(imageSeeds)
        ref=imageSeeds{cnt};
        
        if isempty(ref) ==0

            L{1} = find(ref(:,:,1)==1.0 & ref(:,:,2)==0.0 & ref(:,:,3)==0.0); % R
            L{2} = find(ref(:,:,1)==0.0 & ref(:,:,2)==1.0 & ref(:,:,3)==0.0); % G
            L{3} = find(ref(:,:,1)==0.0 & ref(:,:,2)==0.0 & ref(:,:,3)==1.0); % B

            num = 0;
            seedsLabelNum{cnt} = 0;
            % get SeedsNum, seedsLabel and seedsIdx
            for i=1:3
                nL = size(L{i},1); % length of 1 kind of seed
                if nL > 0
                    seedsLabelNum{cnt} = seedsLabelNum{cnt} + 1;
                    seedsLabels{cnt}(num+1:num+nL) = seedsLabelNum{cnt};
                    seedsIdx{cnt}(num+1:num+nL) = L{i};
                    num = num + nL;
                end
            end
        else
            seedsLabelNum{cnt} = [];
            seedsLabels{cnt} = [];
            seedsIdx{cnt} = [];
        end
    end

    
function [seedsX, seedsY] = getSeedsXY(seedsIdx, sizeX)    
        seedsX = mod(seedsIdx,sizeX)+1;
        seedsY = ceil(seedsIdx / sizeX);        


function pixelRange = getDisplayRange(curCTwindow)
    switch curCTwindow
        case 'Lung'
            windowLevel = -600;
            windowWidth = 1500;
        case 'Mediastinum'
            windowLevel = 50;
            windowWidth = 350;
    end
    maxCTvalue = windowLevel + windowWidth/2;
    minCTvalue = windowLevel - windowWidth/2;
    maxPixelValue = 1024 + maxCTvalue;
    minPixelValue = 1024 + minCTvalue;
    pixelRange = [minPixelValue, maxPixelValue];


function [curMask,curProb,curContour] = segmentRW(src,handles)
    % for 1 slice
    curSlice = handles.volume.curSlice;
    curImageOriginal = handles.volume.imageOriginal{curSlice};
    curImageOriginal = imageValueScale(curImageOriginal);
    
    curSeedsLabels = handles.draw.seedsLabels{curSlice};
    curSeedsIdx = handles.draw.seedsIdx{curSlice};
    [curMask,curProb] = random_walker(curImageOriginal,curSeedsIdx,curSeedsLabels);
    curContour=getContourImage(curImageOriginal,curMask);
    
%     axes(handles.axes_contour);
%     if isempty(curContour) == 0
%         imshow(curContour);
%     end

    
function imgContour=getContourImage(img,mask,bi)
    
    if nargin < 3
        bi = 0;
    end

    %Inputs
    [X Y Z]=size(img);
    nlabels = max(mask(:));
    bg_area = find(mask(:)==nlabels);

    %Build outputs
    imgMasks=reshape(mask,X,Y);

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

    imgContour=img(:,:,1);
    %==== added by Hui
    imgContour(xcont)=255;
    imgContour(ycont)=255;
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
        imgContour(:,:,2)=imgTmp2;

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
        imgContour(:,:,3)=imgTmp3;
    else
        imgTmp1=img(:,:,1);
        imgTmp1(xcont)=0;
        imgTmp1(ycont)=0;
        imgContour(:,:,2)=imgTmp1;
        imgContour(:,:,3)=imgTmp1;
    end
    
    
function img = imageValueScale(img)
    % reduce redundant pixel and scale

    if isa(img,'double')
        img=im2uint16(img);
    end
    if isa(img,'uint32')
        img=uint16(img);
    end
    if isa(img,'uint16')
        mx=65536;
        img=img-min(img(:));
        low_in = double(min(img(:)))/mx;
        high_in =double(max(img(:)))/mx;
        img = imadjust(double(img)./mx,[low_in; high_in],[]);
    end

function [mask,probabilities] = random_walker(img,seeds,labels,beta)
    %Function random_walker(img,seeds,labels,beta) uses the 
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
    
    % generate empty possibility and mask
    % -> add by bowen 28/01/2018
    if isempty(probabilities)
    probabilities=ones(size(img));
    mask=ones(size(img));
    end

%%-----------------------------------------------------------------------------------    
% -------------------------- SELF DEFINED CALLBACK FROM HERE ------------------------
% -----------------------------------------------------------------------------------

function draw_mouse_down(src, event)
    % get data from handles
    handles = guidata(src);
    sizeX = handles.volume.sizeX;
    sizeY = handles.volume.sizeY;
    curSlice = handles.volume.curSlice;  
    curImageOriginal = handles.volume.imageOriginal{curSlice};


    curSeedsLabels = handles.draw.seedsLabels{curSlice};
    curSeedsIdx = handles.draw.seedsIdx{curSlice};
    
    numLabel = handles.draw.numLabel;
    curLabel = handles.draw.curLabel;
    colors = {'r', 'g', 'b'};

    
    % Regular click, start placing seeds
    if strcmp(get(src,'SelectionType'),'normal')
        % Mouse moved, means adding more seeds
        cp = get(gca,'CurrentPoint');
        y = ceil(cp(1,1));
        x = ceil(cp(1,2));
        
        % Make sure the click is on the image
        if x > sizeX, x = handles.volume.sizeX; end
        if x < 1, x = 1; end    
        if y > sizeY, y = handles.volume.sizeY; end        
        if y < 1, y = 1; end        
        
        % Add the new seed and display it
        idx = (y-1)*sizeX + x; % x,y to idx 
        if isempty(curSeedsIdx) || isempty(find(curSeedsIdx == idx)) %->get seeds from image
            curSeedsIdx = [curSeedsIdx, idx];
            curSeedsLabels = [curSeedsLabels, curLabel];
            hold on;
            plot(y, x, ['.' colors{curLabel}]);  
            handles.draw.seedsLabels{curSlice} = curSeedsLabels;
            handles.draw.seedsIdx{curSlice} = curSeedsIdx;
        end
        
        set(src,'WindowButtonMotionFcn',@draw_mouse_move)
        set(src,'WindowButtonUpFcn',@draw_mouse_up)
        guidata(gcf, handles);

      
    % alt-click means change regions (red to green)
    elseif strcmp(get(src,'SelectionType'),'alt')
        curLabel = mod(curLabel,numLabel)+1;
        handles.draw.curLabel = curLabel;
        colors = {'r', 'g', 'b'};
        curColor = colors{curLabel};
        set(handles.text_mode, 'BackgroundColor', curColor); % change the status color
        guidata(gcf, handles);

    end
 
    
    



function draw_mouse_move(src,event)
    % get data from handles
    handles = guidata(src);
    sizeX = handles.volume.sizeX;
    sizeY = handles.volume.sizeY;
    curSlice = handles.volume.curSlice;
    curImageOriginal = handles.volume.imageOriginal{curSlice};

    curSeedsLabels = handles.draw.seedsLabels{curSlice};
    curSeedsIdx = handles.draw.seedsIdx{curSlice};
    
    curLabel = handles.draw.curLabel;
    colors = {'r', 'g', 'b'};
    
    % Mouse moved, means adding more seeds
    cp = get(gca,'CurrentPoint');
    y = ceil(cp(1,1));
    x = ceil(cp(1,2));

    % Make sure the click is on the image
    if x > sizeX, x = handles.volume.sizeX; end
    if x < 1, x = 1; end    
    if y > sizeY, y = handles.volume.sizeY; end        
    if y < 1, y = 1; end        

    % Add the new seed and display it
    idx = (y-1)*sizeX + x; % x,y to idx 
    if isempty(curSeedsIdx) || isempty(find(curSeedsIdx == idx)) %->get seeds from image
        curSeedsIdx = [curSeedsIdx, idx];
        curSeedsLabels = [curSeedsLabels, curLabel];
        plot(y, x, ['.' colors{curLabel}]);
        handles.draw.seedsLabels{curSlice} = curSeedsLabels;
        handles.draw.seedsIdx{curSlice} = curSeedsIdx;
    end
    

    guidata(src, handles);


    

function draw_mouse_up(src,event)
    handles = guidata(src);
    hold off;
    set(src,'WindowButtonMotionFcn','')
    set(src,'WindowButtonUpFcn','')
    
    curSlice = handles.volume.curSlice;
    [curMask,curProb,curContour] = segmentRW(src, handles);
    handles.outImage.outImageMask{curSlice} = curMask;
    handles.outImage.outImageContour{curSlice} = curContour;
    handles.outImage.outImageProb{curSlice} = curProb;
%    update_axes_images(handles);
%     impixelinfo(gcf);

    % display axis contour
    axes(handles.axes_contour);
    imshow(curContour);
    guidata(src, handles);



   
    

%%-----------------------------------------------------------------------------------    
% -------------------------- SYSTEM DEFINED CALLBACK FROM HERE ----------------------
% -----------------------------------------------------------------------------------

% --- Outputs from this function are returned to the command line.
function varargout = lungRWapp_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function slider_slice_Callback(hObject, eventdata, handles)
% hObject    handle to slider_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
sliderVal = round(hObject.Value);
hObject.Value=sliderVal;
handles.volume.curSlice=sliderVal;
handles.edit_slice.String=sliderVal;
set_text_scan(handles);
guidata(hObject, handles);
update_axes_images(handles);





% --- Executes during object creation, after setting all properties.
function slider_slice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


function edit_slice_Callback(hObject, eventdata, handles)
% hObject    handle to edit_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_slice as text
%        str2double(get(hObject,'String')) returns contents of edit_slice as a double
editVal = round(str2double(hObject.String));
if editVal < handles.slider_slice.Min || editVal > handles.slider_slice.Max
    msgbox('Input out of range');
else
    handles.volume.curSlice = editVal;
    set_text_scan(handles);
    guidata(hObject, handles);
    update_axes_images(handles);
end


% --- Executes during object creation, after setting all properties.
function edit_slice_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_slice (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on mouse press over axes background.
function axes_seeds_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axes_seeds (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes during object creation, after setting all properties.
function figure_lungRWapp_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure_lungRWapp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in togglebutton_draw_view.
function togglebutton_draw_view_Callback(hObject, eventdata, handles)
% hObject    handle to togglebutton_draw_view (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of togglebutton_draw_view
% default button state is View
curMode = get(handles.text_mode, 'String');
if strcmp(curMode, 'View') 
    % Currently view state; change state to draw
    set(handles.text_mode, 'String', 'Draw')
    colors = {'r', 'g', 'b'};
    curColor = colors{handles.draw.curLabel};
    set(handles.text_mode, 'BackgroundColor', curColor); % change the status color
    
    set(hObject, 'String', 'View');
    set(handles.figure_lungRWapp,'Pointer','crosshair'); % set mouse as crosshair
    set(handles.figure_lungRWapp, 'WindowButtonDownFcn',@draw_mouse_down); % set draw callback
elseif strcmp(curMode, 'Draw')
    % This is the draw state, change to view
    set(handles.text_mode, 'String', 'View')
	set(hObject, 'String', 'Draw');
    set(handles.figure_lungRWapp,'Pointer','arrow'); % set mouse as crosshair
    akZoom('all_linked');
%     impixelinfo;
    set(handles.text_mode, 'BackgroundColor', [0.8,0.8,0.8]); % change the status color


end


% --- Executes when selected object is changed in uibuttongroup_ctwindow.
function uibuttongroup_ctwindow_SelectionChangedFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uibuttongroup_ctwindow 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
curCTwindow = get(handles.uibuttongroup_ctwindow, 'SelectedObject');
curCTwindow = get(curCTwindow, 'String');
switch curCTwindow
    case 'Mediastinum'
        handles.volume.curCTwindow = 'Mediastinum';        
    case 'Lung'
        handles.volume.curCTwindow = 'Lung';
end
guidata(hObject, handles);
update_axes_images(handles);



function edit_patient_rootdir_Callback(hObject, eventdata, handles)
% hObject    handle to edit_patient_rootdir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_patient_rootdir as text
%        str2double(get(hObject,'String')) returns contents of edit_patient_rootdir as a double


% --- Executes during object creation, after setting all properties.
function edit_patient_rootdir_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_patient_rootdir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit_patient_num_Callback(hObject, eventdata, handles)
% hObject    handle to edit_patient_num (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit_patient_num as text
%        str2double(get(hObject,'String')) returns contents of edit_patient_num as a double


% --- Executes during object creation, after setting all properties.
function edit_patient_num_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit_patient_num (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in pushbutton_next.
function pushbutton_next_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_next (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
newPatientNum = handles.volume.patientNum + 1;
patientRoot = handles.volume.patientRoot;
max = numel(dir2(patientRoot));
if newPatientNum >= 1 && newPatientNum <= max
    handles.volume.patientNum = newPatientNum ;
    set(handles.edit_patient_num, 'String', num2str(newPatientNum));
    % setup volume,draw and output
    volume = setupVolume(patientRoot, newPatientNum); % setup volume struct
    draw = setup_draw(volume.imageSeeds); % setup draw struct
    outImage = setup_outImage(volume.sizeZ); % setup output struct
    sizeZ = volume.sizeZ;
    set(handles.slider_slice, 'Max', sizeZ, 'Min', 1, 'SliderStep', [1/(sizeZ-1) , 10/(sizeZ-1)]);
    set(handles.slider_slice, 'Value', volume.curSlice);
    handles.volume = volume;
    handles.draw = draw;
    handles.outImage = outImage;

    guidata(hObject, handles); % save
    update_axes_images(handles); %show
    checkPatientWithEmptySeeds(handles.volume.sliceNameSeeds);
    setPatientInfo(handles);
else 
    msgbox('Patient number out of range. ');
end


% --- Executes on button press in pushbutton_back.
function pushbutton_back_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_back (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
newPatientNum = handles.volume.patientNum - 1;
patientRoot = handles.volume.patientRoot;
max = numel(dir2(patientRoot));
if newPatientNum >= 1 && newPatientNum <= max
    handles.volume.patientNum = newPatientNum ;
    set(handles.edit_patient_num, 'String', num2str(newPatientNum));
    % setup volume,draw and outImage
    volume = setupVolume(patientRoot, newPatientNum); % setup volume struct
    draw = setup_draw(volume.imageSeeds); % setup draw struct
    outImage = setup_outImage(volume.sizeZ); % setup outImage struct
    sizeZ = volume.sizeZ;
    set(handles.slider_slice, 'Max', sizeZ, 'Min', 1, 'SliderStep', [1/(sizeZ-1) , 10/(sizeZ-1)]);
    set(handles.slider_slice, 'Value', volume.curSlice);
    setPatientInfo(handles);  
    
    handles.volume = volume;
    handles.draw = draw;
    handles.outImage = outImage;

    guidata(hObject, handles); % save
    update_axes_images(handles); %show
    checkPatientWithEmptySeeds(handles.volume.sliceNameSeeds);
else 
    msgbox('Patient number out of range. ');
end
guidata(hObject, handles);



% --- Executes on button press in pushbutton_back.
function pushbutton_load_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_back (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


patientRoot = handles.edit_patient_rootdir.String;
patientNum = str2double(handles.edit_patient_num.String);

max = numel(dir2(patientRoot));


if patientNum >= 1 && patientNum <= max % check range
    patientIDs = dir2(patientRoot);
    patientID = patientIDs(patientNum);
    caseNum = getPatient(getPatientList(patientRoot), patientNum);

    volume = setupVolume(patientRoot, patientNum); % setup volume struct
    draw = setup_draw(volume.imageSeeds); % setup draw struct
    outImage = setup_outImage(volume.sizeZ); % setup outImage struct
    sizeZ = volume.sizeZ;
    set(handles.slider_slice, 'Max', sizeZ, 'Min', 1, 'SliderStep', [1/(sizeZ-1) , 10/(sizeZ-1)]);
    set(handles.slider_slice, 'Value', volume.curSlice);
    set(handles.edit_slice,'String',num2str(volume.curSlice));
    string = sprintf('PatientID: %s\nCaseID: %s\nVolume: %d * %d * %d', ...
   patientID,caseNum,volume.sizeX,volume.sizeY ,sizeZ );
    set(handles.text_patientinfo, 'String',string, 'HorizontalAlignment','left');
    set(handles.uipanel_patient, 'Visible','On');
    set(handles.axes_original, 'Visible','On');
    set(handles.axes_seeds, 'Visible','On');
    set(handles.axes_contour, 'Visible','On');
    handles.volume = volume;
    handles.draw = draw;
    handles.outImage = outImage;

    update_axes_images(handles); %show
    checkPatientWithEmptySeeds(volume.sliceNameSeeds);
    guidata(hObject, handles);

else
    msgbox('Patient Number out of range. ');
end

function volume = setupVolume(patientRoot, patientNum)
%     patientRoot = 'F:\data\Special Lung Data\2rd patch\Consolidation'; % root directory for patients
%     patientNum = 2; % test data
    
    volume = struct;
    volume.patientRoot = patientRoot;
    volume.patientNum = patientNum;
    
    [imageOriginal, imageSeeds, imageContour,sliceName, sliceNameSeeds] = LoadImage(patientRoot, patientNum);
    volume.imageOriginal = imageOriginal;
    volume.imageSeeds = imageSeeds;
    volume.imageContour = imageContour;
    volume.sliceName = sliceName;
    volume.sliceNameSeeds = sliceNameSeeds;
    
    sizeZ = numel(imageOriginal);
    volume.sizeZ = sizeZ;
    volume.sizeX = size(imageOriginal{1},1);
    volume.sizeY = size(imageOriginal{1},2);
    
    volume.curSlice = 20;
    volume.curCTwindow = 'Mediastinum';

    
function draw = setup_draw(imageSeeds)
    draw = struct;
    [seedsLabelNum,seedsLabels, seedsIdx] = seedsGenerate(imageSeeds);
        
    draw.numLabel = 2; % currently 2 kinds of color used , red(1) and green(2)
    draw.curLabel = 1;
    draw.seedsLabels = seedsLabels;
    draw.seedsIdx = seedsIdx;
    draw.seedsLabelsBackup = seedsLabels; % as back up
    draw.seedsIdxBackup = seedsIdx; % as back up
    
function outImage = setup_outImage(sizeZ)
        % define outImage 
    outImage.outImageMask = cell(sizeZ,1);
    outImage.outImageContour = cell(sizeZ,1);
    outImage.outImageProb = cell(sizeZ,1);


% --- Executes on button press in pushbutton_save.
function pushbutton_save_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    % save curSlice slice MASK, PROBABILITY AND CONTOUR as file image 
    patientList = getPatientList(handles.volume.patientRoot);
    patientNum = handles.volume.patientNum;
    imageRoot = char(patientList(patientNum));
    caseNum = getPatient(patientList, patientNum);
    
    sizeX = handles.volume.sizeX;
    sizeZ = handles.volume.sizeZ;
%     curSlice = handles.volume.curSlice;
%     curSliceName = handles.volume.sliceName{curSlice};
    
    % to check weather directory existed
    outPathSeeds = fullfile(imageRoot, [caseNum '_seeds']);
    outPathMask = fullfile(imageRoot, [caseNum '_output'], 'mask');
    outPathContour = fullfile(imageRoot, [caseNum '_output'], 'contours');
    outPathProp = fullfile(imageRoot, [caseNum '_output'], 'prop');
    outPath = {outPathSeeds, outPathMask, outPathContour, outPathProp};
    for cnt = 1:numel(outPath)
        if exist(outPath{cnt}, 'dir') == 0
            mkdir(outPath{cnt});
        end
    end
    
    errorStr=check_slice_consecutive(handles.draw.seedsLabels, handles.volume.sliceName);

    
    % find all seeds amended and need to save
    sliceNeedSaving = find(~cellfun('isempty', handles.outImage.outImageMask));
    if isempty(sliceNeedSaving) == 0 
        for cnt = 1 : numel(sliceNeedSaving)
            curSlice = sliceNeedSaving(cnt);
            curSliceName = handles.volume.sliceName{curSlice};

            % save seeds
            curSeedsLabel = handles.draw.seedsLabels{curSlice};
            curSeedsIdx = handles.draw.seedsIdx{curSlice};
            curImageOriginal = handles.volume.imageOriginal{curSlice};
            curImageOriginal = imageValueScale(curImageOriginal);
            outSeeds = repmat(im2double(curImageOriginal),[1,1,3]);
            [seedsX, seedsY] = getSeedsXY(curSeedsIdx, sizeX);
            colors = {[1;0;0], [0;1;0], [0,0,1]};
            for cnt = 1:numel(curSeedsLabel)
                outSeeds(seedsX(cnt),seedsY(cnt),:) = colors{curSeedsLabel(cnt)};
            end
            imwrite(outSeeds, fullfile(outPathSeeds, [curSliceName, '.bmp']));

            % save mask
            outImageMask = handles.outImage.outImageMask{curSlice};
            outImageMask(outImageMask==1)=255;
            outImageMask=uint8(outImageMask);
            imwrite(outImageMask, fullfile(outPathMask, [curSliceName, '.bmp']));

            % save contour
            outImageContour = handles.outImage.outImageContour{curSlice};
            imwrite(outImageContour, fullfile(outPathContour, [curSliceName, '.bmp']));

            % save Probibility
            outImageProb = handles.outImage.outImageProb{curSlice};
            for label = 1:2 % 1 is red and 2 is green
                outPathProbLabel = fullfile(outPathProp,num2str(label));
                if exist(outPathProbLabel, 'dir') == 0
                    mkdir(outPathProbLabel);
                end
                prob_map=outImageProb(:,:,label);
                prob_img = sc(outImageProb(:,:,label),'prob_jet');
                imwrite(prob_img, fullfile(outPathProbLabel, [curSliceName, '.bmp']));
            end
        end
        
        if isempty(errorStr) == 1
            msgbox('Save successful');
        else 
            msgbox(['Save successful but Non-Consecutive slices found: ', errorStr]);
        end
    else
        msgbox('No image needs saving');
    end
    


% Support Function for function pushbutton_save_Callback

function patientList = getPatientList(patientRoot)
    % GET VALID  e.g. patient + studyID
    patientList = dir2(patientRoot);
    TF = ~startsWith(patientList, '.'); %% Patient Flies Not started with '.'
    patientList = patientList(TF);
    for i = 1:length(patientList)
        studyID = dir2(fullfile(patientRoot, char(patientList(i))));
        patientList(i) = fullfile(patientRoot, char(patientList(i)), char(studyID));
    end

function patientCaseNum = getPatient(patientList, patientNum)
    % GET VALID CASE
    p_cases = dir2(char(patientList(patientNum)));
    
    % set conditions to get valid image folder
    cond1 = ~endsWith(p_cases, '_seeds'); % e.g. not 2_seeds
    cond2 = ~endsWith(p_cases, '_output');
    p_cases = p_cases(cond1 & cond2);
    % find case
    assert(length(p_cases) == 1, 'wrong case folder');
    patientCaseNum = char(p_cases);


    
            


% --- Executes on button press in pushbutton_clear.
function pushbutton_clear_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_clear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
rect = getrect(handles.axes_seeds);
curSlice = handles.volume.curSlice;
curSeedsIdx = handles.draw.seedsIdx{curSlice};
% disp(numel(curSeedsIdx));
curSeedsLabels = handles.draw.seedsLabels{curSlice};
sizeX = handles.volume.sizeX;
[seedsX, seedsY] = getSeedsXY(curSeedsIdx, sizeX);
rectX =[rect(1), rect(1)+rect(3),rect(1)+rect(3), rect(1), rect(1)];
rectY = [rect(2), rect(2),rect(2)+rect(4), rect(2)+rect(4), rect(2)];
outSelect = ~inpolygon(seedsY, seedsX, rectX, rectY); % CAREFUL DIFFERENT

handles.draw.seedsLabels{curSlice} = curSeedsLabels(outSelect);
handles.draw.seedsIdx{curSlice} = curSeedsIdx(outSelect);
% disp(numel(curSeedsIdx));

rate = rect(3) * rect(4) /handles.volume.sizeX/handles.volume.sizeY;% ratio of selection rect/ totalimage
if isempty(curSeedsLabels(outSelect)) || rate >0.43
    handles=deleteSlice(curSlice, handles);
end

update_axes_images(handles);

guidata(hObject, handles);


function keyPressCallback(hObject, eventdata)
    keyPressed = eventdata.Key;
    handles = guidata(hObject);
    if strcmp(keyPressed, 'leftarrow')
        curSlice = str2double(handles.edit_slice.String) - 1;
        if curSlice < handles.slider_slice.Min || curSlice > handles.slider_slice.Max
            msgbox('Input out of range');
        else
            handles.edit_slice.String = num2str(curSlice);
            handles.volume.curSlice = curSlice;
            set(handles.slider_slice, 'Value',curSlice);
            update_axes_images(handles);
        end
    elseif strcmp(keyPressed, 'rightarrow')
        curSlice = str2double(handles.edit_slice.String) + 1;
        if curSlice < handles.slider_slice.Min || curSlice > handles.slider_slice.Max
            msgbox('Input out of range');
        else
            handles.edit_slice.String = num2str(curSlice);
            handles.volume.curSlice = curSlice;
            set(handles.slider_slice, 'Value',curSlice);
            update_axes_images(handles);
        end
    end
    
    guidata(hObject, handles);

function checkPatientWithEmptySeeds(sliceNameSeeds)
    if isempty(sliceNameSeeds) == 1
        msgbox('Seeds have not created');
    end
    
function setPatientInfo(handles)
    volume = handles.volume;
    patientRoot = volume.patientRoot;
    patientNum = volume.patientNum;

    patientIDs = dir2(patientRoot);
    patientID = patientIDs(patientNum);
    caseNum = getPatient(getPatientList(patientRoot), patientNum);
    string = sprintf('PatientID: %s\nCaseID: %s\nVolume: %d * %d * %d', ...
   patientID,caseNum,volume.sizeX,volume.sizeY ,volume.sizeZ );
    set(handles.text_patientinfo, 'String',string, 'HorizontalAlignment','left');
    


% --- Executes on button press in pushbutton_browse.
function pushbutton_browse_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
patientRoot = uigetdir;
handles.volume.patientRoot = patientRoot;
set(handles.edit_patient_rootdir, 'String', patientRoot);
guidata(hObject, handles);


function set_text_scan(handles)
    curSlice = handles.volume.curSlice;
    set(handles.text_scan, 'String', handles.volume.sliceName(curSlice));

function errorStr = check_slice_consecutive(seedsLabels, sliceName) 
    sliceNeedCheck = find(~cellfun('isempty', seedsLabels));
    sliceDiff = diff(sliceNeedCheck);
    nonConsecutive = find(sliceDiff~=1);
    errorStr = '';
    if isempty(nonConsecutive) ~= 1
        for cnt = 1 : numel(nonConsecutive)
            nonConSlice = nonConsecutive(cnt);
            errorStr = [errorStr, char(sliceName(sliceNeedCheck(nonConSlice))), ', '];
        end
    end
        
        

function handles = deleteSlice(curSlice, handles)
    patientList = getPatientList(handles.volume.patientRoot);
    patientNum = handles.volume.patientNum;
    imageRoot = char(patientList(patientNum));
    caseNum = getPatient(patientList, patientNum);
    sliceName = handles.volume.sliceName;
    curFileName = [char(sliceName(curSlice)), '.bmp'];
    
    % delete files
    delPathSeeds = fullfile(imageRoot, [caseNum '_seeds'], curFileName);
    delPathMask = fullfile(imageRoot, [caseNum '_output'], 'mask', curFileName);
    delPathContour = fullfile(imageRoot, [caseNum '_output'], 'contours',curFileName);
    outPathProp1 = fullfile(imageRoot, [caseNum '_output'], 'prop','1', curFileName);
    outPathProp2 = fullfile(imageRoot, [caseNum '_output'], 'prop', '2', curFileName);
    
    if exist(delPathSeeds)
        delete(delPathSeeds, delPathMask, delPathContour);
    end
    % delete read in data
    handles.outImage.outImageContour{curSlice} = [];
    handles.outImage.outImageMask{curSlice} = [];
    handles.outImage.outImageProb{curSlice} = [];
    handles.draw.seedsIdx{curSlice} = [];
    handles.draw.seedsLabels{curSlice} = [];
    handles.volume.imageContour{curSlice} = [];
    
% --- Executes on button press in pushbutton_deleteAll.
function pushbutton_deleteAll_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_deleteAll (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    patientList = getPatientList(handles.volume.patientRoot);
    patientNum = handles.volume.patientNum;
    imageRoot = char(patientList(patientNum));
    caseNum = getPatient(patientList, patientNum);
    
    sizeX = handles.volume.sizeX;
    sizeZ = handles.volume.sizeZ;
    
    % to check weather directory existed
    delPathSeeds = fullfile(imageRoot, [caseNum '_seeds']);
    delPathOutput = fullfile(imageRoot, [caseNum '_output']);
    
    % Construct a questdlg with three options
    choice = questdlg('Would you like delete all the files?', ...
        'Confirmation', ...
        'Yes','No','No');
    % Handle response
    switch choice
        case 'Yes'
            if exist(delPathSeeds, 'dir') || exist(delPathOutput, 'dir')
                rmdir(delPathSeeds, 's');
                rmdir(delPathOutput, 's');
            end
            handles.volume.imageContour=cell(1,95);
            handles.draw.seedsLabels=cell(1,95);
            handles.draw.seedsIdx=cell(1,95);
            handles.outImage.outImageMask=cell(1,95);
            handles.outImage.outImageImageContour=cell(1,95);
            handles.outImage.outImageProb=cell(1,95);
            guidata(hObject,handles);
            update_axes_images(handles);
            msgbox('All seeds and outputs deleted.');
        case 'No'
    end
    
    
