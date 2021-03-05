function [imageOriginal, imageSeeds, imageContour, sliceNameOriginal, sliceNameSeeds] = LoadImage(patientRoot, patientNum)
    % Load images
    % Original Image dataype 512*512 uint16
    % Seeds Image datatype 512*512 double
    % Contour Image datatype 512*512*3 uint8    

    % sample input
%     patientRoot = 'F:\data\Special Lung Data\2rd patch\Consolidation'; % root directory for patients
%     patientNum = 2;
    patientList = getPatientList(patientRoot);
    [imageOriginal,sliceNameOriginal] = getImageOriginal(patientList, patientNum);
    [imageSeeds, sliceNameSeeds] = getImageSeeds(patientList, patientNum,sliceNameOriginal);
    [imageContour, sliceNameContour] = getImageContour(patientList,patientNum, sliceNameOriginal);

end



%% support function

function patientList = getPatientList(patientRoot)
    % GET VALID  e.g. patient + studyID
    patientList = dir2(patientRoot);
    if ismember(".DS_Store", patientList)
        msgbox(['Please open your terminal and run,\n',...
            'sudo find / -name ".DS_Store" -depth -exec rm {} \;',...
            'and then reload']);
    end
    TF = ~startsWith(patientList, '.'); %% Patient Flies Not started with '.'
    patientList = patientList(TF);
    
    for i = 1:length(patientList)
        studyID = dir2(fullfile(patientRoot, char(patientList(i))));
        patientList(i) = fullfile(patientRoot, char(patientList(i)), char(studyID));
    end
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
end

% Read Original Image from a patient 
function [imageOriginal, sliceNameOriginal] = getImageOriginal(patientList, patientNum)  
    %input: patientList, patientNum
    %output: imageOriginal, sliceNameOriginal
    pathPatient = char(patientList(patientNum));
    patientCaseNum = getPatient(patientList, patientNum);

    pathImageOriginal = fullfile(pathPatient,patientCaseNum);
    dicomlist = dir(fullfile(pathImageOriginal,'*.dcm'));
    [~,ndx] = natsortfiles({dicomlist.name}); % sort
    dicomlist = dicomlist(ndx); 
    sliceNameOriginal = string(numel(dicomlist));
    for cnt = 1 : numel(dicomlist)
        sliceNameOriginal(cnt) = extractBefore(dicomlist(cnt).name, '.dcm');
        imageOriginal{cnt} = dicomread(fullfile(pathImageOriginal,dicomlist(cnt).name));
        if isa(imageOriginal{cnt},'int16')
            imageOriginal{cnt}=uint16(imageOriginal{cnt});
        end
    end
end

% Read Seeds Image from a patient 
function [imageSeeds, sliceNameSeeds] = getImageSeeds(patientList, patientNum, sliceNameOriginal)
    pathPatient = char(patientList(patientNum));
    patientCaseNum = getPatient(patientList, patientNum);
    
    pathImageSeeds = fullfile(pathPatient, [patientCaseNum, '_seeds']);
    
    if exist(pathImageSeeds)>0
        bmplist = dir(fullfile(pathImageSeeds,'*.bmp'));
        [~,ndx] = natsortfiles({bmplist.name}); % sort
        bmplist = bmplist(ndx); 
        sliceNameSeeds = strings(1, numel(bmplist));
        for cnt = 1 : numel(bmplist)
            sliceNameSeeds(cnt) = extractBefore(bmplist(cnt).name, '.bmp');
        end
        
        for cnt = 1 : numel(sliceNameOriginal)
            if ismember(sliceNameOriginal(cnt), sliceNameSeeds)
                imageSeeds{cnt} = imread(fullfile(pathImageSeeds,[sliceNameOriginal{cnt}, '.bmp']));
                imageSeeds{cnt} = im2double(imageSeeds{cnt});
            else
                imageSeeds{cnt} = [];
            end
        end
        
    else
        imageSeeds = cell(1,numel(sliceNameOriginal));
        sliceNameSeeds = [];
%         msgbox('Image Seeds have not created yet. ')
        warning('Image Seeds and Contour not found');
    end
    
end 
    
% Read Contour Image from a patient 
function [imageContour, sliceNameContour] = getImageContour(patientList, patientNum, sliceNameOriginal)
    pathPatient = char(patientList(patientNum));
    patientCaseNum = getPatient(patientList, patientNum);
    
    pathImageContour = fullfile(pathPatient, [patientCaseNum, '_output'], 'contours');
    
    if exist(pathImageContour)>0
        bmplist = dir(fullfile(pathImageContour,'*.bmp'));
        [~,ndx] = natsortfiles({bmplist.name}); % sort
        bmplist = bmplist(ndx); 
        sliceNameContour = strings(1, numel(bmplist));
        for cnt = 1 : numel(bmplist)
            sliceNameContour(cnt) = extractBefore(bmplist(cnt).name, '.bmp');
        end
        
        for cnt = 1 : numel(sliceNameOriginal)
            if ismember(sliceNameOriginal(cnt), sliceNameContour)
                imageContour{cnt} = imread(fullfile(pathImageContour,[sliceNameOriginal{cnt}, '.bmp']));
            else
                imageContour{cnt} = [];
            end
        end
        
    else
        imageContour = cell(1,numel(sliceNameOriginal));
        sliceNameContour = [];
        warning('Image Contour not found');
    end
    
end    


