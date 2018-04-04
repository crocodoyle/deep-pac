clear all;
clc;

% Import atlas data
[~, ~, raw] = xlsread('vols\BNA_subregions.xlsx','Sheet1','A2:I124');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};
cellVectors = raw(:,[1,2,3,6,7,8,9]);
raw = raw(:,[4,5]);
data = reshape([raw{:}],size(raw));
BNAsubregions = table;
BNAsubregions.Lobe = cellVectors(:,1);
BNAsubregions.Gyrus = cellVectors(:,2);
BNAsubregions.LeftandRightHemisphere = cellVectors(:,3);
BNAsubregions.LabelIDL = data(:,1);
BNAsubregions.LabelIDR = data(:,2);
BNAsubregions.VarName6 = cellVectors(:,4);
BNAsubregions.AnatomicalandmodifiedCytoarchitectonicdescriptions = cellVectors(:,5);
BNAsubregions.lhMNIXYZ = cellVectors(:,6);
BNAsubregions.rhMNIXYZ = cellVectors(:,7);

clearvars data raw cellVectors;

coords = zeros(size(BNAsubregions, 1)*2, 3);
names = cell(size(BNAsubregions, 1)*2, 1);
var_names = cell(size(BNAsubregions, 1)*2, 1);
% Organize atlas data to necessary format
for i = 0:size(BNAsubregions, 1) - 1
    region_name = BNAsubregions.LeftandRightHemisphere(i + 1);
    left_region_name = replace(region_name, {'(',')','R'}, '');
    right_region_name = replace(left_region_name, 'L', 'R');
    left_coords = cell2mat(cellfun(@str2num,BNAsubregions.lhMNIXYZ(i + 1),'uniform',0));
    right_coords = cell2mat(cellfun(@str2num,BNAsubregions.rhMNIXYZ(i + 1),'uniform',0));
    coords(2*i + 1,:) = left_coords;
    coords(2*i + 2, :) = right_coords;
    names(2*i + 1) = left_region_name;
    names(2*i + 2) = right_region_name;
    var_names(2*i + 1) = strcat('var_', left_region_name);
    var_names(2*i + 2) = strcat('var_', right_region_name);
end;

% Import subject data
[~, ~, raw] = xlsread('C:\Users\joshu\Desktop\PAC2018\PAC2018_Covariates_Upload.xlsx','Sheet1','A2:E1793');
raw(cellfun(@(x) ~isempty(x) && isnumeric(x) && isnan(x),raw)) = {''};
cellVectors = raw(:,1);
raw = raw(:,[2,3,4,5]);
data = reshape([raw{:}],size(raw));
PAC2018CovariatesUpload = table;
PAC2018CovariatesUpload.PAC_ID = cellVectors(:,1);
PAC2018CovariatesUpload.Label = data(:,1);
PAC2018CovariatesUpload.Age = data(:,2);
PAC2018CovariatesUpload.Gender = data(:,3);
PAC2018CovariatesUpload.TIV = data(:,4);

clearvars data raw cellVectors;

IMGSDIR = 'C:/PAC2018/';

atlasname = 'brainnetome';

files = dir([IMGSDIR filesep() 'brainnetome' filesep() 'PAC2018_*_GM-mean-brainnetome.txt']);

data = struct([]);
gmd_mean = zeros(size(files, 1), size(coords, 1));
gmd_var = zeros(size(files, 1), size(coords, 1));
for i = 1:size(files, 1)    
    pidx = strfind(files(i).name, '.txt') - 21;
    ptid = files(i).name(1:(pidx));
    data(i).ptid = ptid;
    data(i).atlas = 'brainnetome';
    data(i).gm_mean =  load([IMGSDIR filesep() 'brainnetome' filesep() files(i).name], '-ascii');
    data(i).gm_var = load([IMGSDIR filesep() 'brainnetome' filesep() ptid '_GM-var-brainnetome.txt'], '-ascii');
    gmd_mean(i, :) = data(i).gm_mean(2:247,2);
    gmd_var(i, :) = data(i).gm_var(2:247,2);
end;

tabgmd_mean = array2table(gmd_mean);
tabgmd_mean.Properties.VariableNames = names;
tabgmd_var = array2table(gmd_var);
tabgmd_var.Properties.VariableNames = var_names;
PAC2018CovariatesUpload = [PAC2018CovariatesUpload tabgmd_mean tabgmd_var];

writetable(PAC2018CovariatesUpload, 'PAC2018Covariates_and_regional_GMD.csv');