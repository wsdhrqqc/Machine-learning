
function V = ICARTTreader(ICTname)

% Author: QING N AUG 11th

% http://cires1.colorado.edu/jimenez-group/software/ICARTTreader.m

% function V = ICARTTreader(ICTname)
% Parses an ICARTT-formatted file (.ict) into matlab variables.
%
% INPUTS:
% ICTname: name of .ict file to parse, INCLUDING extension.
%
% OUTPUTS:
% V: a structure containing all variables and the header info.
%    Use struct2var to put these variables into your workspace, if desired.
%
% 20130424 GMW
% 20130810 GMW  Modifed data-reading to use textscan instead of fscanf. This helps deal with files
%               that contain white space between values and commas.
% 20130811 GMW  Added to list of invalid characters for variable names.
% 20131219 GMW  Added some code to modify variable names that start with a number, as these are
%               invalid names for matlab. In these cases, the letter "n" is tacked onto the front
%               of the variable name.
% 20140612 GMW  Added try-catch statement to use fscanf is textscan fails to read data.
% 20150416 GMW  Modified to deal with case of empty ULOD and LLOD fields.


%%%%%OPEN FILE%%%%%
[fid,message] = fopen(ICTname);
if fid==-1
    error(message)
end

%%%%%GRAB HEADER%%%%%
numlines=fscanf(fid,'%f',1);
frewind(fid);
header = cell(numlines,1);
for i=1:numlines
    header{i} = fgetl(fid);
end
numvar1 = str2num(header{18})+1; %number of variables, including independent variable
numvar2 = str2num(header{11})+1;
%%%%%GRAB DATA, CLOSE FILE%%%%%
fstr1 = repmat('%f, ',1,numvar1); %format string
fstr1 = fstr1(1:end-2);

fstr2 = repmat('%f, ',1,numvar2); %format string
fstr2 = fstr2(1:end-2);
% floc1 = ftell(fid);
try
    data1 = textscan(fid,fstr1,1);
    data2 = textscan(fid,fstr2,inf);
    data2 = cell2mat(data2);
    data1 = cell2mat(data1);
catch
    disp('ICARTTreader: textscan failed. Using fscanf.')
    fseek(fid,floc,'bof');
    data2=fscanf(fid,fstr,[numvar,inf]); %old method
    data2=data2';
end
% floc1 = ftell(fid);

[nrow,ncol] = size(data2);

status = fclose(fid);
if status
    disp('Problem closing file.');
end

%%%%%PARSE HEADER%%%%%
scale = [1 str2num(header{12})]; %scaling factors
miss  = str2num(header{13}); %missing data indicators
miss = [-9999 miss]; %extend for independent variable
miss = repmat(miss,nrow,1);

llod=miss(end);
i = strmatch('LLOD_FLAG',header); %lower detection limit
if ~isempty(i)
    lod = str2num(header{i}(12:end));
    if ~isempty(lod), llod=lod; end
end
    
ulod=miss(end);
i = strmatch('ULOD_FLAG',header); %upper detection limit
if ~isempty(i)
    lod = str2num(header{i}(12:end));
    if ~isempty(lod), ulod=lod; end
end

% %filter variable names
varnames = textscan(header{end},'%s','Delimiter',',');
varnames = varnames{1};
badchar = '=()[]{}/\+-<>;:!@#$%^&*|?.'' '; %invalid characters
for i=1:numvar1+numvar2
    var = varnames{i};
    bad = ismember(var,badchar);
    var(bad)=[];
    
    %if first character is not a letter, make it so
    if ~isletter(var(1))
        var = ['n' var];
    end
    
    varnames{i} = var;
end

%%%%%PARSE DATA%%%%%
data2(data2==miss | data2==llod | data2==ulod) = NaN; %replace bad data
data2 = data2.*repmat(scale,size(data2,1),1); %scale

V.header = header;

for i=1:numvar1+numvar2
    
    if i <= 7
        V.(varnames{i}) = data1(:,i);
    else
        V.(varnames{i}) = data2(:,i-7);
    end
    
end
