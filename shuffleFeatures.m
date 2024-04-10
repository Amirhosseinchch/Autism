function stim = shuffleFeatures(stim, featureSetIdx, featureIdxs, varargin)
%SHUFFLEFEATURES Shuffle stimulus features in a stimulus struct
%   STIM = SHUFFLEFEATURES(STIM, FEATURESETIDX, FEATUREIDXS, VARARGIN) finds 
%   features at featureIdxs in feature set at featureSetIdx in stim and 
%   creates a new feature set where selected features are shuffled 
%   according to one of the possible shuffling methods.
%
%   [...] = SHUFFLEFEATURES(...,'PARAM1',VAL1,'PARAM2',VAL2,...) specifies
%   additional parameters and their values. Valid parameters are the
%   following:
%
%       Parameter   Value
%       'method'    Shuffling method to use:
%                       'permute'     shuffle by value (default): keeping the
%                                   timing fixed shuffles the corresponding
%                                   values
%                       'timing'    jitter timings: keeping the
%                                   values ordered but shuffles the timing
%                                   resulting in jitter in onsets
%                                   values
%                       'circ'      circular shift: moves all values right
%                                   by 'shiftBy'. Nb.: not appropriate for
%                                   for music
%                       'absolute'  randomised shuffle: like 'timing' but
%                                   it doesn't preserve the order of the
%                                   values
%       'shiftBy'   Used only for 'circ' specifies the samples to shift by.
%       'seed'      Random seed to use while shuffling. Use non-negative
%                   integer or "shuffle.
%       'name'      The name by which to save the new shuffled feature-set.
% Author name: Martin Winchester


arg = parsevarargin(varargin);
rng(arg.seed)
feature = stim.data(featureSetIdx,:);
shuffled_features=cell(size(feature));
num_of_feat = size(stim.data,1);
if ~isempty(arg.name)
    stim.names{num_of_feat+1} = char(arg.name);
else
    stim.names{num_of_feat+1} = char(stim.names{featureSetIdx} + "_shuffled");
end
if strcmp(arg.method, 'permute')
    for tr=1:size(stim.data,2)
        shuffled_features{tr} = feature{tr};
        if size(shuffled_features{tr},2) == 1
            featureIdxs = 1;
        end
        for featureIdx=featureIdxs
            shuffled_features{tr}(:,featureIdx) = 0;
            nonZeroValues = nonzeros(feature{tr}(:,featureIdx));
            nonZeroIdxs = find(feature{tr}(:,featureIdx));
            nonZeroValues = nonZeroValues(randperm(numel(nonZeroValues)));
            shuffled_features{tr}(nonZeroIdxs,featureIdx) = nonZeroValues;
        end
    end
    stim.data(num_of_feat+1, :) = shuffled_features;
elseif strcmp(arg.method, 'timing')
    for tr=1:size(stim.data,2)
        shuffled_features{tr} = feature{tr};
        if size(shuffled_features{tr},2) == 1
            featureIdxs = 1;
        end
        for featureIdx=featureIdxs
            shuffled_features{tr}(:,featureIdx) = 0;
            nonZeroValues = nonzeros(feature{tr}(:,featureIdx));
            timings = sort(randperm(size(feature{tr}, 1),length(nonZeroValues)));
            shuffled_features{tr}(timings,featureIdx) = nonZeroValues;
        end
    end
    stim.data(num_of_feat+1, :) = shuffled_features;
elseif strcmp(arg.method, 'circ')
    shift_by = arg.shiftBy;
    for tr=1:size(stim.data,2)
        shuffled_features{tr} = feature{tr};
        if size(shuffled_features{tr},2) == 1
            featureIdxs = 1;
        end
        for featureIdx=featureIdxs
            shuffled_features{tr}(:,featureIdx) = 0;
            nonZeroValues = nonzeros(feature{tr}(:,featureIdx));
            nonZeroIdxs = find(feature{tr}(:,featureIdx));
            % nonZeroValues = nonZeroValues(randperm(numel(nonZeroValues)));
            shuffled_features{tr}(circshift(nonZeroIdxs,-shift_by),featureIdx) = nonZeroValues;
        end
    end
    stim.data(num_of_feat+1, :) = shuffled_features;
elseif strcmp(arg.method, 'absolute')
    for tr=1:size(stim.data,2)
        shuffled_features{tr} = feature{tr};
        if size(shuffled_features{tr},2) == 1
            featureIdxs = 1;
        end
        for featureIdx=featureIdxs
            shuffled_features{tr}(:,featureIdx) = 0;
            nonZeroValues = nonzeros(feature{tr}(:,featureIdx));
            timings = randperm(size(feature{tr}, 1),length(nonZeroValues));
            shuffled_features{tr}(timings,featureIdx) = nonZeroValues;
        end
    end
    stim.data(num_of_feat+1, :) = shuffled_features;
end
end


function arg = parsevarargin(varargin)
%PARSEVARARGIN  Parse input arguments.
%   [PARAM1,PARAM2,...] = PARSEVARARGIN('PARAM1',VAL1,'PARAM2',VAL2,...)
%   parses the input arguments of the main function.

% Create parser object
p = inputParser;

% method of shuffling
shuffleOptions = {'permute','timing','circ','absolute'};
validFcn = @(x) any(validatestring(x,shuffleOptions));
addParameter(p,'method','permute',validFcn);

% name of shuffled feature set
addParameter(p,'name','');

% samples to circshift by
errorMsg = 'It must be a positive integer scalar. Can only be used if method is "circ".';
validFcn = @(x) assert(isnumeric(x)&&isscalar(x)&&any(strcmp('circ', varargin{1,1})),errorMsg);
addParameter(p,'shiftBy',1,validFcn);

% samples to circshift by
errorMsg = 'Seed must be non-negative integer or "shuffle".';
validFcn = @(x) assert((isnumeric(x)&&x>=0)||strcmp(x, "shuffle"),errorMsg);
addParameter(p,'seed',"shuffle",validFcn);

% Parse input arguments
parse(p,varargin{1,1}{:});
arg = p.Results;
end