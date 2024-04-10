% clc
% clear

%% adding secondry stim file to stim file



%% Combining conditions eeg data
dataMainFolder = 'E:\Bonnie\Bonnie\Autism_Data\dataCND\';
condfiles = dir([dataMainFolder,'/Cond*']);

eegFilenames = dir([dataMainFolder,'/Cond1/Group*/','pre_pre_dataSub*.mat']);

neegfiles = length(eegFilenames);

rootsavepath =  ['E:\Bonnie\Bonnie\Autism_Data\dataCND\CombCond']

for n=1:neegfiles

    subdataname = eegFilenames(n).name

    catdata = dir([dataMainFolder,'/Cond*/Group*/',subdataname])

    ncond = length(catdata)
    data = []
    extdata=[]
    for cond = 1:ncond

        eegfile = fullfile(catdata(cond).folder,catdata(cond).name)

        load(eegfile,'eeg')



        data= [data, eeg.data];
        extdata = [extdata,eeg.extChan{1,1}.data];
        emptytrials = cellfun(@isempty,data);

        skiptrial = find(emptytrials==1)

        if any(skiptrial(:)==1)
            disp('*******************************************************************')
        end


    end

    eeg.data = data;
    eeg.extChan{1,1}.data=extdata;
    savepath =  ['E:\Bonnie\Bonnie\Autism_Data\dataCND\CombCond\',catdata(cond).name]
    save(savepath,'eeg')
    clear data extdata
end


%% preprocessed
%%% re movingexternal channels
dataMainFolder = 'E:\Bonnie\Bonnie\Autism_Data\dataCND';
dataCNDSubfolder = '/Cond4/';
eegFilenames = dir([dataMainFolder,dataCNDSubfolder,'pre_dataSub*.mat']);

nsubs=length(eegFilenames)
for sub=1:nsubs
    % Loading preprocessed EEG
    eegPreFilename = [dataMainFolder,dataCNDSubfolder,eegFilenames(sub).name]; %%%
    savepath =  [dataMainFolder,dataCNDSubfolder,'pre_',eegFilenames(sub).name]
    disp(['Loading preprocessed EEG data:',eegFilenames(sub).name])

    load(eegPreFilename,'eeg')
    
    emptytrials = cellfun(@isempty,eeg.data); %% getting index of trials with no data

    skiptrial = find(emptytrials==1);

    extch.data{1,length(eeg.data)}=[]  
    
    for tr = 1:length(eeg.data)
        if any(skiptrial(:)==tr) %%% skipping empty trails
            
            continue
        end
        extch.data{1,tr} = eeg.data{1,tr}(:,33:34); %%% putting last two electrodes in the extch.data field
        eeg.data{1,tr} = eeg.data{1,tr}(:,1:32) %%% removing last two channels from eeg structur
    end
    extch.description='Mastoids'
    eeg.extChan{1,1}=extch;%%% putting last two electrodes in the eeg.extChan structure

    save(savepath,'eeg')
    clear extch
end
%% Parameters - Natural speech listening experiment
dataMainFolder = 'E:\Bonnie\Bonnie\Autism_Data\dataCND\';
dataCNDSubfolder = '/Co*/Group*/';
reRefType = 'Mastoids'; % or 'Avg' or Mastoids'
bandpassFilterRange = [1,8]; % Hz (indicate 0 to avoid running the low-pass
                          % or high-pass filters or both)
                          % e.g., [0,8] will apply only a low-pass filter
                          % at 8 Hz
downFs = 128; % Hz. *** fs/downFs must be an integer value ***

eegFilenames = dir([dataMainFolder,dataCNDSubfolder,'pre_pre_dataSub*.mat']);
nSubs = length(eegFilenames);
%% Preprocess EEG - Bonnie

for sub = 1:nSubs
    % Loading EEG data
    eegFilename = [eegFilenames(sub).folder,'\',eegFilenames(sub).name];
    disp(['Loading EEG data: ',eegFilenames(sub).name])
    load(eegFilename,'eeg') % loading eeg
    eeg = cndNewOp(eeg,'Load'); % Saving the processing pipeline in the eeg struct

    stimFilename = [eegFilenames(sub).folder,'\','vall_CondStim.mat']; %%% be carefull
    disp(['Loading preprocessed stim data:','vall_CondStim.mat'])
    load(stimFilename,'stim')


    emptytrials = cellfun(@isempty,eeg.data);

    eeg.extChan{1,1}.data(emptytrials)=[];
    eeg.data(emptytrials)=[]
    stim.data(:,emptytrials)=[];

    stim=cndNormalise(stim,[1,2,6])

    for tr=1:size(stim.data,2)
 
        stim.data{7,tr}=cat(2,stim.data{1,tr},stim.data{2,tr})
        stim.data{8,tr}=cat(2,stim.data{2,tr},stim.data{3,tr})
        stim.data{9,tr}=cat(2,stim.data{1,tr},stim.data{2,tr},stim.data{4,tr},stim.data{5,tr},stim.data{6,tr})

    end

    
    stimPreFilename = [eegFilenames(sub).folder,'\','dataStim_',eegFilenames(sub).name];
    disp(['Saving preprocessed stim data: datastim',eegFilenames(sub).name])
    % stim.data = cellfun(@transpose,stim.data,'UniformOutput',false) %% transposing stim.data arrays
    save(stimPreFilename,'stim')


end
%%
for i=1
 % 
 % 
 %    %%% geeting envelope derivitive:
 %    stim.names{1,6}="B_Envelope"
 %    stim.names{1,7}="wordonset"
 %    tim.names{1,8}="phonemeonset"
 %    for tr=1:size(stim.data,2)
 %    %%%%% my envelope derivative
 %        env_tr=stim.data{1,tr};
 % 
 %        env_tr(env_tr<0)=0;
 % 
 %        stim.data{1,tr}=env_tr;
 % 
 %        env_der = diff(env_tr);
 % 
 %        env_der(env_der<0)=0;
 % 
 %        env_der(end+1)=0;
 % 
 %        stim.data{9,tr}=env_der;
 % %%%%% bonnie envelope derivative envelope derivative
 %        env_tr=stim.data{6,tr};
 % 
 %        env_tr(env_tr<0)=0;
 % 
 %        stim.data{6,tr}=env_tr;
 % 
 %        env_der = diff(env_tr);
 % 
 %        env_der(env_der<0)=0;
 % 
 %        env_der(end+1)=0;
 % 
 %        stim.data{10,tr}=env_der;
 % 
 %    end
 %    stim.names{1,9}=" my Env' after resample"
 %    stim.names{1,10}="Bonnie Env' after resample"
 %    % stim.names{1,2}='Envelope derivitive resample'
 %    % stim.names{1,3}='Envelope derivitive downsample'
 %    % stim.names{1,9}="Env' after downsampling"
 %    % %%%% getting Amplitude binned envelopes
 %    % % 
 % 
 %    %%%%%%%% my envelope binned
 %    for tr=1:size((stim.data),2)
 %        env_tr=stim.data{1,tr};
 %        env_tr(env_tr<0)=0;
 %        env_ab=10*log10(env_tr); %%%% dont know why am doing this (Ed. lalor 2019 eneuro)
 %        edge =[0:10:40]
 %        binindx=discretize(env_ab,edge);
 %        ABE = zeros(length(edge)-1,length(binindx));
 %        for i=1:max(binindx)
 %            envindx=find(binindx==i);
 %            ABE(i,envindx)=env_ab(envindx)/max(env_ab(envindx));
 %        end
 % 
 %        stim.data{11,tr}=ABE'
 %    end
 %    stim.names{1,11}="my AB_Env after resample"
 % 
 %    %%%%%%%% Bonnie envelope binned
 %    for tr=1:size((stim.data),2)
 %        env_tr=stim.data{6,tr};
 %        env_tr(env_tr<0)=0;
 %        env_ab=-10*log10(env_tr); %%%% dont know why am doing this (Ed. lalor 2019 eneuro)
 %        edge =[0:10:40]
 %        binindx=discretize(env_ab,edge);
 %        ABE = zeros(length(edge)-1,length(binindx));
 %        for i=1:max(binindx)
 %            envindx=find(binindx==i);
 %            ABE(i,envindx)=env_ab(envindx)/max(env_ab(envindx));
 %        end
 % 
 %        stim.data{12,tr}=ABE'
 %    end
 %    stim.names{1,12}="bonnie AB_Env after resample"
 % 
 % 
 %    stim=cndNormalise(stim,[1,2,3,6,9,10]);
 % 
 %    for tr=1:size((stim.data),2)
 %        %%%%%% my features
 % 
 %        stim.data{13,tr} = cat(2,stim.data{1,tr},stim.data{2,tr},stim.data{4,tr});
 %        stim.data{14,tr} = cat(2,stim.data{1,tr},stim.data{3,tr},stim.data{5,tr});
 %        stim.data{15,tr} = cat(2,stim.data{1,tr},stim.data{9,tr},stim.data{11,tr});
 % 
 %        stim.data{16,tr} = cat(2,stim.data{1,tr},stim.data{2,tr});
 %        stim.data{17,tr} = cat(2,stim.data{1,tr},stim.data{3,tr});
 %        stim.data{18,tr} = cat(2,stim.data{1,tr},stim.data{9,tr});
 % 
 %        %%%%%% Bonnie features
 %        stim.data{19,tr} = cat(2,stim.data{6,tr},stim.data{10,tr},stim.data{12,tr});
 %        stim.data{20,tr} = cat(2,stim.data{6,tr},stim.data{10,tr});
 % 
 %        stim.data{21,tr} = cat(2,stim.data{6,tr},stim.data{10,tr},stim.data{7,tr},stim.data{8,tr});
 % 
 % 
 %    end
 %    %%% my feature names
 %    stim.names{1,13}="Env + Env'_resample + AB_Env_resample "
 %    stim.names{1,14}="Env + Env'_downsample + AB_Env_downsample"
 %    stim.names{1,15}="Env +Env' after resample + AB_Env after resample"
 % 
 %    stim.names{1,16}="Env + Env'_resample"
 %    stim.names{1,17}="Env + Env'_downsample"
 %    stim.names{1,18}="Env +Env' after resample"
 % 
 %    %%% Bonnie feature names
 %    stim.names{1,19}="B_Env + B_Env' after resample + B_AB_Env after resample"
 %    stim.names{1,20}="B_Env + B_Env' after resample"
 % 
 %    stim.names{1,21}="B_Env + B_Env' after resample + wordonset + phonemeonset"



    % Filtering - LPF (low-pass filter)
    % if bandpassFilterRange(2) > 0
    %     hd = getLPFilt(eeg.fs,bandpassFilterRange(2));
    % 
    %     % Filtering each trial/run with a cellfun statement
    %     eeg.data = cellfun(@(x) filtfilthd(hd,x),eeg.data,'UniformOutput',false);
    % 
    %     % Filtering external channels
    %     if isfield(eeg,'extChan')
    %         for extIdx = 1:length(eeg.extChan)
    %             eeg.extChan{extIdx}.data = cellfun(@(x) filtfilthd(hd,x),eeg.extChan{extIdx}.data,'UniformOutput',false);
    %         end
    %     end
    % 
    %     eeg = cndNewOp(eeg,'LPF');
    % end
    
    % Downsampling EEG and external channels
    eeg = cndDownsample(eeg,downFs);
    
    
    % Filtering - HPF (high-pass filter)
    if bandpassFilterRange(1) > 0 
        hd = getHPFilt(eeg.fs,bandpassFilterRange(1));

        % Filtering EEG data
        eeg.data = cellfun(@(x) filtfilthd(hd,x),eeg.data,'UniformOutput',false);

        % Filtering external channels
        if isfield(eeg,'extChan')
            for extIdx = 1:length(eeg.extChan)
                eeg.extChan{extIdx}.data = cellfun(@(x) filtfilthd(hd,x),eeg.extChan{extIdx}.data,'UniformOutput',false);
            end  
        end

        eeg = cndNewOp(eeg,'HPF');
    end


    %Replacing bad channels
    if isfield(eeg,'chanlocs')
        for tr = 1:length(eeg.data)
            eeg.data{tr} = removeBadChannels(eeg.data{tr}, eeg.chanlocs);
        end
    end

%     % Re-referencing EEG data
    eeg = cndReref(eeg,reRefType);
    
    % Removing initial padding (specific to this dataset)
    if isfield(eeg,'paddingStartSample')
        for tr = 1:length(eeg.data)
            eeg.data{tr} = eeg.data{tr}(eeg.paddingStartSample+1:end,:);
%             for extIdx = 1:length(eeg.extChan)
%                 eeg.extChan{extIdx}.data = eeg.extChan{extIdx}.data{tr}(1+eeg.paddingStartSample+1:end,:);
%             end
        end
    end

    
    % Saving preprocessed data
    eegPreFilename = [eegFilenames(sub).folder,'\','pre1hz_',eegFilenames(sub).name]; 
    disp(['Saving preprocessed MEG data: pre1hz_',eegFilenames(sub).name])
    save(eegPreFilename,'eeg')


end


%% RUNNING TRF ANALYSIS
clear
dataMainFolder = 'E:\Bonnie\Bonnie\Autism_Data\dataCND\';

% TRF hyperparameter
%
% TRF hyperparameters
tmin = -200;
tmax = 600;
lambdas = [1e-8,1e-9,1e-10,1e-6,1e-4,1e-2,1e0,1e2,1e4,1e6,1e8,1e9,1e10,1e11,1e12,1e13];
% lambdas = 1e4;
dirTRF = 1; % Forward TRF model
stimidx=[6,9] %%%%  refer to preprocessing for correct index
groups=["Group1","Group2"]; 
cond=["Cond1","Cond2","Cond3","Cond4","CombCond"]
%%
for g=groups

    for c=cond

        dataCNDSubfolder = char(fullfile(c,g,'/'));
        eegFilenames = dir([dataMainFolder,dataCNDSubfolder,'pre1hz_pre_pre_dataSub*.mat']);
        stimFilenames = dir([dataMainFolder,dataCNDSubfolder,'dataStim_pre_pre_dataSub*.mat'])
        nSubs = length(eegFilenames)

        for sid=stimidx
            
            save_path=[]

            for sub = 1:nSubs
                % Loading preprocessed EEG
                eegPreFilename = [eegFilenames(sub).folder,'/',eegFilenames(sub).name]; %%%
                disp(['Loading preprocessed EEG data:',eegFilenames(sub).name])

                load(eegPreFilename,'eeg')

                stimPreFilename = [stimFilenames(sub).folder,'/',stimFilenames(sub).name]; %%% ; %%%
                disp(['Loading preprocessed stim data:',stimFilenames(sub).name])
                load(stimPreFilename,'stim')



                if all(g=='Group1')
                    savepath = fullfile([eegFilenames(sub).folder, ...
                        '\results','\vall_',num2str(sid),'_mTrfResultsgroup1.mat'])
                end

                if all(g=='Group2')
                    savepath = fullfile([eegFilenames(sub).folder, ...
                        '\results','\vall_',num2str(sid),'_mTrfResultsgroup2.mat'])
                end


                % stim.data = cellfun(@transpose,stim.data,'UniformOutput',false) %% transposing stim.data arrays
                %%%remove empty trials
                % emptytrials = cellfun(@isempty,eeg.data)
                %
                % eeg.data(emptytrials)=[];
                % stim.data(emptytrials)=[];


                % Making sure that stim and neural data have the same length

                stimFeature = stim
                stimFeature.data=stimFeature.data(sid,:)% refer to preprocessing for names

                
                featname=char(stimFeature.names{sid});



                if eeg.fs ~= stimFeature.fs
                    disp('Error: EEG and STIM have different sampling frequency')
                    return
                end
                if length(eeg.data) ~= length(stimFeature.data)
                    disp('Error: EEG.data and STIM.data have different number of trials')
                    return
                end
                for tr = 1:length(stimFeature.data)
                    envLen = size(stimFeature.data{tr},1);
                    eegLen = size(eeg.data{tr},1);
                    minLen = min(envLen,eegLen);
                    stimFeature.data{tr} = double(stimFeature.data{tr}(1:minLen,:));
                    eeg.data{tr} = double(eeg.data{tr}(1:minLen,:));
                end

                % Normalising EEG data
                clear tmpEnv tmpEeg
                
                tmpEeg = eeg.data{1};
                for tr = 2:length(stimFeature.data) % getting all values
                    
                    tmpEeg = cat(1,tmpEeg,eeg.data{tr});
                end
                
                normFactorEeg = std(tmpEeg(:)); clear tmpEeg;
                for tr = 1:length(stimFeature.data) % normalisation
                    
                    eeg.data{tr} = eeg.data{tr}/normFactorEeg;
                end

                % TRF - Compute model weights
                disp('Running mTRFcrossval')
                [stats,t] = mTRFcrossval(stimFeature.data,eeg.data,eeg.fs,dirTRF,tmin,tmax,lambdas,'verbose',0);
                [maxR,bestLambda] = max(squeeze(mean(mean(stats.r,1),3)));
                disp(['r = ',num2str(maxR)])
                rAll(sub) = maxR;
                statsAll(sub)=stats
                best_lambda(sub) = lambdas(bestLambda);
                disp('Running mTRFtrain')
                model = mTRFtrain(stimFeature.data,eeg.data,eeg.fs,dirTRF,tmin,tmax,lambdas(bestLambda),'verbose',0);

                modelAll(sub) = model;

                % Plot average TRF
                normFlag = 1;
                avgModel = mTRFmodelAvg(modelAll,normFlag);
                disp(['Mean r = ',num2str(mean(rAll))])


            end
            disp('*************** DONE *****************')
            savepath
            save(savepath,'rAll','modelAll','avgModel','best_lambda','statsAll','lambdas','featname','bestLambda')
            clear rAll modelAll avgModel best_lambda statsAll featname bestLambda
        end
        disp('*************** DONE *****************')
    end
end