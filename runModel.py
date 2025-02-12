import os
import pandas as pd

rootDir = '/home/tsa/Projects/250115-BirdID-Model-Zoo/BirdID-Model-Zoo/'

outputRootDir = rootDir + 'TestOutputsTemp/'


## Choose model

#modelID = 'birdnet_v2.4'
#modelID = 'birdnet_v2.2'

#modelID = 'avesecho_v1.3.0'
#modelID = 'avesecho_v1.3.0_transformer'

#modelID = 'birdid-europe254-medium'
modelID = 'birdid-europe254-large'



# Get some input files
inputDir = rootDir + 'TestFiles/'

filePaths = [
    inputDir + 'LusMeg00027.mp3',
    inputDir + 'LusMeg00028.mp3'
    ]




def getModelResults(modelID, outputRootDir, listOfFilePathsOrFolder, instanceIndex=None, removeContainer=True):

    # Define Docker run command depending on model

    if modelID == 'birdnet_v2.4':
        dockerInputDir = '/input'
        dockerOutputDir = '/output'
        #dockerParamString = 'birdnet-v24 -m birdnet_analyzer.analyze --i input --o output --min_conf 0.01 --overlap 2.0 --rtype csv'
        dockerParamString = 'ghcr.io/mfn-berlin/birdnet-v24 -m birdnet_analyzer.analyze --i input --o output --min_conf 0.01 --overlap 2.0 --rtype csv'
    
    if modelID == 'birdnet_v2.2':
        dockerInputDir = '/input'
        dockerOutputDir = '/output'
        #dockerParamString = 'birdnet-v22 -m birdnet_analyzer.analyze --i input --o output --min_conf 0.01 --overlap 2.0 --rtype csv'
        dockerParamString = 'ghcr.io/mfn-berlin/birdnet-v22 -m birdnet_analyzer.analyze --i input --o output --min_conf 0.01 --overlap 2.0 --rtype csv'


    if modelID == 'avesecho_v1.3.0':
        dockerInputDir = '/app/audio'
        dockerOutputDir = '/app/outputs'
        dockerParamString = "--shm-size=4g --gpus all -e CUDA_VISIBLE_DEVICES=0 registry.gitlab.com/arise-biodiversity/dsi/algorithms/avesecho-v1/avesechov1:v1.3.0 --i audio --model_name 'fc' --add_csv --mconf 0.01"

    if modelID == 'avesecho_v1.3.0_transformer':
        dockerInputDir = '/app/audio'
        dockerOutputDir = '/app/outputs'
        # --model_name 'passt'
        dockerParamString = "--shm-size=4g --gpus all -e CUDA_VISIBLE_DEVICES=0 registry.gitlab.com/arise-biodiversity/dsi/algorithms/avesecho-v1/avesechov1:v1.3.0 --i audio --model_name 'passt' --add_csv --mconf 0.01"


    if modelID == 'birdid-europe254-medium':
        dockerInputDir = '/input'
        dockerOutputDir = '/output'
        #dockerParamString = "--gpus device=0 --ipc=host birdid-europe254-v250212-1 python inference.py -i /input -o /output --fileOutputFormats labels_csv --minConfidence 0.01 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize medium"
        dockerParamString = "--gpus device=0 --ipc=host birdid-europe254-v250212-1 python inference.py -i /input -o /output --fileOutputFormats labels_csv --minConfidence 0.01 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize medium"

    if modelID == 'birdid-europe254-large':
        dockerInputDir = '/input'
        dockerOutputDir = '/output'
        #dockerParamString = "--gpus device=0 --ipc=host birdid-europe254-v250212-1 python inference.py -i /input -o /output --fileOutputFormats labels_csv --minConfidence 0.01 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize large"
        dockerParamString = "--gpus device=0 --ipc=host ghcr.io/mfn-berlin/birdid-europe254-v250212-1 python inference.py -i /input -o /output --fileOutputFormats labels_csv --minConfidence 0.01 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize large"



    # ToDo: Maybe add instance index if multiple instances of the same model are run in parallel
    containerName = modelID
    if instanceIndex is not None:
        containerName += '_' + str(instanceIndex)
    
    dockerRunString = 'docker run --name ' + containerName

    if removeContainer:
        dockerRunString += ' --rm'


    # Create outputDir (subfolder in outputRootDir) for temporary results
    outputDir = outputRootDir + modelID + '/'
    os.makedirs(outputDir, exist_ok=True)
    
    # Check if listOfFilePathsOrFolder is a list of files or a folder
    if isinstance(listOfFilePathsOrFolder, str) and os.path.isdir(listOfFilePathsOrFolder):
        inputDir = listOfFilePathsOrFolder
        os.system(dockerRunString + ' -v ' + inputDir + ':' + dockerInputDir +' -v ' + outputDir + ':' + dockerOutputDir + ' ' + dockerParamString)
    else:
        filePaths = listOfFilePathsOrFolder
        # Construct the Docker run command with bind mounts for each file
        dockerCommand = dockerRunString
        inputMounts = ''
        for filePath in filePaths:
            fileName = os.path.basename(filePath)
            inputMounts += ' -v ' + filePath + ':' + dockerInputDir + '/' + fileName # + '/' is important here
        dockerCommand += inputMounts + ' -v ' + outputDir + ':' + dockerOutputDir + ' ' + dockerParamString
        print('Length of docker command: ' + str(len(dockerCommand)))
        os.system(dockerCommand)



def postProcessResults(modelID, outputRootDir, removeTemporaryResultFiles=False):

    # Load labelToId mapping table
    path = rootDir + 'LabelToIdMappings/' + modelID + '.csv'
    df_labelToId = pd.read_csv(path)

    # Collect results from all csv files in outputDir
    outputDir = outputRootDir + modelID + '/' 
    df_list = []  # List to collect all DataFrames
    for file in os.listdir(outputDir):
        if file.endswith('.csv'):
            print('Post-processing ' + file)
            # Read the csv file
            df_perFile = pd.read_csv(outputDir + file)
            df_list.append(df_perFile)

    
    if not df_list:
        print('No csv files found in ' + outputDir)
        return
    
    # Concatenate all DataFrames into one
    df = pd.concat(df_list, ignore_index=True)
    

    ## Post process results depending on the model
    
    if modelID == 'birdnet_v2.4' or modelID == 'birdnet_v2.2':

        '''
        Original format:

                    Start (s)  End (s)        Scientific name         Common name  Confidence                   File
        0         0.0      3.0  Luscinia megarhynchos  Common Nightingale      0.9317  input/LusMeg00027.mp3
        1         0.0      3.0   Atrichornis clamosus    Noisy Scrub-bird      0.0372  input/LusMeg00027.mp3
        2         0.0      3.0      Luscinia luscinia  Thrush Nightingale      0.0289  input/LusMeg00027.mp3
        3         1.0      4.0  Luscinia megarhynchos  Common Nightingale      0.9726  input/LusMeg00027.mp3
        '''


        # Add col filename (col File: input/LusMeg00027.mp3 --> LusMeg00027.mp3)
        df['filename'] = df['File'].apply(lambda x: os.path.basename(x))

        ## Rename cols: 
        # Start (s) --> start_time
        # End (e) --> end_time
        # Confidence --> confidence
        df.rename(columns={'Start (s)': 'start_time', 'End (s)': 'end_time', 'Confidence': 'confidence'}, inplace=True)

        # Add col label_id by merging with df_labelToId (Scientific name is srcLabelName)
        df = pd.merge(df, df_labelToId, how='left', left_on='Scientific name', right_on='srcLabelName')

        # Remove unnecessary cols: Scientific name, Common name, File, srcLabelIx
        df.drop(columns=['Scientific name', 'Common name', 'File', 'srcLabelIx'], inplace=True)

        # Rename cols: srcLabelName --> label_model, dstLabelId --> label_id
        df.rename(columns={'srcLabelName': 'label_model', 'dstLabelId': 'label_id'}, inplace=True)

        # Add col model_id
        df['model_id'] = modelID

        # Reorder cols: model_id, filename, start_time, end_time, confidence, label_model, label_id
        df = df[['model_id', 'filename', 'start_time', 'end_time', 'confidence', 'label_model', 'label_id']]


    if modelID == 'avesecho_v1.3.0' or modelID == 'avesecho_v1.3.0_transformer':

        '''
        Original format:
        
                Begin Time End Time             File                                         Prediction     Score
        0        0:00     0:03  LusMeg00028.mp3                       MeadowPipit_Anthus pratensis  0.010736
        1        0:00     0:03  LusMeg00028.mp3           GreatSpottedWoodpecker_Dendrocopos major  0.015183
        2        0:00     0:03  LusMeg00028.mp3              MelodiousWarbler_Hippolais polyglotta  0.011034
        '''


        ## Rename cols: 
        # Begin Time --> start_time
        # End Time --> end_time
        # Score --> confidence
        # File --> filename
        df.rename(columns={'Begin Time': 'start_time', 'End Time': 'end_time', 'Score': 'confidence', 'File': 'filename'}, inplace=True)    

        # Add col label_id by merging with df_labelToId (Prediction is srcLabelName)
        df = pd.merge(df, df_labelToId, how='left', left_on='Prediction', right_on='srcLabelName')

        # Remove unnecessary cols: Scientific name, Common name, File, srcLabelIx
        df.drop(columns=['srcLabelIx'], inplace=True)

        # Rename cols: srcLabelName --> label_model, dstLabelId --> label_id
        df.rename(columns={'srcLabelName': 'label_model', 'dstLabelId': 'label_id'}, inplace=True)

        # Add col model_id
        df['model_id'] = modelID

        # Reorder cols: model_id, filename, start_time, end_time, confidence, label_model, label_id
        df = df[['model_id', 'filename', 'start_time', 'end_time', 'confidence', 'label_model', 'label_id']]


    if modelID == 'birdid-europe254-medium' or modelID == 'birdid-europe254-large':

        '''
        Original format:
        
            startTime [s]  endTime [s]                 species  confidence                filePath
        0          0.00000      5.00000   Luscinia megarhynchos     0.94881  /input/LusMeg00027.mp3
        1          0.00000      5.00000  Phylloscopus collybita     0.03471  /input/LusMeg00027.mp3
        2          0.00000      5.00000       Fringilla coelebs     0.03384  /input/LusMeg00027.mp3
        '''

        # Add col filename (col filePath: /input/LusMeg00027.mp3 --> LusMeg00027.mp3)
        df['filename'] = df['filePath'].apply(lambda x: os.path.basename(x))

        ## Rename cols: 
        # startTime [s] --> start_time
        # endTime [s] --> end_time
        df.rename(columns={'startTime [s]': 'start_time', 'endTime [s]': 'end_time'}, inplace=True)


        # Add col label_id by merging with df_labelToId (species is srcLabelName)
        df = pd.merge(df, df_labelToId, how='left', left_on='species', right_on='srcLabelName')

        # Remove unnecessary cols: srcLabelIx
        df.drop(columns=['srcLabelIx'], inplace=True)

        # Rename cols: srcLabelName --> label_model, dstLabelId --> label_id
        df.rename(columns={'srcLabelName': 'label_model', 'dstLabelId': 'label_id'}, inplace=True)

        # Add col model_id
        df['model_id'] = modelID

        # Reorder cols: model_id, filename, start_time, end_time, confidence, label_model, label_id
        df = df[['model_id', 'filename', 'start_time', 'end_time', 'confidence', 'label_model', 'label_id']]
        


    
    print(df)

    '''
                model_id         filename  start_time  end_time  confidence            label_model  label_id
    0   birdnet_v2.4  LusMeg00027.mp3         0.0       3.0      0.9317  Luscinia megarhynchos      3307
    1   birdnet_v2.4  LusMeg00027.mp3         0.0       3.0      0.0372   Atrichornis clamosus       594
    2   birdnet_v2.4  LusMeg00027.mp3         0.0       3.0      0.0289      Luscinia luscinia      3306
    '''


    # Save to csv
    dstPath = outputRootDir + modelID + '.csv'
    df.to_csv(dstPath, index=False)


    ## Do something with the result csv file
    # Maybe reduce confidence values to 5 decimal places, ...
    # Add results to DB, ...

    ## Delete outputDir and its contents
    if removeTemporaryResultFiles:
        os.system('rm -rf ' + outputDir)



    


################################################################################################################
################################################################################################################
################################################################################################################

# Create temp outputRootDir
os.makedirs(outputRootDir, exist_ok=True)

# Run model
#getModelResults(modelID, outputRootDir, inputDir) # Pass folder
getModelResults(modelID, outputRootDir, filePaths) # Pass list of files

# Post process results
postProcessResults(modelID, outputRootDir)
    
    

print('Done.')