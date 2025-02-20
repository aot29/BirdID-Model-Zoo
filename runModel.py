import os
import pandas as pd
from multiprocessing import cpu_count, freeze_support, Pool
import argparse



## Default values

# Model ID

#modelID = 'birdnet_v2.4'
#modelID = 'birdnet_v2.2'

#modelID = 'avesecho_v1.3.0'
#modelID = 'avesecho_v1.3.0_transformer'

modelID = 'birdid-europe254-medium'
#modelID = 'birdid-europe254-large'


# Output root directory
rootDir = os.path.dirname(os.path.abspath(__file__)) + '/'
outputRootDir = rootDir + 'TestOutputsTemp/'

# Path to text file with list of file paths
inputTextFilePath = rootDir + 'inputPaths.txt'

workerIx = None
nFilesPerBatch = 100
stepDuration = 2.0
removeContainer = True

# Docker config
dockerConfig = {
    'birdnet_v2.4': {
        'options': '--shm-size=4g --ipc=host',
        'inputDir': '/input',
        'outputDir': '/output',
        'image': 'ghcr.io/mfn-berlin/birdnet-v24',
        'command': '-m birdnet_analyzer.analyze --i input --o output --min_conf 0.01 --overlap 1.0 --rtype csv'
    },
    'birdnet_v2.2': {
        'options': '--shm-size=4g --ipc=host',
        'inputDir': '/input',
        'outputDir': '/output',
        'image': 'ghcr.io/mfn-berlin/birdnet-v22',
        'command': '-m birdnet_analyzer.analyze --i input --o output --min_conf 0.01 --overlap 1.0 --rtype csv'
    },
    'avesecho_v1.3.0': {
        'options': '--shm-size=4g --ipc=host --gpus device=0',
        'inputDir': '/app/audio',
        'outputDir': '/app/outputs',
        'image': 'registry.gitlab.com/arise-biodiversity/dsi/algorithms/avesecho-v1/avesechov1:v1.3.0',
        'command': "--i audio --model_name 'fc' --add_csv --mconf 0.01"
    },
    'avesecho_v1.3.0_transformer': {
        'options': '--shm-size=4g --ipc=host --gpus device=0',
        'inputDir': '/app/audio',
        'outputDir': '/app/outputs',
        'image': 'registry.gitlab.com/arise-biodiversity/dsi/algorithms/avesecho-v1/avesechov1:v1.3.0',
        'command': "--i audio --model_name 'passt' --add_csv --mconf 0.01"
    },
    'birdid-europe254-medium': {
        'options': '--shm-size=4g --ipc=host --gpus device=0',
        'inputDir': '/input',
        'outputDir': '/output',
        'image': 'ghcr.io/mfn-berlin/birdid-europe254-v250212-1',
        'command': 'python inference.py -i /input -o /output --fileOutputFormats labels_csv --minConfidence 0.01 --overlapInPerc 60 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize medium'
    },
    'birdid-europe254-large': {
        'options': '--shm-size=4g --gpus device=0 --ipc=host',
        'inputDir': '/input',
        'outputDir': '/output',
        'image': 'ghcr.io/mfn-berlin/birdid-europe254-v250212-1',
        'command': 'python inference.py -i /input -o /output --fileOutputFormats labels_csv --minConfidence 0.01 --overlapInPerc 60 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize large'
    }
}


# # Get some input files
# inputDir = rootDir + 'TestFiles/'

# filePaths = [
#     inputDir + 'LusMeg00027.mp3',
#     inputDir + 'LusMeg00028.mp3'
#     ]




def getModelResults(modelID, outputRootDir, listOfFilePathsOrFolder, workerIx=None, nFilesPerBatch=100, stepDuration=2.0, removeContainer=True):

    if modelID not in dockerConfig:
        raise ValueError(f"Unknown modelID: {modelID}")

    # Create outputDir (subfolder in outputRootDir) for temporary results
    outputDir = outputRootDir + modelID + '/'
    os.makedirs(outputDir, exist_ok=True)


    ## Define Docker run command string depending on model

    dockerCommand = 'docker run --name ' + modelID
    
    # Add instance index to name if multiple instances of the same model are run in parallel
    if workerIx is not None: 
        dockerCommand += '_' + str(workerIx)
    # Add option to remove container after run
    if removeContainer: 
        dockerCommand += ' --rm'


    
    ## Create inputMounts depending on whether listOfFilePathsOrFolder is a list of files or a folder

    # listOfFilePathsOrFolder is a folder
    if isinstance(listOfFilePathsOrFolder, str) and os.path.isdir(listOfFilePathsOrFolder):
        nBatches = 1
        inputIsFolder = True
    else:
        nBatches = (len(listOfFilePathsOrFolder) + nFilesPerBatch - 1) // nFilesPerBatch
        inputIsFolder = False

    # Loop over batches
    for batchIx in range(nBatches):

        if inputIsFolder:
            inputMounts = ' -v ' + listOfFilePathsOrFolder + ':' + dockerConfig[modelID]['inputDir']
        else:
            print('Batch:', batchIx)
            batchStartIx = batchIx * nFilesPerBatch
            batchEndIx = min((batchIx + 1) * nFilesPerBatch, len(listOfFilePathsOrFolder))
            batchFilePaths = listOfFilePathsOrFolder[batchStartIx:batchEndIx]

            # Construct the Docker run command with bind mounts for each file
            inputMounts = ''
            for filePath in batchFilePaths:
                fileName = os.path.basename(filePath)
                inputMounts += ' -v ' + filePath + ':' + dockerConfig[modelID]['inputDir'] + '/' + fileName # + '/' is important here


        dockerCommand += inputMounts
        dockerCommand += ' -v ' + outputDir + ':' + dockerConfig[modelID]['outputDir']
        dockerCommand += ' ' + dockerConfig[modelID]['options']
        dockerCommand += ' ' + dockerConfig[modelID]['image']

        # Modify command depending passed arguments on model
        command = dockerConfig[modelID]['command']

        if stepDuration:
            if modelID.startswith('birdnet'):
                overlap = 3.0 - stepDuration
                overlapArg = '--overlap ' + str(overlap)
                command = command.replace('--overlap 1.0', overlapArg)
            if modelID.startswith('birdid'):
                overlap = 5.0 - stepDuration
                overlapInPerc = int(overlap / 5.0 * 100)
                overlapArg = '--overlapInPerc ' + str(overlapInPerc)
                command = command.replace('--overlapInPerc 60', overlapArg)

        dockerCommand += ' ' + command

        #print('dockerCommand:\n', dockerCommand)
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

if __name__ == "__main__":


    # On Windows calling this function is necessary.
    # On Linux/OSX it does nothing.
    freeze_support()

    ## Parse arguments
    parser = argparse.ArgumentParser(description='Identify birds in audio files with various models.')

    parser.add_argument('-m', '--modelID', type=str, metavar='', default=modelID, help='Model ID. Defaults to ' + modelID)
    parser.add_argument('-o', '--outputRootDir', type=str, metavar='', default=outputRootDir, help='Output root directory. Defaults to ' + outputRootDir)

    parser.add_argument('-i', '--inputDirOrTextFilePath', type=str, metavar='', default=inputTextFilePath, help='Input directory or path of text file with list of file paths. Defaults to ' + inputTextFilePath)
    #parser.add_argument('--filePaths', nargs='+', help='List of file paths', required=False)

    parser.add_argument('-s', '--stepDuration', type=float, metavar='', default=2.0, help='Step duration in seconds. Defaults to ' + str(stepDuration))
    parser.add_argument('-w', '--workerIx', type=int, metavar='', default=workerIx, help='Worker index. Defaults to ' + str(workerIx))
    parser.add_argument('-n', '--nFilesPerBatch', type=int, metavar='', default=nFilesPerBatch, help='Number of files per batch. Defaults to ' + str(nFilesPerBatch))
    #parser.add_argument('-r', '--removeContainer', action='store_true', help='Remove container after run. Defaults to True')
    parser.add_argument('--removeTemporaryResultFiles', action='store_true', help='Remove temporary result files after post processing. Defaults to False')
    
    # To check and add
    #parser.add_argument('-b', '--nFilesPerBatch', type=int, metavar='', default=100, help='Number of files per batch. Defaults to 100')
    #parser.add_argument('-r', '--removeContainer', action='store_true', help='Remove container after run. Defaults to True')
    #parser.add_argument('--removeTemporaryResultFiles', action='store_true', help='Remove temporary result files after post processing. Defaults to False')

    args = parser.parse_args()

    ## Assign arguments to variables
    modelID = args.modelID
    outputRootDir = args.outputRootDir

    inputDirOrTextFilePath = args.inputDirOrTextFilePath
    stepDuration = args.stepDuration
    workerIx = args.workerIx
    nFilesPerBatch = args.nFilesPerBatch
    #removeContainer = args.removeContainer
    removeTemporaryResultFiles = args.removeTemporaryResultFiles

    # Check if inputDirOrTextFilePath is existing file or folder
    if not os.path.exists(inputDirOrTextFilePath):
        raise FileNotFoundError(f"File or folder not found: {inputDirOrTextFilePath}")


    if os.path.isfile(inputDirOrTextFilePath):
        with open(inputDirOrTextFilePath, 'r') as file:
            filePaths = file.readlines()
            listOfFilePathsOrFolder = [x.strip() for x in filePaths]
            nFilesToProcess = len(listOfFilePathsOrFolder)
            print('Number of files to process:', nFilesToProcess)
    else:
        listOfFilePathsOrFolder = inputDirOrTextFilePath
        # Todo: check if inputDirOrTextFilePath is a folder





    #print('listOfFilePathsOrFolder', listOfFilePathsOrFolder)

    # inputDir = args.inputDir
    # filePaths = args.filePaths



    ## Create temp outputRootDir
    # Append '/' to outputRootDir if not already there
    if outputRootDir[-1] != '/':
        outputRootDir += '/'
    os.makedirs(outputRootDir, exist_ok=True)

    # Run model
    getModelResults(modelID, outputRootDir, listOfFilePathsOrFolder, workerIx=workerIx, nFilesPerBatch=nFilesPerBatch, stepDuration=stepDuration, removeContainer=True)

    # Post process results
    postProcessResults(modelID, outputRootDir)
    
    

print('Done.')