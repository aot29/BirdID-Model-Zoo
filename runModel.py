import os
import pandas as pd
from multiprocessing import freeze_support
import argparse
from typing import Union, Optional
import time
import subprocess

# Get root directory of script
rootDir = os.path.dirname(os.path.abspath(__file__)) + "/"


## Default values

# Model ID
modelID = "birdnet_v2.2"  # birdnet_v2.2, birdnet_v2.4, avesecho_v1.3.0, avesecho_v1.3.0_transformer, birdid-europe254-medium, birdid-europe254-large

# Output root directory
outputRootDir = rootDir + "TestOutputsTemp/"

# Path to text file with list of file paths
inputTextFilePath = rootDir + "inputPaths.txt"


workerId = "0"
nFilesPerBatch = 100
batchSizeFiles = 16
nCpuWorkers = 8
batchSize = 16
gpuIx = 0  # None, 0, 1, ...

minConfidenceThreshold = 0.01
stepDuration = 2.0

sharedMemorySizeStr = "4g"  # None, '4g', '8g', ...

removeTemporaryResultFiles = False
fileOutputFormatsValid = ["csv", "excel", "pkl"]
fileOutputFormats = ["csv"]

removeContainer = True


# Docker config
dockerConfig = {
    "birdnet_v2.4": {
        "inputDir": "/input",
        "outputDir": "/output",
        "image": "ghcr.io/mfn-berlin/birdnet-v24",
        "command": "-m birdnet_analyzer.analyze --i input --o output --overlap 1.0 --rtype csv",
    },
    "birdnet_v2.2": {
        "inputDir": "/input",
        "outputDir": "/output",
        "image": "ghcr.io/mfn-berlin/birdnet-v22",
        "command": "-m birdnet_analyzer.analyze --i input --o output --overlap 1.0 --rtype csv",
    },
    "avesecho_v1.3.0": {
        "inputDir": "/app/audio",
        "outputDir": "/app/outputs",
        "image": "registry.gitlab.com/arise-biodiversity/dsi/algorithms/avesecho-v1/avesechov1:v1.3.0",
        "command": "--i audio --model_name 'fc' --add_csv",
    },
    "avesecho_v1.3.0_transformer": {
        "inputDir": "/app/audio",
        "outputDir": "/app/outputs",
        "image": "registry.gitlab.com/arise-biodiversity/dsi/algorithms/avesecho-v1/avesechov1:v1.3.0",
        "command": "--i audio --model_name 'passt' --add_csv",
    },
    "birdid-europe254-medium": {
        "inputDir": "/input",
        "outputDir": "/output",
        # "image": "ghcr.io/mfn-berlin/birdid-europe254-v250119-1",
        # "image": "ghcr.io/mfn-berlin/birdid-europe254-v250327-1:8870e63f5b14148207835925f2b9db160d61eda7",
        "image": "ghcr.io/mfn-berlin/birdid-europe254-v250331-1:311173148b6396f89a9d4b104cad95c064697fda",
        # "image": "birdid-europe254-v250326-1",
        # "command": "python inference.py -i /input -o /output --fileOutputFormats labels_csv --overlapInPerc 60 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize medium",
        "command": "python inference.py -i /input -o /output --fileOutputFormats labels_csv --overlapInPerc 60 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize medium --debug",
    },
    "birdid-europe254-large": {
        "inputDir": "/input",
        "outputDir": "/output",
        # "image": "ghcr.io/mfn-berlin/birdid-europe254-v250119-1",
        # "image": "ghcr.io/mfn-berlin/birdid-europe254-v250327-1:8870e63f5b14148207835925f2b9db160d61eda7",
        "image": "ghcr.io/mfn-berlin/birdid-europe254-v250331-1:311173148b6396f89a9d4b104cad95c064697fda",
        # "image": "birdid-europe254-v250326-1",
        "command": "python inference.py -i /input -o /output --fileOutputFormats labels_csv --overlapInPerc 60 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize large",
        # "command": "python inference.py -i /input -o /output --fileOutputFormats labels_csv --overlapInPerc 60 --csvDelimiter , --sortSpecies --nameType sci --includeFilePathInOutputFiles --modelSize large --debug",
    },
}


def getModelResults(
    modelID,
    outputRootDir,
    listOfFilePathsOrFolder,
    workerId=None,
    nFilesPerBatch=100,
    batchSizeFiles=16,
    nCpuWorkers=8,
    batchSize=16,
    gpuIx=0,
    minConfidenceThreshold=0.01,
    stepDuration=2.0,
    sharedMemorySizeStr="4g",
    removeContainer=True,
    containerPrefix="bmz_",
):

    if modelID not in dockerConfig:
        raise ValueError(f"Unknown modelID: {modelID}")

    # Create outputDir (subfolder in outputRootDir) for temporary results
    outputDir = outputRootDir + modelID + "/"
    os.makedirs(outputDir, exist_ok=True)

    ## Define Docker run command string depending on model

    dockerCommandFirstPart = "docker run --name " + containerPrefix + modelID

    # Add instance index to name if multiple instances of the same model are run in parallel
    if workerId is not None:
        dockerCommandFirstPart += "_" + str(workerId)
    # Add option to remove container after run
    if removeContainer:
        dockerCommandFirstPart += " --rm"

    ## Create inputMounts depending on whether listOfFilePathsOrFolder is a list of files or a folder

    # listOfFilePathsOrFolder is a folder
    if isinstance(listOfFilePathsOrFolder, str) and os.path.isdir(
        listOfFilePathsOrFolder
    ):
        nBatches = 1
        inputIsFolder = True
    else:
        nBatches = (len(listOfFilePathsOrFolder) + nFilesPerBatch - 1) // nFilesPerBatch
        inputIsFolder = False

    # Loop over batches
    for batchIx in range(nBatches):

        if inputIsFolder:
            inputMounts = (
                " -v "
                + listOfFilePathsOrFolder
                + ":"
                + dockerConfig[modelID]["inputDir"]
            )
        else:
            print("Batch:", batchIx)
            batchStartIx = batchIx * nFilesPerBatch
            batchEndIx = min(
                (batchIx + 1) * nFilesPerBatch, len(listOfFilePathsOrFolder)
            )
            batchFilePaths = listOfFilePathsOrFolder[batchStartIx:batchEndIx]

            # Construct the Docker run command with bind mounts for each file
            inputMounts = ""
            for filePath in batchFilePaths:
                fileName = os.path.basename(filePath)
                inputMounts += (
                    " -v "
                    + filePath
                    + ":"
                    + dockerConfig[modelID]["inputDir"]
                    + "/"
                    + fileName
                )  # + '/' is important here

        dockerCommand = dockerCommandFirstPart
        dockerCommand += inputMounts
        dockerCommand += " -v " + outputDir + ":" + dockerConfig[modelID]["outputDir"]

        # Add docker options
        options = "--ipc=host"
        if sharedMemorySizeStr:
            options += " --shm-size=" + sharedMemorySizeStr

        # Add GPU option
        print("DEBUG: USE_GPU raw value ->", repr(gpuIx))
        if gpuIx is None or gpuIx.strip() == "":
            # Do nothing: no GPU flag
            pass
        elif gpuIx.lower() == "all":
            options += " --gpus all"
        else:
            try:
                gpuIx_int = int(gpuIx)
                if gpuIx_int >= 0:
                    options += f" --gpus device={gpuIx_int}"
            except ValueError:
                raise RuntimeError(f"Invalid USE_GPU value: {gpuIx!r}")

        dockerCommand += " " + options

        # Add image
        dockerCommand += " " + dockerConfig[modelID]["image"]

        # Add model arguments
        command = dockerConfig[modelID]["command"]

        ## Modify command depending of passed arguments on model

        # Modify minConfidenceThreshold
        if modelID.startswith("birdnet"):
            command += " --min_conf " + str(minConfidenceThreshold)
        if modelID.startswith("birdid"):
            command += " --minConfidence " + str(minConfidenceThreshold)
        if modelID.startswith("avesecho"):
            command += " --mconf " + str(minConfidenceThreshold)

        # Modify stepDuration
        if stepDuration:
            if modelID.startswith("birdnet"):
                overlap = 3.0 - stepDuration
                overlapArg = "--overlap " + str(overlap)
                command = command.replace("--overlap 1.0", overlapArg)
            if modelID.startswith("birdid"):
                overlap = 5.0 - stepDuration
                overlapInPerc = overlap / 5.0 * 100
                overlapArg = "--overlapInPerc " + str(overlapInPerc)
                command = command.replace("--overlapInPerc 60", overlapArg)

        # Modify nCpuWorkers and batchSize
        if modelID.startswith("birdnet"):
            command += (
                " --threads " + str(nCpuWorkers) + " --batchsize " + str(batchSize)
            )
        if modelID.startswith("birdid"):
            command += (
                " --batchSizeFiles "
                + str(batchSizeFiles)
                + " --nCpuWorkers "
                + str(nCpuWorkers)
                + " --batchSizeInference "
                + str(batchSize)
            )

        dockerCommand += " " + command

        print("dockerCommand:\n", dockerCommand)
        print("Length of docker command: " + str(len(dockerCommand)))
        os.system(dockerCommand)


def postProcessResults(
    modelID,
    outputRootDir,
    fileOutputFormats=["csv"],
    removeTemporaryResultFiles=False,
    chown=None,
    outputName="output.csv",
):

    # Load labelToId mapping table
    path = rootDir + "LabelToIdMappings/" + modelID + ".csv"
    df_labelToId = pd.read_csv(path)

    # Collect results from all csv files in outputDir
    outputDir = outputRootDir + modelID + "/"

    # Check if outputDir exists before trying to list files
    if not os.path.exists(outputDir):
        print(f"Warning: Output directory does not exist: {outputDir}")
        print("No results to process (inference may have failed)")
        return

    df_list = []  # List to collect all DataFrames
    for file in os.listdir(outputDir):
        if file.endswith(".csv"):
            print("Post-processing " + file)
            # Read the csv file
            df_perFile = pd.read_csv(outputDir + file)
            df_list.append(df_perFile)

    if not df_list:
        print("No csv files found in " + outputDir)
        return

    # Concatenate all DataFrames into one
    df = pd.concat(df_list, ignore_index=True)

    ## Post process results depending on the model

    if modelID == "birdnet_v2.4" or modelID == "birdnet_v2.2":

        """
        Original format:

                    Start (s)  End (s)        Scientific name         Common name  Confidence                   File
        0         0.0      3.0  Luscinia megarhynchos  Common Nightingale      0.9317  input/LusMeg00027.mp3
        1         0.0      3.0   Atrichornis clamosus    Noisy Scrub-bird      0.0372  input/LusMeg00027.mp3
        2         0.0      3.0      Luscinia luscinia  Thrush Nightingale      0.0289  input/LusMeg00027.mp3
        3         1.0      4.0  Luscinia megarhynchos  Common Nightingale      0.9726  input/LusMeg00027.mp3
        """

        # Add col filename (col File: input/LusMeg00027.mp3 --> LusMeg00027.mp3)
        df["filename"] = df["File"].apply(lambda x: os.path.basename(x))

        ## Rename cols:
        # Start (s) --> start_time
        # End (e) --> end_time
        # Confidence --> confidence
        df.rename(
            columns={
                "Start (s)": "start_time",
                "End (s)": "end_time",
                "Confidence": "confidence",
            },
            inplace=True,
        )

        # Add col label_id by merging with df_labelToId (Scientific name is srcLabelName)
        df = pd.merge(
            df,
            df_labelToId,
            how="left",
            left_on="Scientific name",
            right_on="srcLabelName",
        )

        # Remove unnecessary cols: Scientific name, Common name, File, srcLabelIx
        df.drop(
            columns=["Scientific name", "Common name", "File", "srcLabelIx"],
            inplace=True,
        )

        # Rename cols: srcLabelName --> label_model, dstLabelId --> label_id
        df.rename(
            columns={"srcLabelName": "label_model", "dstLabelId": "label_id"},
            inplace=True,
        )

        # Add col model_id
        df["model_id"] = modelID

        # Reorder cols: model_id, filename, start_time, end_time, confidence, label_model, label_id
        df = df[
            [
                "model_id",
                "filename",
                "start_time",
                "end_time",
                "confidence",
                "label_model",
                "label_id",
            ]
        ]

    if modelID == "avesecho_v1.3.0" or modelID == "avesecho_v1.3.0_transformer":

        """
        Original format:

                Begin Time End Time             File                                         Prediction     Score
        0        0:00     0:03  LusMeg00028.mp3                       MeadowPipit_Anthus pratensis  0.010736
        1        0:00     0:03  LusMeg00028.mp3           GreatSpottedWoodpecker_Dendrocopos major  0.015183
        2        0:00     0:03  LusMeg00028.mp3              MelodiousWarbler_Hippolais polyglotta  0.011034
        """

        ## Rename cols:
        # Begin Time --> start_time
        # End Time --> end_time
        # Score --> confidence
        # File --> filename
        df.rename(
            columns={
                "Begin Time": "start_time",
                "End Time": "end_time",
                "Score": "confidence",
                "File": "filename",
            },
            inplace=True,
        )

        # Convert start_time and end_time back to seconds (0:03 --> 3.0)
        def time_str_to_seconds(time_str):
            """Converts a time string in 'M:SS' format to total seconds."""
            minutes, seconds = map(int, time_str.split(':'))
            return float(minutes * 60 + seconds)

        df['start_time'] = df['start_time'].apply(time_str_to_seconds)
        df['end_time'] = df['end_time'].apply(time_str_to_seconds)






        # Add col label_id by merging with df_labelToId (Prediction is srcLabelName)
        df = pd.merge(
            df, df_labelToId, how="left", left_on="Prediction", right_on="srcLabelName"
        )

        # Remove unnecessary cols: Scientific name, Common name, File, srcLabelIx
        df.drop(columns=["srcLabelIx"], inplace=True)

        # Rename cols: srcLabelName --> label_model, dstLabelId --> label_id
        df.rename(
            columns={"srcLabelName": "label_model", "dstLabelId": "label_id"},
            inplace=True,
        )

        # Add col model_id
        df["model_id"] = modelID

        # Reorder cols: model_id, filename, start_time, end_time, confidence, label_model, label_id
        df = df[
            [
                "model_id",
                "filename",
                "start_time",
                "end_time",
                "confidence",
                "label_model",
                "label_id",
            ]
        ]

    if modelID == "birdid-europe254-medium" or modelID == "birdid-europe254-large":

        """
        Original format:

            startTime [s]  endTime [s]                 species  confidence                filePath
        0          0.00000      5.00000   Luscinia megarhynchos     0.94881  /input/LusMeg00027.mp3
        1          0.00000      5.00000  Phylloscopus collybita     0.03471  /input/LusMeg00027.mp3
        2          0.00000      5.00000       Fringilla coelebs     0.03384  /input/LusMeg00027.mp3
        """

        # Add col filename (col filePath: /input/LusMeg00027.mp3 --> LusMeg00027.mp3)
        df["filename"] = df["filePath"].apply(lambda x: os.path.basename(x))

        ## Rename cols:
        # startTime [s] --> start_time
        # endTime [s] --> end_time
        df.rename(
            columns={"startTime [s]": "start_time", "endTime [s]": "end_time"},
            inplace=True,
        )

        # Add col label_id by merging with df_labelToId (species is srcLabelName)
        df = pd.merge(
            df, df_labelToId, how="left", left_on="species", right_on="srcLabelName"
        )

        # Remove unnecessary cols: srcLabelIx
        df.drop(columns=["srcLabelIx"], inplace=True)

        # Rename cols: srcLabelName --> label_model, dstLabelId --> label_id
        df.rename(
            columns={"srcLabelName": "label_model", "dstLabelId": "label_id"},
            inplace=True,
        )

        # Add col model_id
        df["model_id"] = modelID

        # Reorder cols: model_id, filename, start_time, end_time, confidence, label_model, label_id
        df = df[
            [
                "model_id",
                "filename",
                "start_time",
                "end_time",
                "confidence",
                "label_model",
                "label_id",
            ]
        ]

    ### Post-process DataFrame

    ## Unknown label_id hack --> to remove later !
    # If label_id is NaN or -1, set it to None
    df["label_id"] = df["label_id"].apply(
        lambda x: None if pd.isna(x) or x == -1 else x
    )

    ## Sort rows by model_id, filename, start_time, confidence
    df.sort_values(
        by=["model_id", "filename", "start_time", "confidence"],
        ascending=[True, True, True, False],
        inplace=True,
    )

    ## Reset index
    df.reset_index(drop=True, inplace=True)



    print(df)

    """
                model_id         filename  start_time  end_time  confidence            label_model  label_id
    0   birdnet_v2.4  LusMeg00027.mp3         0.0       3.0      0.9317  Luscinia megarhynchos      3307
    1   birdnet_v2.4  LusMeg00027.mp3         0.0       3.0      0.0372   Atrichornis clamosus       594
    2   birdnet_v2.4  LusMeg00027.mp3         0.0       3.0      0.0289      Luscinia luscinia      3306
    """

    ## Save outout to different file formats
    # Check if format is valid
    for fileOutputFormat in fileOutputFormats:
        if fileOutputFormat not in fileOutputFormatsValid:
            print("Warning, invalid file output format:", fileOutputFormat, flush=True)

    dstPathWithoutExt = outputRootDir + outputName

    if "csv" in fileOutputFormats:
        df.to_csv(dstPathWithoutExt + ".csv", index=False)

    if "pkl" in fileOutputFormats:
        df.to_pickle(dstPathWithoutExt + ".pkl")

    if "excel" in fileOutputFormats:
        with pd.ExcelWriter(dstPathWithoutExt + ".xlsx", engine="xlsxwriter") as writer:
            # writer.book.use_zip64()
            writer.book.strings_to_urls = False
            df.to_excel(writer, sheet_name="Sheet1", index=False)
            worksheet = writer.sheets["Sheet1"]
            (max_row, max_col) = df.shape  # Get the dimensions of the DataFrame
            column_settings = [
                {"header": column} for column in df.columns
            ]  # Define the column settings
            worksheet.add_table(
                0,
                0,
                max_row,
                max_col - 1,
                {
                    "columns": column_settings,
                    "name": "Table1",
                    "style": "Table Style Light 9",
                },
            )  # Add the table

    ## Do something with the result csv file
    # Maybe reduce confidence values to 5 decimal places, ...
    # Add results to DB, ...
    if chown:
        os.system("chown -R " + chown + " " + outputRootDir)

    ## Delete outputDir and its contents
    if removeTemporaryResultFiles:
        os.system("rm -rf " + outputDir)


################################################################################################################
################################################################################################################
################################################################################################################

if __name__ == "__main__":

    # On Windows calling this function is necessary.
    # On Linux/OSX it does nothing.
    freeze_support()

    timeStampStart = time.time()

    ## Parse arguments
    parser = argparse.ArgumentParser(
        description="Identify birds in audio files with various models."
    )

    parser.add_argument(
        "-m",
        "--modelID",
        type=str,
        metavar="",
        default=modelID,
        help="Model ID. Defaults to " + modelID,
    )
    parser.add_argument(
        "-o",
        "--outputRootDir",
        type=str,
        metavar="",
        default=outputRootDir,
        help="Output root directory. Defaults to " + outputRootDir,
    )
    parser.add_argument(
        "-ov",
        "--outputRootDirVolume",
        type=str,
        metavar="",
        help="If you use the dockerized script you have to set here the Output root directory on the host machine",
    )

    parser.add_argument(
        "-on",
        "--outputName",
        type=str,
        metavar="",
        help="Output name. Defaults to modelID.csv",
    )
    parser.add_argument(
        "-i",
        "--inputDirOrTextFilePath",
        type=str,
        metavar="",
        default=inputTextFilePath,
        help="Input directory or path of text file with list of file paths. Defaults to "
        + inputTextFilePath,
    )

    parser.add_argument(
        "-w",
        "--workerId",
        type=str,
        metavar="",
        default=workerId,
        help="Worker index. Defaults to " + str(workerId),
    )
    parser.add_argument(
        "-n",
        "--nFilesPerBatch",
        type=int,
        metavar="",
        default=nFilesPerBatch,
        help="Number of files per batch. Defaults to " + str(nFilesPerBatch),
    )
    parser.add_argument(
        "-bf",
        "--batchSizeFiles",
        type=int,
        metavar="",
        default=batchSizeFiles,
        help="Number of files per preprocessing batch (only birdid). Defaults to "
        + str(batchSizeFiles),
    )
    parser.add_argument(
        "-c",
        "--nCpuWorkers",
        type=int,
        metavar="",
        default=nCpuWorkers,
        help="Number of CPU workers. Defaults to " + str(nCpuWorkers),
    )
    parser.add_argument(
        "-b",
        "--batchSize",
        type=int,
        metavar="",
        default=batchSize,
        help="Batch size. Defaults to " + str(batchSize),
    )
    parser.add_argument(
        "-g",
        "--gpuIx",
        type=str,  # Accepts both integers as strings and 'all'
        metavar="",
        default=None,
        help="GPU index. Can be an integer or 'all'. Defaults to None",
    )
    parser.add_argument(
        "-t",
        "--minConfidenceThreshold",
        type=float,
        metavar="",
        default=minConfidenceThreshold,
        help="Minimum confidence threshold. Defaults to " + str(minConfidenceThreshold),
    )
    parser.add_argument(
        "-s",
        "--stepDuration",
        type=float,
        metavar="",
        default=2.0,
        help="Step duration in seconds. Defaults to " + str(stepDuration),
    )

    parser.add_argument(
        "--sharedMemorySizeStr",
        type=str,
        metavar="",
        default=sharedMemorySizeStr,
        help="Shared memory size. Defaults to " + sharedMemorySizeStr,
    )
    parser.add_argument(
        "--removeTemporaryResultFiles",
        action="store_true",
        help="Remove temporary result files after post processing. Defaults to "
        + str(removeTemporaryResultFiles),
    )

    parser.add_argument(
        "-f",
        "--fileOutputFormats",
        nargs="*",
        default=fileOutputFormats,
        type=str,
        metavar="",
        help="Format of output file(s). List of values in [csv, excel, pkl]. Defaults to csv.",
    )
    parser.add_argument(
        "-chown",
        "--changeOutputOwner",
        type=str,
        metavar="",
        help="Change output owner to this user:group. Defaults to None",
    )

    parser.add_argument(
        "--containerPrefix",
        type=str,
        metavar="",
        default="bmz_",
        help="Container prefix. Defaults to bmz_",
    )
    # To check and add
    # parser.add_argument('-r', '--removeContainer', action='store_true', help='Remove container after run. Defaults to True')

    args = parser.parse_args()

    ## Assign arguments to variables
    modelID = args.modelID
    outputRootDir = args.outputRootDir
    outputRootDirVolume = args.outputRootDirVolume
    inputDirOrTextFilePath = args.inputDirOrTextFilePath
    outputName = args.outputName if args.outputName else args.modelID
    workerId = args.workerId
    nFilesPerBatch = args.nFilesPerBatch
    batchSizeFiles = args.batchSizeFiles
    nCpuWorkers = args.nCpuWorkers
    batchSize = args.batchSize
    gpuIx = args.gpuIx

    chown = args.changeOutputOwner
    minConfidenceThreshold = args.minConfidenceThreshold
    stepDuration = args.stepDuration
    sharedMemorySizeStr = args.sharedMemorySizeStr
    removeTemporaryResultFiles = args.removeTemporaryResultFiles
    fileOutputFormats = args.fileOutputFormats
    containerPrefix = args.containerPrefix
    # removeContainer = args.removeContainer

    # Check if inputDirOrTextFilePath is existing file or folder
    if os.path.isfile(inputDirOrTextFilePath):
        with open(inputDirOrTextFilePath, "r") as file:
            filePaths = file.readlines()
            listOfFilePathsOrFolder = [x.strip() for x in filePaths]
            print(f"Loaded {len(listOfFilePathsOrFolder)} file paths from {inputDirOrTextFilePath}")
    else:
        listOfFilePathsOrFolder = inputDirOrTextFilePath

    ## Create temp outputRootDir
    # Append '/' to outputRootDir if not already there
    if outputRootDir[-1] != "/":
        outputRootDir += "/"
    if outputRootDirVolume and outputRootDirVolume[-1] != "/":
        outputRootDirVolume += "/"
    os.makedirs(outputRootDir, exist_ok=True)

    # Run model
    getModelResults(
        modelID,
        outputRootDirVolume if outputRootDirVolume else outputRootDir,
        listOfFilePathsOrFolder,
        workerId=workerId,
        nFilesPerBatch=nFilesPerBatch,
        batchSizeFiles=batchSizeFiles,
        nCpuWorkers=nCpuWorkers,
        batchSize=batchSize,
        gpuIx=gpuIx,
        minConfidenceThreshold=minConfidenceThreshold,
        stepDuration=stepDuration,
        sharedMemorySizeStr=sharedMemorySizeStr,
        removeContainer=True,
        containerPrefix=containerPrefix,
    )

    # Post process results
    postProcessResults(
        modelID,
        outputRootDir,
        fileOutputFormats=fileOutputFormats,
        chown=chown,
        removeTemporaryResultFiles=removeTemporaryResultFiles,
        outputName=outputName,
    )

    timeStampEnd = time.time()
    elapsed = timeStampEnd - timeStampStart
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print('ElapsedTime [hh:mm:ss.ms]: {:02}:{:02}:{:06.3f}'.format(int(hours), int(minutes), seconds))

    print("Done.")
