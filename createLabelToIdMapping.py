import os
import numpy as np
import pandas as pd

# Create labelToId mapping table for each model


# Get root directory of script
rootDir = os.path.dirname(os.path.abspath(__file__)) + '/'


def getLabelsBirdNetV24():

    # BirdNet 2.4
    #path = rootDir + 'Models/BirdNET/BirdNET-Analyzer/birdnet_analyzer/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt'
    path = rootDir + 'LabelToIdMappings/ModelLabelFiles/BirdNET_GLOBAL_6K_V2.4_Labels.txt'

    '''
    Abroscopus albogularis_Rufous-faced Warbler
    Abroscopus schisticeps_Black-faced Warbler
    Abroscopus superciliaris_Yellow-bellied Warbler
    Aburria aburri_Wattled Guan
    Acanthagenys rufogularis_Spiny-cheeked Honeyeater
    Acanthidops bairdi_Peg-billed Finch
    Acanthis cabaret_Lesser Redpoll
    Acanthis flammea_Common Redpoll
    Acanthis hornemanni_Hoary Redpoll
    '''

    # Read the file
    labels = np.loadtxt(path, dtype=str, delimiter='\t').tolist()

    # Split each label and only use first part (scientific name) before _ (Abroscopus albogularis_Rufous-faced Warbler --> Abroscopus albogularis)
    labels = [label.split('_')[0] for label in labels]
    #print(labels)

    return labels


def getLabelsBirdNetV22():

    # BirdNet 2.2
    #path = rootDir + 'Models/BirdNET/BirdNET-Analyzer/birdnet_analyzer/checkpoints/V2.2/BirdNET_GLOBAL_3K_V2.2_Labels.txt'
    path = rootDir + 'LabelToIdMappings/ModelLabelFiles/BirdNET_GLOBAL_3K_V2.2_Labels.txt'

    '''
    Abroscopus albogularis_Rufous-faced Warbler
    Abroscopus superciliaris_Yellow-bellied Warbler
    Aburria aburri_Wattled Guan
    Acanthagenys rufogularis_Spiny-cheeked Honeyeater
    Acanthis cabaret_Lesser Redpoll
    Acanthis flammea_Common Redpoll
    Acanthis hornemanni_Hoary Redpoll
    Acanthiza chrysorrhoa_Yellow-rumped Thornbill
    Acanthiza ewingii_Tasmanian Thornbill
    '''

    # Read the file
    labels = np.loadtxt(path, dtype=str, delimiter='\t').tolist()

    # Split each label and only use first part (scientific name) before _ (Abroscopus albogularis_Rufous-faced Warbler --> Abroscopus albogularis)
    labels = [label.split('_')[0] for label in labels]
    #print(labels)

    return labels


def getLabelsAvesEcho130():

    # Load labels
    #path = rootDir + 'Models/AvesEchoV1/avesecho-v1/inputs/list_AvesEcho.csv'
    path = rootDir + 'LabelToIdMappings/ModelLabelFiles/list_AvesEcho.csv'
    df_labels = pd.read_csv(path, header=None)
    #print(df_labels)

    '''
                             0                     1
    0        Gypaetus barbatus        BeardedVulture
    1          Anas bahamensis  White-cheekedPintail
    2    Pelecanus onocrotalus     GreatWhitePelican
    3     Tetraogallus caspius       CaspianSnowcock
    4        Puffinus yelkouan    YelkouanShearwater
    '''

    # Get sciNames (scientific names of col 0)
    sciNames = df_labels[0].tolist()

    # Get comNames (common names of col 1)
    comNames = df_labels[1].tolist()

    # Output labels are comNames '_' sciNames
    labels = [comNames[i] + '_' + sciNames[i] for i in range(len(sciNames))]
    #print(labels)

    return labels


def getLabelsBirdIDEurope254():

    # Load labels
    #path = rootDir + 'Models/BirdID-Europe254/BirdID-Europe254/species.csv'
    path = rootDir + 'LabelToIdMappings/ModelLabelFiles/species.csv'
    
    df_labels = pd.read_csv(path, sep=';')
    #print(df_labels)

    '''
        ix       id                     sci                    de                  en  minConfidence
    0      0  ParMaj0             Parus major             Kohlmeise           Great Tit            NaN
    1      1  FriCoe0       Fringilla coelebs              Buchfink    Common Chaffinch            NaN
    2      2  LusMeg0   Luscinia megarhynchos            Nachtigall  Common Nightingale            NaN
    3      3  TurMer0           Turdus merula                 Amsel    Common Blackbird            NaN
    '''

    # Get sciNames as labels (column sci in df_labels)
    labels = df_labels['sci'].tolist()
    #print(labels)

    return labels


def createLabelToIdMapping(labels, modelID):

    # Load label name to ID mapping csv
    path = rootDir + 'LabelNameToIdMapping_v04.csv'
    df_map = pd.read_csv(path, sep=',')

    # Cast label_id to int
    #df_map['label_id'] = df_map['label_id'].astype(int)

    #print(df_map)

    ## Create dataframe with cols srcLabelIx, srcLabelName, dstLabelId
    df = pd.DataFrame({'srcLabelIx': range(len(labels)), 'srcLabelName': labels})

    #print(df[495:506])

    # Get sciNames to match label ids if modelID starts with avesecho
    if modelID.startswith('avesecho'):
        df['sciNames'] = [label.split('_')[1] for label in labels]
        df = pd.merge(df, df_map, how='left', left_on='sciNames', right_on='name')
        df.drop(columns=['sciNames'], inplace=True)
    else:
        # Merge df with df_map to get dstLabelId
        df = pd.merge(df, df_map, how='left', left_on='srcLabelName', right_on='name')


    
    #print(df)
    # Print df rows 495-505
    #print(df[495:506])
    #print(labels[495:506])
    
    # print(len(labels))
    # print(len(df))

    assert len(labels) == len(df)


    # Rename label_id to dstLabelId
    df.rename(columns={'label_id': 'dstLabelId'}, inplace=True)

    # Drop unnecessary columns: label_type,name,name_type,name_source
    df.drop(columns=['label_type', 'name', 'name_type', 'name_source'], inplace=True)

    # Check for missing dstLabelId
    missing = df[df['dstLabelId'].isnull()]
    nLabelsWithoutId = len(missing)
    # Print missing labels (list of srcLabelName values)
    if nLabelsWithoutId > 0:
        print(missing['srcLabelName'].tolist())
    print('Number of labels without dstLabelId: ', nLabelsWithoutId)



    # Check for duplicate dstLabelId
    nDuplicates = df.duplicated(subset=['dstLabelId']).sum()
    print('Number of duplicate dstLabelId: ', nDuplicates)
    # Print labels (list of srcLabelName values) with duplicate dstLabelId
    if nDuplicates > 0:
        df_duplicates = df[df.duplicated(subset=['dstLabelId'], keep=False)]
        # Sort by dstLabelId
        df_duplicates.sort_values(by='dstLabelId', inplace=False)
        print(df_duplicates)


    # Replace NaN with -1 and cast to int
    df['dstLabelId'] = df['dstLabelId'].fillna(-1).astype(int)

    #print(df)

    # Save to csv
    dstDir = rootDir + 'LabelToIdMappings/'
    dstFileName = modelID + '.csv'
    df.to_csv(dstDir + dstFileName, index=False)


####################################################################################################
####################################################################################################
####################################################################################################


labels = getLabelsBirdNetV24()
createLabelToIdMapping(labels, 'birdnet_v2.4')

labels = getLabelsBirdNetV22()
createLabelToIdMapping(labels, 'birdnet_v2.2')

labels = getLabelsAvesEcho130()
createLabelToIdMapping(labels, 'avesecho_v1.3.0')
createLabelToIdMapping(labels, 'avesecho_v1.3.0_transformer')

labels = getLabelsBirdIDEurope254()
createLabelToIdMapping(labels, 'birdid-europe254-medium')
createLabelToIdMapping(labels, 'birdid-europe254-large')

print('Done.')