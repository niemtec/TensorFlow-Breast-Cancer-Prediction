import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt

# Get the training data file
train_filename = "data.csv"

# Define column keys for data.csv
idKey = "id"
diagnosisKey = "diagnosis"
radiusMeanKey = "radius_mean"
textureMeanKey = "texture_mean"
perimeterMeanKey = "perimeter_mean"
areaMeanKey = "area_mean"
smoothnessMeanKey = "smoothness_mean"
compactnessMeanKey = "compactness_mean"
concavityMeanKey = "concavity_mean"
concavePointsMeanKey = "concave points_mean"
symmetryMeanKey = "symmetry_mean"
fractalDimensionMean = "fractal_dimension_mean"
radiusSeKey = "radius_se"
textureSeKey = "texture_se"
perimeterSeKey = "perimeter_se"
areaSeKey = "area_se"
smoothnessSeKey = "smoothness_se"
compactnessSeKey = "compactness_se"
concavitySeKey = "concavity_se"
concavePointsSeKey = "concave points_se"
symmetrySeKey = "symmetry_se"
fractalDimensionSeKey = "fractal_dimension_se"
radiusWorstKey = "radius_worst"
textureWorstKey = "texture_worst"
perimeterWorstKey = "perimeter_worst"
areaWorstKey = "area_worst"
smoothnessWorstKey = "smoothness_worst"
compactnessWorstKey = "compactness_worst"
concavityWorstKey = "concavity_worst"
concavePointsWorstKey = "concave points_worst"
symmetryWorstKey = "symmetry_worst"
fractalDimensionWorstKey = "fractal_dimension_worst"

# Columns used for training
train_columns = [idKey, diagnosisKey, radiusMeanKey, textureMeanKey, perimeterMeanKey, areaMeanKey, smoothnessMeanKey,
                 compactnessMeanKey, concavityMeanKey, concavePointsMeanKey, symmetryMeanKey, fractalDimensionMean,
                 radiusSeKey, textureSeKey, perimeterSeKey, areaSeKey, smoothnessSeKey, compactnessSeKey,
                 concavitySeKey, concavePointsSeKey, symmetrySeKey, fractalDimensionSeKey, radiusWorstKey,
                 textureWorstKey, perimeterWorstKey, areaWorstKey, smoothnessWorstKey, compactnessWorstKey,
                 concavityWorstKey, concavePointsWorstKey, symmetryWorstKey, fractalDimensionWorstKey]


# Method for loading the training data
def GetTrainingData():
    dataFile = pd.read_csv(train_filename, names=train_columns, delimiter=',', skiprows=1)
    return dataFile


# Load the training data
train_data = GetTrainingData()


def GraphAreaMeanFeature():
    # Compare the mean area on benign vs malignant tumors
    print("Malignant")
    print(train_data.area_mean[train_data.diagnosis == "M"].describe())
    print()
    print("Benign")
    print(train_data.area_mean[train_data.diagnosis == "B"].describe())

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))

    bins = 50

    ax1.hist(train_data.area_mean[train_data.diagnosis == "M"], bins=bins)
    ax1.set_title('Malignant')

    ax2.hist(train_data.area_mean[train_data.diagnosis == "B"], bins=bins)
    ax2.set_title('Benign')

    plt.xlabel('Area Mean')
    plt.ylabel('Number of Diagnosis')
    plt.savefig('graphs/MalignantVsBenign-MeanAreaPlot.png')
    plt.show()


def GraphAreaWorstFeature():
    print("Malignant")
    print(train_data.area_worst[train_data.diagnosis == "M"].describe())
    print()
    print("Benign")
    print(train_data.area_worst[train_data.diagnosis == "B"].describe())

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 4))

    bins = 30

    ax1.hist(train_data.area_worst[train_data.diagnosis == "M"], bins=bins)
    ax1.set_title('Malignant')

    ax2.hist(train_data.area_worst[train_data.diagnosis == "B"], bins=bins)
    ax2.set_title('Benign')

    plt.xlabel('Area Worst')
    plt.ylabel('Number of Diagnosis')
    plt.yscale('log')
    plt.savefig('graphs/MalignantVsBenign-WorstAreaPlot.png')
    plt.show()


def GraphRestOfFeatures():
    # Select only the rest of the features.
    r_data = train_data.drop([idKey, areaMeanKey, areaWorstKey, diagnosisKey], axis=1)
    r_features = r_data.columns

    plt.figure(figsize=(12, 28 * 4))
    gs = gridspec.GridSpec(28, 1)
    for i, cn in enumerate(r_data[r_features]):
        ax = plt.subplot(gs[i])
        sns.distplot(train_data[cn][train_data.diagnosis == "M"], bins=50)
        sns.distplot(train_data[cn][train_data.diagnosis == "B"], bins=50)
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(cn))

    plt.savefig('graphs/MalignantVsBenign-RestOfFeaturesHistogram.png')
    plt.show()


# Plot the data graphs
# GraphAreaMeanFeature()
# GraphAreaWorstFeature()
# GraphRestOfFeatures()


# Update the value of diagnosis (1 = malignant and 0 = benign)
train_data.loc[train_data.diagnosis == 'M', 'diagnosis'] = 1
train_data.loc[train_data.diagnosis == 'B', 'diagnosis'] = 0

# Create a new feature for benign (non-malignant) diagnosis
train_data.loc[train_data.diagnosis == 0, 'benign'] = 1
train_data.loc[train_data.diagnosis == 1, 'benign'] = 0

# Convert benign column type to integer
train_data['benign'] = train_data.benign.astype(int)

# Rename 'Class' to 'Malignant'
train_data = train_data.rename(columns={'diagnosis': 'malignant'})

# Print result stats
# print(train_data.benign.value_counts())
# print()
# print(train_data.malignant.value_counts())


# Create dataframes for only Malignant and Benign diagnosis
Malignant = train_data[train_data.malignant == 1]
Benign = train_data[train_data.benign == 1]

# Set train_X = 80% of the malignant diagnosis
train_X = Malignant.sample(frac = 0.8)
count_Malignants = len(train_X)

# Add 805 of benign diagnosis to the train_X set
train_X = pd.concat([train_X, Benign.sample(frac = 0.8)], axis = 0)

