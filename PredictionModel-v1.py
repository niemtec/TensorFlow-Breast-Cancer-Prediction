import tensorflow as tf
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt


# Define methods for printing data into graphs
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

# Plot the data graphs
# GraphAreaMeanFeature()
# GraphAreaWorstFeature()
# GraphRestOfFeatures()

train_data.head()
train_data.describe()
train_data.isnull().sum()

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

pd.set_option("display.max_columns", 101)
train_data.head()

# Print result stats
# print(train_data.benign.value_counts())
# print()
# print(train_data.malignant.value_counts())


# Create dataframes for only Malignant and Benign diagnosis
Malignant = train_data[train_data.malignant == 1]
Benign = train_data[train_data.benign == 1]

# Set train_X = 80% of the malignant diagnosis
train_X = Malignant.sample(frac=0.8)
count_Malignants = len(train_X)

# Add 805 of benign diagnosis to the train_X set
train_X = pd.concat([train_X, Benign.sample(frac=0.8)], axis=0)

# Text_X dataset should contain all the diagnostics not present in train_X
test_X = train_data.loc[~train_data.index.isin(train_X.index)]

# Shuffle the data frames for training to be done in random order
train_X = shuffle(train_X)
test_X = shuffle(test_X)

# Add target features to train_Y and test_Y
train_Y = train_X.malignant
train_Y = pd.concat([train_Y, train_X.benign], axis=1)

test_Y = test_X.malignant
test_Y = pd.concat([test_Y, test_X.benign], axis=1)

# Drop target features from train_X and test_X
train_X = train_X.drop(['malignant', 'benign'], axis=1)
test_X = test_X.drop(['malignant', 'benign'], axis=1)

# Check if all training/testing dataframes are of the right length
# print(len(train_X))
# print(len(train_Y))
# print(len(test_X))
# print(len(test_Y))


# Names of all the features in train_X
features = train_X.columns.values

# Transform each feature in features so that it has a mean of 0 and s.d. of 1. (Helps training the softmax algorithm)
for feature in features:
    mean, std = train_data[feature].mean(), train_data[feature].std()
    train_X.loc[:, feature] = (train_X[feature] - mean) / std
    test_X.loc[:, feature] = (test_X[feature] - mean) / std

# Training the Neural Network

# Neural Network Parameters
learning_rate = 0.005
training_dropout = 0.9
display_step = 1
training_epochs = 5
batch_size = 100
accuracy_history = []
cost_history = []
valid_accuracy_history = []
valid_cost_history = []

# Number of input nodes
input_nodes = train_X.shape[1]

# Number of labels (malignant and benign)
num_labels = 2

# Split the testing data into validation and testing sets
split = int(len(test_Y) / 2)

train_size = train_X.shape[0]
n_samples = train_Y.shape[0]

input_X = train_X.as_matrix()
input_Y = train_Y.as_matrix()
input_X_valid = test_X.as_matrix()[:split]
input_Y_valid = test_Y.as_matrix()[:split]
input_X_test = test_X.as_matrix()[split:]
input_Y_test = test_Y.as_matrix()[split:]


def CalculateHiddenNodes(nodes):
    return (((2 * nodes) / 3) + num_labels)


# Number of nodes in each hidden layer
hidden_nodes1 = round(CalculateHiddenNodes(input_nodes))
hidden_nodes2 = round(CalculateHiddenNodes(hidden_nodes1))
hidden_nodes3 = round(CalculateHiddenNodes(hidden_nodes2))

print(input_nodes, hidden_nodes1, hidden_nodes2, hidden_nodes3)

# Percent of nodes to keep during dropout
pkeep = tf.placeholder(tf.float32)

# Input
x = tf.placeholder(tf.float32, [None, input_nodes])

# Layer 1
W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_nodes1], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_nodes1]))
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# Layer 2
W2 = tf.Variable(tf.truncated_normal([hidden_nodes1, hidden_nodes2], stddev=0.1))
b2 = tf.Variable(tf.zeros([hidden_nodes2]))
y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

# Layer 3
W3 = tf.Variable(tf.truncated_normal([hidden_nodes2, hidden_nodes3], stddev=0.1))
b3 = tf.Variable(tf.zeros([hidden_nodes3]))
y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)
y3 = tf.nn.dropout(y3, pkeep)

# Layer 4
W4 = tf.Variable(tf.truncated_normal([hidden_nodes3, 2], stddev=0.1))
b4 = tf.Variable(tf.zeros([2]))
y4 = tf.nn.softmax(tf.matmul(y3, W4) + b4)

# Output
y = y4
y_ = tf.placeholder(tf.float32, [None, num_labels])

# Minimise error using cross entropy
cost = -tf.reduce_sum(y_ * tf.log(y))

# Adam Optimiser
optimiser = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Test Model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialise variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        for batch in range(int(n_samples / batch_size)):
            batch_x = input_X[batch * batch_size: (1 + batch) * batch_size]
            batch_y = input_Y[batch * batch_size: (1 + batch) * batch_size]

            sess.run([optimiser], feed_dict={
                x: batch_x,
                y_: batch_y,
                pkeep: training_dropout
            })

        # Display logs after every 10 epochs
        if (epoch) % display_step == 0:
            train_accuracy, newCost = sess.run([accuracy, cost],
                                               feed_dict={x: input_X, y_: input_Y,
                                                          pkeep: training_dropout})

            valid_accuracy, valid_newCost = sess.run([accuracy, cost],
                                                     feed_dict={x: input_X_valid,
                                                                y_: input_Y_valid, pkeep: 1})

            print("Epoch: ", epoch, "  Accuracy: ", "{:.5f}".format(train_accuracy),
                  "  Cost: ", "{:.5f}".format(newCost),
                  "  Valid Accuracy: ", "{:.5f}".format(valid_accuracy),
                  "  Valid Cost: ", "{:.5f}".format(valid_newCost))

            # Record the results of the results of the model
            accuracy_history.append(train_accuracy)
            cost_history.append(newCost)
            valid_accuracy_history.append(valid_accuracy)
            valid_cost_history.append(valid_newCost)

            # If the model does n ot improve after 15 logs, stop the training
            if valid_accuracy < max(valid_accuracy_history) and epoch > 100:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0

    print("Run Complete Finished.")

    # Plot the accuracy and cost summaries
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

    ax1.plot(accuracy_history, color='b')  # blue
    ax1.plot(valid_accuracy_history, color='g')  # green
    ax1.set_title('Accuracy')

    ax2.plot(cost_history, color='b')
    ax2.plot(valid_cost_history, color='g')
    ax2.set_title('Cost')

    plt.xlabel('Epochs (x10)')
    plt.savefig('graphs/FinalAccuracyAndCostSummary.png')
    plt.show()