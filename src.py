import csv
import string
from sklearn.cross_validation import train_test_split


def load_stopWords(filepath):
    stopWords = []
    with open(filepath, 'rb') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            tempWord = row[0]
            stopWords.append(tempWord)

    return stopWords


def load_data(filePath, stopWords=None):
    # reader object iterates over lines of text file.  Each row is returned as list of strings.

    with open(filePath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        rawList = [row for row in reader]

    endList, classVector = [], []

    for line in rawList:
        tempList = []
        for word in line:
            if line.index(word) == 0:
                classVector.append(word)
            # For each word in each line, remove the punctuation and make lowercase
            # Append every word to a tempList.  Append tempList to final list once whole song processed
            else:
                noPunct = word.translate(string.maketrans("",""), string.punctuation).lower()
                noPunct = noPunct.translate(None, string.digits)
                if (len(noPunct) > 1) and not(noPunct in stopWords):
                    tempList.append(noPunct)
        endList.append(tempList)

    classVector = map(int, classVector[:-1])
    dataTrain, dataTest, classTrain, classTest = train_test_split(endList, classVector, test_size=0.3, random_state=42)

    return dataTrain, dataTest, classTrain, classTest


def create_vocabulary(endList):
    vocabList = set()

    for item in endList:
        vocabList = vocabList | set(item)

    return list(vocabList)


def create_feature_vectors(endList, vocabList):
    wordCnt = len(vocabList)
    docCnt = len(endList)

    # Create list of lists. All entries zero.  Each sublist is a feature vector.
    featureMatrix = [[0 for col in range(wordCnt)] for row in range(docCnt)]

    for i, doc in enumerate(endList):
        for word in doc:
            if word in vocabList:
                featureMatrix[i][vocabList.index(word)] += 1

    return featureMatrix


def compute_NB_probability(featureMatrix, classVector, vocabList):
    # Prior probability
    classBProb = sum(classVector) / float(len(classVector))

    # Feature probability
    wordCnt = len(featureMatrix[0])
    docCnt = len(featureMatrix)
    featureProb = [0 for word in range(wordCnt)]

    # Class conditional probability
    classAFeatureN = [1 for word in range(wordCnt)]
    classBFeatureN = [1 for word in range(wordCnt)]
    classAFeatureD, classBFeatureD = 2.0, 2.0

    for j in range(wordCnt):
        for i in range(docCnt):
            featureProb[j] += featureMatrix[i][j]

            if classVector[i] == 0:
                classAFeatureN[j] += featureMatrix[i][j]
                classAFeatureD += 1.0

            else:
                classBFeatureN[j] += featureMatrix[i][j]
                classBFeatureD += 1.0

    featureProb = [float(x) / sum(featureProb) for x in featureProb]
    classAFeatProb = [float(x) / classAFeatureD for x in classAFeatureN]
    classBFeatProb = [float(x) / classBFeatureD for x in classBFeatureN]

    return classBProb, featureProb, classAFeatProb, classBFeatProb


def classify_NB(testData, classBProb, featureProb, classAFeatProb, classBFeatProb):
    classBPostProb = [x * classBProb for x in classBFeatProb]
    classBPostProb = [x / y for x, y in zip(classBPostProb, featureProb)]

    classAPostProb = [x * (1-classBProb) for x in classAFeatProb]
    classAPostProb = [x / y for x, y in zip(classAPostProb, featureProb)]

    classifyTestData = []
    for item in testData:
        thisItemBProb = [x * y for x, y in zip(item, classBPostProb)]
        thisItemAProb = [x * y for x, y in zip(item, classAPostProb)]

        productB, productA = 1.0, 1.0

        for probB in thisItemBProb:
            if probB > 0:
                productB *= probB

        for probA in thisItemAProb:
            if probA > 0:
                productA *= probA

        if productB > productA:
            classifyTestData.append(1)
        else:
            classifyTestData.append(0)

    return classifyTestData


def compute_accuracy(classifyTestData, classVector):
    correct, total = 0.0, 0.0

    for x, y in zip(classifyTestData, classVector):
        total += 1.0
        if x == y:
            correct += 1.0

    score = float(correct / total)
    print "Naive Bayes classifier accuracy: %i for %i tests is %0.2f" % (correct, total, score)


# Load stop words list
stopWordsPath = 'NaiveBayes/StopWords.txt'
stopWordsList = load_stopWords(stopWordsPath)

# Load and separate data into training and testing sets
dataPath = 'NaiveBayes/CourseData.txt'
trainingData, testingData, trainingClass, testingClass = load_data(dataPath, stopWordsList)

# Create a set that contains each unique word from all courses
trainingVocab = create_vocabulary(trainingData)

# Create matrix from training data of counts of the tokens used in each line
trainingFeatureVects = create_feature_vectors(trainingData, trainingVocab)

# Calculate all probabilities for Bayes Rule: p(Ci|w) = P(w|Ci) * P(Ci) / P(w)
probClassB, probWord, probWordGivenA, probWordGivenB = compute_NB_probability(trainingFeatureVects, trainingClass, trainingVocab)

# Create matrix from testing data of counts of the tokens used in each line
testingFeatureVects = create_feature_vectors(testingData, trainingVocab)

# Classify testing docs into classes
predictedClass = classify_NB(testingFeatureVects, probClassB, probWord, probWordGivenA, probWordGivenB)

# Calculate and display accuracy
compute_accuracy(predictedClass, testingClass)
