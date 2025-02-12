#  _______        _   _             
# |__   __|      | | (_)            
#    | | ___  ___| |_ _ _ __   __ _ 
#    | |/ _ \/ __| __| | '_ \ / _` |
#    | |  __/\__ \ |_| | | | | (_| |
#    |_|\___||___/\__|_|_| |_|\__, |
#                              __/ |
#                             |___/ 

# ===============
# Imports
# ===============

# Libraries
import sys
import csv
import datetime
import platform
import os

# Modules
import database
sys.path.append('./algorithms/recognition')
import eigenfaces
import LBPH

# ===============
# Functions
# ===============

# Main testing function
def test(repeats, trainingTestingRatio):

    # ===============
    # Folder set-up
    # ===============

    # Create folder for results
    currentDirectory = os.path.join(os.getcwd(), "database/results")
    resultFolder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resultFolderPath = os.path.join(currentDirectory, resultFolder + "_" + platform.system() + "_" + str(trainingTestingRatio))
    os.mkdir(resultFolderPath)

    # ===============
    # Testing
    # ===============

    # Test eigenfaces
    eigenfacesResults = []
    eigenfacesHeader = ["Eigenfaces"]
    correctTotal = 0
    print("Testing Eigenfaces (" + str(repeats), "Repeats)")
    for index in range (0, repeats):
        # ===============
        # Database
        # ===============

        # Load database for this run
        faces, labels = database.load("database/database")
        trainingFaces, trainingLabels, testingFaces, testingLabels = database.separate(faces, labels, trainingTestingRatio)

        # ===============
        # Training
        # ===============

        # Train and save models to folder
        eigenfacesModel = eigenfaces.train(trainingFaces, trainingLabels)
        
        print("Repeat", str(index + 1) + "/" + str(repeats))
        totalCorrect, resultArray, totalTimeTaken = eigenfaces.test(eigenfacesModel, testingFaces, testingLabels)
        eigenfacesResults.append(resultArray)
        correctTotal = correctTotal + totalCorrect
        averageConfidence = sum(resultArray[2::2]) / len(resultArray[2::2])
    eigenfacesTotalAccuracy = correctTotal / (len(testingFaces) * repeats) * 100
    eigenfacesAccuracyRow = ["Total accuracy %:", str(eigenfacesTotalAccuracy), "Total accuracy:", str(correctTotal) + "/" + str(len(testingFaces) * repeats), "Average confidence:", str(averageConfidence), "Total time taken:", str(totalTimeTaken), "Average time taken:", str(totalTimeTaken / (len(testingFaces) * repeats) * 100)]
    print("")

    # Test LBPH
    LBPHResults = []
    LBPHHeader = ["LBPH"]
    correctTotal = 0
    print("Testing LBPH (" + str(repeats), "Repeats)")
    for index in range (0, repeats):
        # ===============
        # Database
        # ===============

        # Load database for this run
        faces, labels = database.load("database/database")
        trainingFaces, trainingLabels, testingFaces, testingLabels = database.separate(faces, labels, trainingTestingRatio)

        LBPHModel = LBPH.train(trainingFaces, trainingLabels)

        print("Repeat", str(index + 1) + "/" + str(repeats))
        totalCorrect, resultArray, totalTimeTaken = LBPH.test(LBPHModel, testingFaces, testingLabels)
        LBPHResults.append(resultArray)
        correctTotal = correctTotal + totalCorrect
        averageConfidence = sum(resultArray[2::2]) / len(resultArray[2::2])
    LBPHTotalAccuracy = correctTotal / (len(testingFaces) * repeats) * 100
    LBPHTotalAccuracyRow = ["Total accuracy %:", str(LBPHTotalAccuracy), "Total accuracy:", str(correctTotal) + "/" + str(len(testingFaces) * repeats), "Average confidence:", str(averageConfidence), "Total time taken:", str(totalTimeTaken), "Average time taken:", str(totalTimeTaken / (len(testingFaces) * repeats) * 100)]
    print("")

    # ===============
    # Storing results
    # ===============

    # Create header for results file
    headerArray = []
    headerArray.append("")
    for label in trainingLabels:
        labelHeader = "Label: " + str(label)
        headerArray.append(labelHeader)
        headerArray.append("Confidence")
    
    # Create and open results file
    with open (resultFolderPath + "/result_" + str(trainingTestingRatio * 100) + ".csv", 'w') as csvFile:

        writer = csv.writer(csvFile) # Create a CSV writer object

        # Begin writing
        writer.writerow(headerArray)
        writer.writerow(eigenfacesHeader)
        writer.writerows(eigenfacesResults)
        writer.writerow(eigenfacesAccuracyRow)
        writer.writerow(LBPHHeader)
        writer.writerows(LBPHResults)
        writer.writerow(LBPHTotalAccuracyRow)

        # Create the "database" file in the results folder
    databasePath = os.path.join(resultFolderPath, "database.csv")
    with open(databasePath, 'w') as csvFile:

        writer = csv.writer(csvFile) # Create a CSV writer object

        # Write the training faces
        writer.writerow(trainingFaces)

        # Write the training labels
        writer.writerow(trainingLabels)

        # Write the testing faces
        writer.writerow(testingFaces)

        # Write the testing labels
        writer.writerow(testingLabels)

# Read database from file (to be able to test same DB on both platforms)
def readTestingDatabase(databasePath):

    # Open the "database" file
    with open(databasePath, 'r') as csvFile:

        # Create a CSV reader object
        reader = csv.reader(csvFile)

        # Read the training faces
        trainingFaces = []
        for row in reader:
            trainingFaces.append(row[0])

        # Read the training labels
        trainingLabels = []
        for row in reader:
            trainingLabels.append(row[1])

        # Read the testing faces
        testingFaces = []
        for row in reader:
            testingFaces.append(row[0])

        # Read the testing labels
        testingLabels = []
        for row in reader:
            testingLabels.append(row[1])

    return trainingFaces, trainingLabels, testingFaces, testingLabels
