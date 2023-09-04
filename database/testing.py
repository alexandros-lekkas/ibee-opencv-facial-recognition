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

def test(repeats, trainingTestingRatio):

    # ===============
    # Folder set-up
    # ===============

    # Create folder for results
    currentDirectory = os.path.join(os.getcwd(), "database/results")
    resultFolder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    resultFolderPath = os.path.join(currentDirectory, resultFolder + "_" + platform.system())
    os.mkdir(resultFolderPath)

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
    eigenfacesModel.save(os.path.join(resultFolderPath, "eigenfacesModel.yml"))
    LBPHModel = LBPH.train(trainingFaces, trainingLabels)
    LBPHModel.save(os.path.join(resultFolderPath, "LBPHModel.yml"))

    # ===============
    # Testing
    # ===============

    # Test eigenfaces
    eigenfacesResults = []
    eigenfacesHeader = ["Eigenfaces"]
    correctTotal = 0
    print("Testing Eigenfaces (" + str(repeats), "Repeats)")
    for index in range (0, repeats):
        print("Repeat", str(index + 1) + "/" + str(repeats))
        totalCorrect, resultArray = eigenfaces.test(eigenfacesModel, testingFaces, testingLabels)
        eigenfacesResults.append(resultArray)
        correctTotal = correctTotal + totalCorrect
    eigenfacesTotalAccuracy = (correctTotal / len(testingFaces)) * 100
    eigenfacesAccuracyRow = ["Total accuracy %:", str(eigenfacesTotalAccuracy), "Total accuracy:", str(correctTotal) + "/" + str(len(testingFaces))]
    print("")

    # Test LBPH
    LBPHResults = []
    LBPHHeader = ["LBPH"]
    correctTotal = 0
    print("Testing LBPH (", str(repeats), "Repeats)")
    for index in range (0, repeats):
        print("Repeat", str(index + 1) + "/" + str(repeats))
        totalCorrect, resultArray = LBPH.test(LBPHModel, testingFaces, testingLabels)
        LBPHResults.append(resultArray)
        correctTotal = correctTotal + totalCorrect
    LBPHTotalAccuracy = (correctTotal / len(testingFaces)) * 100
    LBPHTotalAccuracyRow = ["Total accuracy %:", str(LBPHTotalAccuracy), "Total accuracy:", str(correctTotal) + "/" + str(len(testingFaces))]
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
    with open (resultFolderPath + "/result.csv", 'w') as csvFile:

        writer = csv.writer(csvFile) # Create a CSV writer object

        # Begin writing
        writer.writerow(headerArray)
        writer.writerow(eigenfacesHeader)
        writer.writerows(eigenfacesResults)
        writer.writerow(eigenfacesAccuracyRow)
        writer.writerow(LBPHHeader)
        writer.writerows(LBPHResults)
        writer.writerow(LBPHTotalAccuracyRow)

