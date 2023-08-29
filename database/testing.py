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
import time

# Modules
import database
sys.path.append('./algorithms/recognition')
import eigenfaces
import LBPH

# ===============
# Functions
# ===============

def test(trainingFaces, trainingLabels, testingFaces, testingLabels, testNumber):

    # Create header for results file
    headerArray = []
    headerArray.append("")
    for label in trainingLabels:
        headerArray.append("Label", label)
        headerArray.append("Confidence")

    # Create and open the results file
    fileName = time.time() + "-results.csv"
    with open (fileName, 'w') as csvFile:

        writer = csv.writer(csvFile) # Create a CSV writer object

        writer.writerow(headerArray)

        for index in range (0, testNumber):

            # Train and test Eigenfaces model
            eigenfacesModel = eigenfaces.train(trainingFaces, trainingLabels)
            correct, resultArray = eigenfaces.test(eigenfacesModel, testingFaces, testingLabels)

            # Train and test LBPH model
            LBPHModel = LBPH.train(trainingFaces, trainingLabels)
            correct, resultArray = LBPH.test(LBPHModel, testingFaces, testingLabels)

            # Write results to CSV file
            writer.writerow(resultArray)

