#  _____        _        _                    
# |  __ \      | |      | |                   
# | |  | | __ _| |_ __ _| |__   __ _ ___  ___ 
# | |  | |/ _` | __/ _` | '_ \ / _` / __|/ _ \
# | |__| | (_| | || (_| | |_) | (_| \__ \  __/
# |_____/ \__,_|\__\__,_|_.__/ \__,_|___/\___|

# ===============
# Imports
# ===============

# Libraries
import cv2
import os
import random
import math
import sys

# Modules
sys.path.append('algorithms/normalization')
import normalizer
                                     
# ===============
# Images
# =============== 

# Show a preview of an image
def showPreview(image):
    cv2.imshow("", image)
    cv2.waitKey(0) # Waits for key press to close window

# ===============
# Loading
# ===============  

# Load database
def load(databasePath):

    # Define two empty lists to store detected faces and corresponding labels
    faces = []
    labels = []

    imageDirectories = os.listdir(databasePath) # Get list of directories in database

    for directoryName in imageDirectories:

        label = int(directoryName) # Set label for person by folder name
        
        # Set path of the each folder of training
        imageDirectory = databasePath + "/" + directoryName
        imagePaths = os.listdir(imageDirectory) # Get list of images in folder

        for image in imagePaths:

            imagePath = imageDirectory + "/" + image # Set path of each image
            image = cv2.imread(imagePath) # Read image
            isNormalized = normalizer.check(imagePath) # Check if image is normalized

            if (not isNormalized): # If image is not normalized
                
                print("[/] Normalizing: " + imagePath)
                image = normalizer.normalize(image) # Normalize image
                showPreview(image) # Show preview of normalized image

                [print("[>] Skipping: " + imagePath)]

            faces.append(image)
            labels.append(label)

            print("[+] Added: " + imagePath + " (Label: " + str(label) + ")") # Print message

    print("")
    print("[>] Total faces:", len(faces))
    print("[>] Total labels:", len(labels))

    return (faces, labels)

# ===============
# Organizing
# ===============
 
# Seperates faces into faces for training and testing
def separate(faces, labels, ratio):

    # ===============
    # Dictionary
    # ===============

    print("[+] Created face dictionary")
    faceDictionary = {} # Initialize dictionary to store faces

    for index in range(len(faces)): # Loop through faces
        if labels[index] not in faceDictionary:
            faceDictionary[labels[index]] = []
        faceDictionary[labels[index]].append(faces[index]) # Add face to dictionary
        print("[+] Added face " + str(index) + " to dictionary with label " + str(labels[index]))
    
    print("[+] Finalised face dictionary")
    print("")

    # ===============
    # Separating
    # ===============

    trainingFaceDictionary = {} # Initialize dictionary to store training data
    testingFaceDictionary = {} # Initialize dictionary to store testing data

    for label in faceDictionary:

        print("[/] Separating images with label " + str(label) + " into training and testing data.")
        
        # Initialize dictionaries for label
        if label not in trainingFaceDictionary:
            trainingFaceDictionary[label] = []
            print("[+] Created training dictionary for label " + str(label))
        if label not in testingFaceDictionary:
            testingFaceDictionary[label] = []
            print("[+] Created testing dictionary for label " + str(label))

        faces = faceDictionary[label] # Get faces with label
        random.shuffle(faces) # Shuffle faces
        numberOfTrainingFaces = math.ceil(len(faces) * ratio) # Get number of training faces
        print("[/] Separating " + str(len(faces)) + " faces into " + str(numberOfTrainingFaces) + " training faces and " + str(len(faces) - numberOfTrainingFaces) + " testing faces.")

        # Add selected faces to training dictionary
        for index in range(numberOfTrainingFaces):
            trainingFaceDictionary[label].append(faces[index])
            print("[+] Added face " + str(index) + " to training dictionary with label " + str(label))
        
        # Add remaining images to testing dictionary
        for index in range(numberOfTrainingFaces, len(faces)):
            testingFaceDictionary[label].append(faces[index])
            print("[+] Added face " + str(index) + " to testing dictionary with label " + str(label))

        print("")
        print("[>] Training dictionary for label " + str(label) + " has " + str(len(trainingFaceDictionary[label])) + " faces.")
        print("[>] Testing dictionary for label " + str(label) + " has " + str(len(testingFaceDictionary[label])) + " faces.")
        print("")

    # ===============
    # Re-arraying
    # ===============

    print("[>] Re-arraying training data")

    # Initialize lists to store training and testing data
    trainingFaces, trainingLabels = [], []
    testingFaces, testingLabels = [], []

    # Add training data to lists
    for label in trainingFaceDictionary:
        for face in trainingFaceDictionary[label]:
            trainingFaces.append(face)
            trainingLabels.append(label)
            print("[+] Added face to training data with label " + str(label))

    # Add testing data to lists
    for label in testingFaceDictionary:
        for face in testingFaceDictionary[label]:
            testingFaces.append(face)
            testingLabels.append(label)
            print("[+] Added face to testing data with label " + str(label))

    print("[>] Re-arraying complete")

    return trainingFaces, trainingLabels, testingFaces, testingLabels

# ===============
# Main
# ===============

# Main function
if __name__ == "__main__":
    print("[>] This is a module and cannot be run as a script.")