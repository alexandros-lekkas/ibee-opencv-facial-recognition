#  _      ____  _____  _    _ 
# | |    |  _ \|  __ \| |  | |
# | |    | |_) | |__) | |__| |
# | |    |  _ <|  ___/|  __  |
# | |____| |_) | |    | |  | |
# |______|____/|_|    |_|  |_|
           
# ===============
# Imports
# ===============

# Libaries
import cv2
import cv2.face
import numpy as np
import pathlib

# ===============
# Training
# ===============

# Train LBPH model
def train(faces, labels):

    print("[/] Starting model training...")
    print("")

    grayFaces = [] # Initialize list to store grayscale faces
    print("[/] CV2 re-grayscaling " + str(len(faces)) + " faces...")
    for face in faces:
        grayFace = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # Convert face to grayscale
        grayFaces.append(grayFace) # Add grayscale face to list
        print("[+] Grayscaling face " + str(len(grayFaces)) + " of " + str(len(faces)))
    print("")
    model = cv2.face.LBPHFaceRecognizer_create() # Create LBPH model
    print("[+] Model created")

    print("[>] Training model on " + str(len(grayFaces)) + " faces...")
    model.train(grayFaces, np.array(labels)) # Train LBPH model

    print("[>] Model finished training")
    print("")

    return model

# ===============
# Testing
# ===============

# Test LBPH model
def test(model, testingDictionary):

    correctIdentifications, missedIdentifications, falseIdentifcations = 0, 0, 0 # Initialize the scores

    # Loop through testing data
    for person in testingDictionary:
        standardImage = cv2.imread(person + "/Standard_normalized.jpg")
        faceVector = np.asarray([standardImage])

        # Convert person to a NumPy array
        personVector = np.asarray([person])

        # Make a prediction
        predictedPerson = model.predict(faceVector)

        # Check if the prediction is correct
        if predictedPerson == person:
            correctIdentifications += 1
        elif predictedPerson is None:
            missedIdentifications += 1
        else:
            falseIdentifcations += 1
        
    # Calculate the accuracy
    accuracy = (correctIdentifications / len(testingDictionary)) * 100

    print(accuracy)

# ===============
# Main
# ===============

# Main function
if __name__ == "__main__":
    print("This file is not meant to be run directly.")
    print("Please run main.py instead.")