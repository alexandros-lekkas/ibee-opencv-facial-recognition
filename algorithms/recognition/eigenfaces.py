#  ______ _                   __                    
# |  ____(_)                 / _|                   
# | |__   _  __ _  ___ _ __ | |_ __ _  ___ ___  ___ 
# |  __| | |/ _` |/ _ \ '_ \|  _/ _` |/ __/ _ \/ __|
# | |____| | (_| |  __/ | | | || (_| | (_|  __/\__ \
# |______|_|\__, |\___|_| |_|_| \__,_|\___\___||___/
#            __/ |                                  
#           |___/                                   

# ===============
# Imports
# ===============

# Libraries
import cv2
import cv2.face
import numpy as np
from sklearn.decomposition import PCA

# ===============
# Training
# ===============

# Train Eigenfaces model
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
    
    pca = PCA(n_components=100) # Create PCA model with 100 components
    print("[+] PCA model created")

    print("[>] Training PCA model on " + str(len(grayFaces)) + " faces...")
    pca.fit(grayFaces) # Train PCA model

    print("[>] Model finished training")
    print("")

    return pca

# ===============
# Testing
# ===============

# Predict
def predict(face, label, pca):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # Convert face to grayscale

    projectedFace = pca.transform(face.reshape(1, -1)) # Project the face onto the PCA subspace

    predictedLabel = pca.inverse_transform(projectedFace).argmax() # Predict the label of the image
    if predictedLabel == label: # If the predicted label is the same as the actual label
        result = "CORRECT"
        resultBoolean = True
    else:
        result = "INCORRECT"
        resultBoolean = False
    
    print("[>] Predicted label: " + str(predictedLabel) + " Confidence - [" + result + "]")
    return resultBoolean

# Test Eigenfaces model
def test(model, testingFaces, testingLabels):

    # Loop through testing faces and predict
    correct = 0
    for index, face in enumerate(testingFaces):
        result = predict(face, testingLabels[index], model)
        if result == True:
            correct += 1

    # Print results
    print("[>] Accuracy: " + str(correct) + "/" + str(len(testingFaces)))

# ===============
# Main
# ===============

# Main function
if __name__ == "__main__":
    print("This file is not meant to be run directly.")
    print("Please run main.py instead.")