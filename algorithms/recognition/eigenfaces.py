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

# ===============
# Training
# ===============

# Train Eigenfaces model
def train(faces, labels):

    model = cv2.face.EigenFaceRecognizer_create() # Create Eigenfaces model
    model.train(faces, np.array(labels)) # Train Eigenfaces model

    return model

# ===============
# Testing
# ===============

# Predict
def predict(face, label, model):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # Convert face to grayscale

    predictedLabel = model.predict(face) # Predict the label of the image

    if predictedLabel[0] == label: # If the predicted label is the same as the actual label
        result = "CORRECT"
        resultBoolean = True
    else:
        result = "INCORRECT"
        resultBoolean = False
    
    print("[>] Predicted label: " + str(predictedLabel[0]) + " - " + str(predictedLabel[1]) + " Confidence - [" + result + "]")
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