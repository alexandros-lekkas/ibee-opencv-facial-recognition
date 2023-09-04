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
import numpy as np

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
    return resultBoolean, predictedLabel[0], predictedLabel[1]

# Test LBPH model
def test(model, testingFaces, testingLabels):
    
    resultArray = []
    resultArray.append("")

    # Loop through testing faces and predict
    correct = 0
    for index, face in enumerate(testingFaces):
        result, label, confidence = predict(face, testingLabels[index], model)
        if result == True:
            correct += 1
        resultArray.append(label)
        resultArray.append(confidence)

    # Print results
    print("[>] Accuracy: " + str(correct) + "/" + str(len(testingFaces)))
    averageConfidence = sum(resultArray[2::2]) / len(resultArray[2::2])
    resultArray.append(averageConfidence)
    resultArray.append(correct)
    return correct, resultArray

# ===============
# Main
# ===============

# Main function
if __name__ == "__main__":
    print("This file is not meant to be run directly.")
    print("Please run main.py instead.")