#  _   _                            _ _              
# | \ | |                          | (_)             
# |  \| | ___  _ __ _ __ ___   __ _| |_ _______ _ __ 
# | . ` |/ _ \| '__| '_ ` _ \ / _` | | |_  / _ \ '__|
# | |\  | (_) | |  | | | | | | (_| | | |/ /  __/ |   
# |_| \_|\___/|_|  |_| |_| |_|\__,_|_|_/___\___|_|   

# ===============
# Imports
# ===============

# Libraries
import cv2
import os

# ===============
# Normalization
# ===============

# Normalize
def normalize(image):
    faceCascade = cv2.CascadeClassifier(os.path.join(os.getcwd(), "algorithms/normalization/haarcascade_frontalface_default.xml")) # Load the cascade
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    faces = faceCascade.detectMultiScale(grayImage, scaleFactor=1.1, minNeighbors=5) # Detect the faces

    # Crop the face
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        n = w
        croppedFace = grayImage[y:y + n, x:x + w] # Crop the face
        rescaledFace = cv2.resize(croppedFace, (100, 100)) # Rescale the face
        return rescaledFace # Return the face
    else: # If no face is detected
        return image # Return the original image

# Check
def check(imagePath):
    if not imagePath.endswith("_normalized.jpg"): return False # Return false if image is not normalized
    else: return True # Return true if the image is already normalized

# ===============
# Main
# ===============

# Main
if __name__ == "__main__":
    print("[>] This is a module. Run main.py instead!") # Print a message

