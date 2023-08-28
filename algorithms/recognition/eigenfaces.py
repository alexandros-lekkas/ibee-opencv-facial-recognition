def recognizeFace(lbphModel, faceVector):

    # Calculate the distances to all of the people in the training dataset
    distances = []
    for person in lbphModel.get_labels():
        distance = np.linalg.norm(faceVector - lbphModel.predict(faceVector))
        distances.append(distance)

    # Find the person with the smallest distance
    closestPerson = distances.index(min(distances))

    # Return the name of the person
    return lbphModel.get_labels()[closestPerson]

# Train Eigenfaces Model
def trainEigenfacesModel(faces):
    
    faces = np.array(faces) # Convert faces to array
    
    covarianceMatrix = np.cov(faces.T) # Calculate covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix) # Calculate eigenvalues and eigenvectors

    sortedEigenvalues = eigenvalues.orgsort()[::-1] # Sort eigenvalues in descending order

    numberOfEigenvectors = 100 # Number of eigenvectors to use
    eigenvectors = eigenvectors[:, sortedEigenvalues[:numberOfEigenvectors]] # Get the first numberOfEigenvectors eigenvectors

    np.save("eigenfacesModel.npy")
    return eigenvectors