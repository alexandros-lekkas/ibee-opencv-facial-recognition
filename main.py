#  __  __       _       
# |  \/  |     (_)      
# | \  / | __ _ _ _ __  
# | |\/| |/ _` | | '_ \ 
# | |  | | (_| | | | | |
# |_|  |_|\__,_|_|_| |_|

# ===============
# Imports
# ===============

# Libraries
import sys

# Modules
sys.path.append('database')
import database
sys.path.append('algorithms/recognition')
import eigenfaces
import LBPH

# ===============
# Main
# ===============

# Print "Database" title
print("""
===============
Database
===============   
""")

# Load and separate the database
faces, labels = database.load("database/database")
trainingFaces, trainingLabels, testingFaces, testingLabels = database.separate(faces, labels, 0.5)

# Print "Eigenfaces" title
print("""
===============
Eigenfaces
===============
""")

# Train and test Eigenfaces model
eigenfacesModel = eigenfaces.train(trainingFaces, trainingLabels)
eigenfaces.test(eigenfacesModel, testingFaces, testingLabels)

# Print "LBPH" title
print("""
===============
LBPH
===============
""")

# Train and test LBPH model
LBPHModel = LBPH.train(trainingFaces, trainingLabels)
LBPH.test(LBPHModel, testingFaces, testingLabels)