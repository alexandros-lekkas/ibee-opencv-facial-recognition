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
import LBPH

# ===============
# Main
# ===============

# Print title
print("""
===============
Database
===============   
""")

# Load and separate the database
faces, labels = database.load("database/database")
trainingFaces, trainingLabels, testingFaces, testingLabels = database.separate(faces, labels, 0.8)

# Print title
print("""
===============
LBPH
===============
""")

# Train and test LBPH model
LBPHModel = LBPH.train(trainingFaces, trainingLabels)
#LBPH.test(LBPHModel, testingDictionary)



