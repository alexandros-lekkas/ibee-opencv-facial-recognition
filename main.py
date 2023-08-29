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
import testing

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

testing.test(trainingFaces, trainingLabels, testingFaces, testingLabels)

import csv
