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

testing.test(1, 0.5)
