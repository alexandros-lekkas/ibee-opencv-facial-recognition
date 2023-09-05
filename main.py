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
import testing

# ===============
# Main
# ===============

testing.test(10, 0.05)
testing.test(10, 0.1)
testing.test(10, 0.15)
testing.test(10, 0.2)
testing.test(10, 0.25)
testing.test(10, 0.3)
testing.test(10, 0.35)
testing.test(10, 0.4)
testing.test(10, 0.45)
testing.test(10, 0.5)
testing.test(10, 0.55)
testing.test(10, 0.6)
testing.test(10, 0.65)
testing.test(10, 0.7)
testing.test(10, 0.75)
testing.test(10, 0.8)
testing.test(10, 0.85)
testing.test(10, 0.9)
testing.test(10, 0.95)