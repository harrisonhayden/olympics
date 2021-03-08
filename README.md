# Olympic Dominance for Track and Field Throughout History
# Downloading the Dataset
Go to **[https://www.kaggle.com/jayrav13/olympic-track-field-results/data](https://www.kaggle.com/jayrav13/olympic-track-field-results/data)**, scroll down, and download `results.csv`
**DO NOT OPEN THE CSV FILE IN EXCEL BEFORE USING IT IN THE PROGRAM, IT RUINS THE FORMATTING OF MANY OF THE RESULTS**

# Setup
To run the program you're going to need to install the `plotly`, and `scikit-learn` libraries using the package manager of your choice.

* You might run something like this in the command line:  
`pip install plotly`  
`pip install scikit-learn`

* Make sure you also have `pandas` installed

* Run the following modules (these are at the top of `olympics.py`):  
`import pandas as pd`  

  `from datetime import datetime`  
`import plotly.express as px`  
`from sklearn.model_selection import train_test_split`  
`from sklearn.tree import DecisionTreeRegressor`  
`from clean_file import clean`

* Make sure all the files are in the same directory
Ex: /Downloads/Genius Projects/Olympics should contain
	* `results.csv`
	* `olympics.py`
	*  `clean_file.py`
