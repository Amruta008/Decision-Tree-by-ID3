NAME: Amruta Desai

----------------------------Implementation of Decision Tree using ID3 algorithm-------------------------------
>> Code is tested on Iris, House-votes-84 and Tic-tac-toe dataset from http://archive.ics.uci.edu
>> The python program implements ID3 using the .name and .data file.
>> The data is split into train and test data. Training on 80% and Test on 20%
>> Single split and k-fold validation is performed to check the accuracy.
>> Discretization is done on the numerical data to form the decision tree.
>> The terminal values are also formed depending on the stop criteria.

-----------------------------Running the code-----------------------------
>> The code takes three arguments: filename.py first.names first.data
>> The .name and .data files must be present in the folder where the main file is placed.
>> Runtime command for IRIS dataset is as follow:
	python DecisionTree_ID3.py iris.names iris.data
	
>> Runtime command for House-votes-84 is as follows: [Takes a few minutes to show the entire output as the file is long]
	python DecisionTree_ID3.py house-votes-84.names house-votes-84.data
	
