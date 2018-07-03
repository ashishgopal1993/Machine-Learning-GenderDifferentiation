# Machine-Learning-GenderDifferentiation
Machine learning model to differentiate gender based on MFCC featues of speech signal.
Contains two python files:
1. Training file (GenderRecognition_main.py) 
2. Testing file (GenderRecognition_test.py)

1. Training file:
	Variable 'source' contains path to read training data from.
	Variable 'destination' contains path to store trained data.
	Features extracted are MFCC(Mel Frequency Cepstrum Coeficients) points.
	Gaussian Mixture Model is used for modelling based on MFCC points.

2. Testing file:
	Variable 'sourcepath' holds the local address for testing data.
	Variable 'modelpath' holds the location of trained models.
	This code generates maximum likelihood.
