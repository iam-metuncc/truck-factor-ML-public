# Truck Factor Machine Learning Project

This project uses a combination of machine learning models to estimate the truck factor of multiple repositories and display performance metrics.

## Dataset

The repository contains two folders: "dataset" and "model". The "dataset" folder contains two csv files which include features about developers in 34 repositories and labels on whether they are part of the truck factor or not. One file has the features normalized using Z-score and the other using Minmax. The names of the developers were anonymized.

## Model

The model folder contains the main.py file which contains the code that creates the models and runs the algorithm. The dataset files are also in this folder for ease of access. The model that is evaluated in this code is the best performing model in the Truck Factor - ML study and it is a combination of Random Forest and Naive Bayes.

## Usage

Simply running the python code with the dataset files in the same folder as the code will run the model and display the performance metrics.

## References

The labels in the dataset were identified as part of the work conducted in the following papers:

G. Avelino, L. Passos, A. Hora, M. T. Valente, A novel approach for estimating truck factors, in: 2016 IEEE 24th International Conference on Program Comprehension (ICPC), IEEE, 2016. doi:10.1109/icpc.2016.7503718.

M. Ferreira, T. Mombach, M. Valente, K. Ferreira, Algorithms for estimating truck factors: a comparative study, Software Quality Journal 27 (2019). doi:10.1007/s11219-019-09457-2

