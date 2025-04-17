Hyperparameter tuning is set to be run on the palma cluster with parameters:
(trying now to run it there)

batch_size              512
n_neurons               3100; 3400 
n_hidden layers         2; 24
lr                      1e-4; 1e-2 log
optimizer               ["adam", "sgd"]
activation              ["leaky_relu", "sigmoid", "elu"]
drop_rate               0.1; 0.5 step=0.1



Further Information:

Dataset_preprocess_v2
    Creates a .csv file with cutted sequences (sliding windows) of the positive domain (domain to be classified), negative domain examples and rnd protein sequences

Dataset_preprocess_EVAL_v2.py
    Creates a .csv file for evaluation purpuses. It uses whole protein sequences, cutts them (sliding windows) and labels them positivly if more than 50 % of the positive domain is found in the cutted 
    sequence

DNN_HP_Search.py
    Trainer + Hyperparameter Tuner for the binary domain classification DNN

Testrunner.py
    Loads in a trial.json and checkpoints.weights.h5 previously produced during HP Tuning. Uses this model structure and correpsonding weights to train the model to optimal performance.

Predicter.py
    Used to predict a subset of sequences created with Dataset_preprocess_EVAL_v2.py for performance purpuses.

main.py
    Combo file that accesses all files mentioned above to run the entire pipeline (except)