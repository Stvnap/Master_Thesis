Current tasks:

Hyperparameter tuning is set to be run on the palma cluster with parameters:
(trying now to run it there)

n_neurons               3100; 3400 
n_hidden layers         2; 24
lr                      1e-4; 1e-2 log
optimizer               ["adam", "sgd"]
activation              ["leaky_relu", "sigmoid", "elu"]
drop_rate               0.1; 0.5 step=0.1
batch_size              32; 256 step=32


Further Information:

Dataset_preprocess_v2
    Creates a .csv file with cutted sequences (sliding windows) of the positive domain (domain to be classified), negative domain examples and rnd protein sequences

DNN_pipeline.py
    Trainer + Hyperparameter Tuner for the binary domain classification DNN

main.py
    Combo file that accesses both files mentioned above to run the entire pipeline