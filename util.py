import os 
import numpy as np

CURRENT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = CURRENT_FOLDER+"/data/"
OUT_FOLDER = CURRENT_FOLDER+"/out/"


def read_dataset(filename, folder=DATA_FOLDER):
    # remove first column of ids
    return np.genfromtxt(folder+filename,skip_header=1,delimiter=",")[:,1:]

def train_dataset():
    return read_dataset("X_train.csv"), read_dataset("y_train.csv")

def test_dataset():
    return read_dataset("X_test.csv")

# Not tested
def write_solution(name, data):
    sol_file = open(OUT_FOLDER+name+".csv", "w")
    sol_file.write("id,y\n")

    for i, value in enumerate(data):
        sol_file.write("{},{}\n".format(i, value))
    sol_file.close()

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))


def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
