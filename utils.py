import pickle as p
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import networkx as nx
graph_path = './data/'

def simple_classify_f1(dataset_name,write_to_file=None):
    ''' To take input planetoid classified data and perform statistical analysis'''
    results_str = []
    with open(dataset_name+"_result.pickle","rb") as f:
        x_test,y_test,test_results = p.load(f)
    g_file = open(graph_path+"trans."+dataset_name+".graph",'rb')
    graph = nx.from_dict_of_lists(p.load(g_file))

    #convert probabilities to one-hot encoding
    new_mat = np.zeros(test_results.shape)
    for i in range(test_results.shape[0]):
        index = np.argmax(test_results[i,:])
        new_mat[i,index]=1

    test_results = new_mat

    averages = ["micro", "macro", "samples", "weighted"]
    str_output = "Train ratio: Default with planetoid\n"
    for avg in averages:
        str_output += avg + ': ' + str(f1_score(test_results, y_test,average=avg)) + '\n'

    str_output += "Accuracy: " + str(accuracy_score(test_results, y_test)) + '\n'
    results_str.append(str_output)
    print(nx.info(graph))
    print("Simple classification results")
    if write_to_file:
        with open(write_to_file, 'w') as f:
            f.write(info)
            f.writelines(results_str)
    print(''.join(results_str))
    return write_to_file


def modified_classify_f1(dataset_name,write_to_file=None):
    '''this function spreads the label of neighbor to test node'''
    results_str = []
    with open(dataset_name+"_result.pickle","rb") as f:
        x_test,y_test,test_results = p.load(f)
    g_file = open(graph_path+"trans."+dataset_name+".graph",'rb')
    graph = nx.from_dict_of_lists(p.load(g_file))

    #convert probabilities to one-hot encoding
    new_mat = np.zeros(test_results.shape)
    for i in range(test_results.shape[0]):
        index = np.argmax(test_results[i,:])
        new_mat[i,index]=1

    test_results = new_mat


def classify_analysis(dataset_name):
    with open(dataset_name+"_result.pickle","rb") as f:
        x_test,y_test,test_results = p.load(f)
    g_file = open(graph_path+"trans."+dataset_name+".graph",'rb')
    graph = nx.from_dict_of_lists(p.load(g_file))

    #convert probabilities to one-hot encoding
    new_mat = np.zeros(test_results.shape)
    for i in range(test_results.shape[0]):
        index = np.argmax(test_results[i,:])
        new_mat[i,index]=1

    test_results = new_mat

    #read training set file and load the starting index of test file.
    f1 = open(graph_path+"trans."+dataset+".x")
    temp_x = pickle.load(f1)
    index = temp_x.shape[0]
    table = PrettyTable(['Serial No.','Node no.','True Label','Predicted Label','Degree','Neighbor label','Neighbor Match'])
    counter = 0
    match_counter = 0

    for i in range(index,index+y_test.shape[0]):
        if (np.array_equal(y_test[i,:],test_results[i,:]))==False:
            if graph.degree(i)==1:
                counter = counter + 1
                neighbor = list(graph.neighbors(i))

                if np.array_equal(y_test[i,:],)
