
import os, datetime, pickle
import numpy as np
from utils.utils import *
from models.models import *
from train_test import train, test
import wntr 
import warnings
warnings.filterwarnings('ignore')

def get_area_indices(inp_file):

    wn = wntr.network.WaterNetworkModel(inp_file)
    nodes_df = pd.DataFrame(wn.get_graph().nodes())

    inp_file_A = "networks/L-Town/Toy/L-TOWN_Area_A.inp"
    wn_A = wntr.network.WaterNetworkModel(inp_file_A)
    nodes_df_A = pd.DataFrame(wn_A.get_graph().nodes())

    node_ids = [ n[0] for n in nodes_df.values ]
    node_ids_A = [ n[0] for n in nodes_df_A.values ]
    node_ids_BC = []
    for n in node_ids:
        if n not in node_ids_A:
            node_ids_BC.append(n)

    node_ids_B = ['n226', 'n623', 'n206', 'n205', 'n208', 'n625', 'n337', 'n209', 'n211', 'n626', 
                'n210', 'n213', 'n624', 'n207', 'n212', 'n216', 'n215', 'n217', 'n246', 'n667',
                'n253', 'n254', 'n247', 'n242', 'n243', 'n236', 'n237', 'n233', 'n231', 'n234',
                'n227']

    node_ids_C = []
    for n in node_ids_BC:
        if n not in node_ids_B:
            node_ids_C.append(n)

    node_indices = np.zeros((3, len(node_ids)))
    node_idx_df = pd.DataFrame(node_indices, columns=node_ids)
    for n in node_ids_A:
        node_idx_df[n][0] = 1 
    for n in node_ids_B:
        node_idx_df[n][1] = 2 
    for n in node_ids_C:
        node_idx_df[n][2] = 3 

    indices_ABC = node_idx_df.values
    indices_A = np.where(indices_ABC[0] == 1)[0]
    indices_B = np.where(indices_ABC[1] == 2)[0]
    indices_C = np.where(indices_ABC[2] == 3)[0]
    
    return indices_A, indices_B, indices_C



""" Creating Graph for L-Town WDN from the data files """
inp_file = "networks/L-Town/Real/L-TOWN_Real.inp"
path_to_data = "networks/L-Town/Real/LeakageScenarios/Scenario-"    

model = m_GCN(in_dim = 1, 
            out_dim = 1, 
            edge_dim = 3, 
            latent_dim = 96, 
            batch_size = 16, 
            n_aggr = 45, 
            n_hops = 1, 
            num_layers = 2
            ).to(device)
installed_sensors = np.array([0, 3, 30, 53, 104, 113, 162, 187, 214, 228, \
                                287, 295, 331, 341, 409, 414, 428, 457, 468, 494, \
                                505, 515, 518, 548, 612, 635, 643, 678, 721, 725, \
                                739, 751, 768])


file_dir = os.path.dirname(os.path.realpath(__file__)) 
if not os.path.isdir(os.path.join(file_dir, "tmp")):
    os.system('mkdir ' + os.path.join(file_dir, "tmp"))
save_dir = os.path.join(file_dir, "tmp", str(datetime.date.today()))
if not os.path.isdir(save_dir):
    os.system('mkdir ' + save_dir)
out_f = open(save_dir+"/output_"+str(datetime.date.today())+".txt", "a")





class _args():
    def __init__(self, ):
        super().__init__()           
        self.model_path = "trained_models/model_L-TOWN_2880_45_1.pt"
        self.n_aggr = 45
        self.n_hops = 1
        self.batch_size =16

args = _args()
print(args.model_path)


indices_A, indices_B, indices_C = get_area_indices(inp_file)


scenarios = np.arange(1, 91)

all_results_df = pd.DataFrame(columns=list(installed_sensors) + ['A', 'B', 'C', 'y', 'dia'])

for s in scenarios:
    pressure_path = path_to_data + str(s) + "/Results/Measurements_All/Measurements_All.xlsx"
    leak_path = path_to_data + str(s) + "/Results/Leakages/Leak_0.xlsx"

    print(pressure_path, leak_path)

    wdn_graph, times = create_graph(inp_file, pressure_path, leak_path)   

    """ Normalizing pressure values using the limits used for generating the data """
    X_min, X_max = 0, 80
    wdn_graph.X, wdn_graph.edge_attr = normalize(wdn_graph.X, _min=X_min, _max=X_max), normalize(wdn_graph.edge_attr, dim=1)

    print(wdn_graph.X.shape, wdn_graph.edge_attr.shape, wdn_graph.edge_indices.shape)

    """ Creating train-val-test data based on the specified number of samples. """
    X_test = wdn_graph.X[...,0:1]

    """ Evaluating """
    Y, Y_hat, test_losses  = test(X_test, wdn_graph.edge_indices, wdn_graph.edge_attr, model, installed_sensors, args, save_dir, out_f)

    n_nodes = wdn_graph.X.shape[1]
    n_edges = wdn_graph.edge_attr.shape[1]//2

    """ Analysis """
    mean_abs_errors, abs_errors, p_coefs = plot_errors(Y[:,:,0], Y_hat[:,:,0], args, save_dir, plot=False)
    print("Mean Absolute Error and PCC: ", np.round(abs_errors.mean().item(), 6), np.round(np.mean(p_coefs), 6))
    print("Mean Absolute Error and PCC: ", np.round(abs_errors.mean().item(), 6), np.round(np.mean(p_coefs), 6), file=out_f)


    """ Anomalous Edge Nodes and Index """
    n_idx = wdn_graph.anomalies_nodes_ids

    anomalies_start_idx, anomalies_end_idx = int(wdn_graph.anomalies_df['Value'][4]), int(wdn_graph.anomalies_df['Value'][5])
    anomalies_start_ts, anomalies_end_ts = anomalies_start_idx // 3, anomalies_end_idx // 3

    print(n_idx, anomalies_start_ts, anomalies_end_ts, wdn_graph.anomalies_df)

    n_sensors = len(installed_sensors)
    n_samples = anomalies_end_ts + 1 # wdn_graph.X.shape[0]
    leak_dia = np.round(float(wdn_graph.anomalies_df["Value"][2]), 6)
    s_A, s_B, s_C = 0, 0, 0

    if s != 0:
        if n_idx[0] in indices_A:
            s_A = 1
            print('True')
        elif n_idx[0] in indices_B:
            s_B = 1
        elif n_idx[0] in indices_C:
            s_C = 1

    results = np.zeros((n_samples, n_sensors + 3 + 2))

    residuals = (Y[: n_samples, installed_sensors ,0] - Y_hat[: n_samples, installed_sensors ,0]).abs()

    results[: n_samples, : n_sensors] = residuals
    results[: , n_sensors] = s_A
    results[: , n_sensors + 1] = s_B
    results[: , n_sensors + 2] = s_C
    if s != 0:
        results[anomalies_start_ts : anomalies_end_ts + 1 , n_sensors + 3] = 1
        results[: , n_sensors + 4] = leak_dia

    print(results.shape, times.shape)

    results_df = pd.DataFrame(results, index=pd.to_datetime(times[: n_samples]), columns=list(installed_sensors) + ['A', 'B', 'C', 'y', 'dia'])
    print(results_df)

    all_results_df = pd.concat([all_results_df, results_df])

    all_results_df.to_csv("data_leaks_LTown.csv")

print(all_results_df, all_results_df.shape)
        
