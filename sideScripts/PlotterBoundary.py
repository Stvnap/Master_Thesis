import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def opener():

    ###################################

    df_results = pd.read_csv('../ExpFiles/FINAL.csv',usecols=['Sequence_ID','Domain_Start','Domain_End'])
    df_eval = pd.read_csv('../Dataframes/v3/FoundEntriesSwissProteins_Eval.csv',usecols=['start','end','id'])

    ###############################

    pred_IDs = df_results['Sequence_ID'].tolist()
    eval_IDs = df_eval['id'].tolist()

    print(len(pred_IDs))

    # Remove entries with '_' in the ID
    pred_IDs = [id_val for id_val in pred_IDs if '_' not in id_val]

    print(len(pred_IDs))


    ###############################

    # search all pred_IDs in eval_IDs and extend df_results with the corresponding start and end from eval

    for i in range(len(pred_IDs)):
        if pred_IDs[i] in eval_IDs:
            index = eval_IDs.index(pred_IDs[i])
            start = int(df_eval['start'][index])
            end = int(df_eval['end'][index])
            df_results.at[i, 'Eval_Start'] = int(start)
            df_results.at[i, 'Eval_End'] = int(end)
    df_results['Eval_Start'] = df_results['Eval_Start'].astype('Int64')  # Nullable integer
    df_results['Eval_End'] = df_results['Eval_End'].astype('Int64')

    ###############################

    # print(df_results)
    
    return df_results

###############################

def plotter(df_results):

    colors = cm.get_cmap('tab10').colors



    df_results["Start_Diff"] = df_results["Domain_Start"] - df_results["Eval_Start"]
    df_results["End_Diff"] = df_results["Domain_End"] - df_results["Eval_End"]


    start_diffs = df_results["Start_Diff"].dropna().tolist()
    end_diffs = df_results["End_Diff"].dropna().tolist()

    # start_diffs = [int(x) for x in start_diffs if int(x) > -1000 and int(x) < 1000]
    # end_diffs = [int(x) for x in end_diffs if int(x) > -1000 and int(x) < 1000]



    plt.figure(figsize=(8, 5))
    bins = np.arange(-1025, 1026, 50)  # Create bins with middle bin centered at 0
    plt.hist(start_diffs, histtype="bar", bins=bins, alpha=0.7, label='Start Differences', color=colors[0], edgecolor='black', density=False, weights=np.ones(len(start_diffs)) / len(start_diffs))
    plt.xlabel('Error in residues')
    plt.xticks(range(-1000, 1001, 50), [str(x) if x % 200 == 0 else '' for x in range(-1000, 1001, 50)])
    plt.ylabel('Percentage')
    plt.title('Histogram of Start Position Differences')
    # plt.legend()
    # plt.grid(axis='y', alpha=0.75)
    plt.xlim(-1000, 1000)
    # plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Histogram of Start Position Differences.png', dpi=600)
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.hist(end_diffs, histtype="bar", bins=bins, alpha=0.7, label='End Differences', color=colors[1], edgecolor='black', density=False, weights=np.ones(len(end_diffs)) / len(end_diffs))
    plt.xlabel('Error in residues')
    plt.ylabel('Percentage')
    plt.title('Histogram of End Position Differences')
    # plt.legend()
    # plt.grid(axis='y', alpha=0.75)
    plt.xticks(range(-1000, 1001, 50), [str(x) if x % 200 == 0 else '' for x in range(-1000, 1001, 50)])

    plt.xlim(-1000, 1000)
    # plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Histogram of End Position Differences.png', dpi=600)

    plt.show()

################################
if __name__ == "__main__":
    df_results = opener()
    plotter(df_results)