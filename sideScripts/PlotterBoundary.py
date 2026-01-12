import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def opener():

    ###################################

    df_results = pd.read_csv('../ExpFiles/testset/TestSet_1.csv',usecols=['Sequence_ID','Domain_Start','Domain_End',"Prediction"])
    # df_results2 = pd.read_csv('../ExpFiles/testset/TestSet_2.csv',usecols=['Sequence_ID','Domain_Start','Domain_End',"Prediction"])
    df_eval = pd.read_csv('../Dataframes/v3/SampledEntriesCompleteProteins_MAX.csv',usecols=['start','end','id',"Pfam_id"])

    ###############################

    # Remove entries with '_' in the ID from DataFrame first
    df_results = df_results[~df_results['Sequence_ID'].str.contains('_', na=False)].reset_index(drop=True)
    # df_results2 = df_results2[~df_results2['Sequence_ID'].str.contains('_', na=False)].reset_index(drop=True)

    # Combine both DataFrames
    # df_results = pd.concat([df_results, df_results2], ignore_index=True)

    pred_IDs = df_results['Sequence_ID'].tolist()
    eval_IDs = df_eval['id'].tolist()

    print(len(pred_IDs))

    # Count matching IDs
    match_count = 0
    prediction_match_count = 0

    ###############################

    # Create a dictionary mapping (protein_id, pfam_id) to list of (start, end)
    id_pfam_to_bounds = {}
    for _, row in df_eval.iterrows():
        key = (row['id'], row['Pfam_id'])
        if key not in id_pfam_to_bounds:
            id_pfam_to_bounds[key] = []
        try:
            id_pfam_to_bounds[key].append((int(row['start']), int(row['end'])))
        except ValueError:
            id_pfam_to_bounds[key].append((None, None))

    # Get list of Pfam_ids from eval
    id_to_pfam = df_eval.groupby('id')['Pfam_id'].apply(set).to_dict()
    
    # Dictionary to track counts per Pfam_id
    pfam_prediction_counts = {}
    
    for i in range(len(pred_IDs)):
        protein_id = pred_IDs[i]
        prediction = df_results.at[i, 'Prediction']
        
        if protein_id in id_to_pfam:
            match_count += 1
            
            # Check if Prediction matches ANY Pfam_id for this protein
            if prediction in id_to_pfam.get(protein_id, set()):
                prediction_match_count += 1
                
                # Count this correct prediction by Pfam_id
                if prediction not in pfam_prediction_counts:
                    pfam_prediction_counts[prediction] = 0
                pfam_prediction_counts[prediction] += 1
                
                # Get boundaries for the MATCHING Pfam_id
                key = (protein_id, prediction)
                if key in id_pfam_to_bounds:
                    pred_start = df_results.at[i, 'Domain_Start']
                    pred_end = df_results.at[i, 'Domain_End']
                    
                    # Find the closest matching boundary
                    bounds_list = id_pfam_to_bounds[key]
                    best_match = min(bounds_list, key=lambda x: abs(x[0] - pred_start) + abs(x[1] - pred_end))
                    start, end = best_match
                    
                    df_results.at[i, 'Eval_Start'] = start
                    df_results.at[i, 'Eval_End'] = end
                    df_results.at[i, 'Eval_Pfam_id'] = prediction
    
    # Print statistics per Pfam_id
    print("\n=== Correct Predictions by Pfam_id ===")
    for pfam_id in sorted(pfam_prediction_counts.keys()):
        count = pfam_prediction_counts[pfam_id]
        percentage = (count / prediction_match_count * 100) if prediction_match_count > 0 else 0
        print(f"{pfam_id}: {count} ({percentage:.2f}%)")

    df_results['Eval_Start'] = df_results['Eval_Start'].astype('Int64')  # Nullable integer
    df_results['Eval_End'] = df_results['Eval_End'].astype('Int64')

    ###############################

    print(f"Matching IDs: {match_count}")
    print(f"Matching Predictions: {prediction_match_count}")
    print(f"Ratio of Matching Predictions to Matching IDs: {prediction_match_count / match_count if match_count > 0 else 0:.4f}")
    
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
    plt.ylabel('Fraction')
    plt.title('Histogram of Start Position Differences')
    # plt.legend()
    # plt.grid(axis='y', alpha=0.75)
    plt.xlim(-1000, 1000)
    # plt.ylim(0, 1)
    # plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Histogram of Start Position Differences.png', dpi=600)
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.hist(end_diffs, histtype="bar", bins=bins, alpha=0.7, label='End Differences', color=colors[1], edgecolor='black', density=False, weights=np.ones(len(end_diffs)) / len(end_diffs))
    plt.xlabel('Error in residues')
    plt.ylabel('Fraction')
    plt.title('Histogram of End Position Differences')
    # plt.legend()
    # plt.grid(axis='y', alpha=0.75)
    plt.xticks(range(-1000, 1001, 50), [str(x) if x % 200 == 0 else '' for x in range(-1000, 1001, 50)])

    plt.xlim(-1000, 1000)
    # plt.ylim(0, 1)
    # plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Histogram of End Position Differences.png', dpi=600)

    plt.show()

def classifier_checker(df_results):
    pass
    



def main():
    df_results = opener()
    plotter(df_results)

################################
if __name__ == "__main__":
    main()