import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tbparse import SummaryReader
import matplotlib.pyplot as plt


def plotter():

    ###################################
    # TRAINING VALUES
    ###################################

    d2_prec = 0.9999
    d2_recall = 1

    d10_prec = 0.9738
    d10_recall = 0.9932

    d10_THIO_prec = 0.9564
    d10_THIO_recall = 0.9846


    # -----------------------------

    d100_prec = 0.990                        # 0.9959  - old values from only SwissProt
    d100_recall = 0.998                      # 0.9972

    d1000_prec = 0.992                       # 0.9909  - old values from only SwissProt
    d1000_recall = 0.995                    #0.9952

    # dmax_prec =
    # dmax_recall =

    # -----------------------------

    oh1d_prec = 0.9973
    oh1d_recall = 1

    oh2d_prec = 0.0735   
    oh2d_recall = 0.1021

    # Fixed: Use consistent model names and data
    models = ["D 2", "D 10", "D 10 Thiolase", "D 100", "D 1000"]
    precision_scores = [d2_prec, d10_prec, d10_THIO_prec, d100_prec, d1000_prec]
    recall_scores = [d2_recall, d10_recall, d10_THIO_recall, d100_recall, d1000_recall]

    colors = cm.get_cmap('tab10').colors
    width = 0.35

    # Plot 1: Regular models
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x1 = np.arange(len(models))
    
    # Swapped: Precision first, then Recall
    ax1.bar(x1 - width/2, precision_scores, width, label='Precision', color=colors[0], alpha=0.7)
    ax1.bar(x1 + width/2, recall_scores, width, label='Recall', color=colors[1], alpha=0.7)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('ESM Embeddings Trainings Results')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Add value labels on bars for plot 1 (swapped order)
    for i, (p, r) in enumerate(zip(precision_scores, recall_scores)):
        if p > 0:  # Only show labels for non-zero values (precision first)
            ax1.text(i - width/2, p + 0.002, f'{p:.4f}', ha='center', va='bottom', fontsize=9)
        if r > 0:  # Only show labels for non-zero values (recall second)
            ax1.text(i + width/2, r + 0.002, f'{r:.4f}', ha='center', va='bottom', fontsize=9)
    


    
    # ax1.axvline(x=2.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    # ax1.text(0.95, -0.1, 'Complete UniProt', ha='center', va='top', fontsize=10, fontweight='bold', transform=ax1.transData)
    # ax1.text(3.5, -0.1, 'SwissProt Only', ha='center', va='top', fontsize=10, fontweight='bold', transform=ax1.transData)


    plt.tight_layout()
    plt.show()


    ###################################
    # EVAL VALUES
    ###################################
    d2_prec = 0.9995
    d2_recall = np.mean([0.9652,0.9839])

    d10_prec = np.mean([0.9991,0.9999,0.9999,0.9991,0.9938,0.9941,0.9575,0.8392,0.7837,0.9176])
    d10_recall = np.mean([0.9554,0.9749,0.5508,0.9677,0.9464,0.8879,0.9767,0.6595,0.1863,0.4701])

    d10_THIO_prec = np.mean([0.9698,0.9479,0.7208,0.7972,0.5258,0.9869,0.9090,0.6123,0.3417,0.3489])
    d10_THIO_recall = np.mean([0.9915,0.9117,0.9256,0.8947,0.8167,0.7951,0.9114,0.2781,0.4571,0.0986])


    # -----------------------------

    d100_prec = 0.92495                        # 0.9429
    d100_recall = 0.755013                      # 0.9154

    d1000_prec = 0.8898                        # 0.9241
    d1000_recall = 0.6782                     # 0.8474

    # dmax_prec = 
    # dmax_recall = 



    # Fixed: Use consistent model names and data
    models = ["D 2", "D 10", "D 10 Thiolase", "D 100", "D 1000"]
    precision_scores = [d2_prec, d10_prec, d10_THIO_prec, d100_prec, d1000_prec]
    recall_scores = [d2_recall, d10_recall, d10_THIO_recall, d100_recall, d1000_recall]

    colors = cm.get_cmap('tab10').colors
    width = 0.35

    # Plot 1: Regular models
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x1 = np.arange(len(models))
    
    # Swapped: Precision first, then Recall
    ax1.bar(x1 - width/2, precision_scores, width, label='Precision', color=colors[0], alpha=0.7)
    ax1.bar(x1 + width/2, recall_scores, width, label='Recall', color=colors[1], alpha=0.7)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('ESM Embeddings Evaluation Results')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Add value labels on bars for plot 1 (swapped order)
    for i, (p, r) in enumerate(zip(precision_scores, recall_scores)):
        if p > 0:  # Only show labels for non-zero values (precision first)
            ax1.text(i - width/2, p + 0.002, f'{p:.4f}', ha='center', va='bottom', fontsize=9)
        if r > 0:  # Only show labels for non-zero values (recall second)
            ax1.text(i + width/2, r + 0.002, f'{r:.4f}', ha='center', va='bottom', fontsize=9)
    

    # ax1.axvline(x=2.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    # ax1.text(0.95, -0.1, 'Complete UniProt', ha='center', va='top', fontsize=10, fontweight='bold', transform=ax1.transData)
    # ax1.text(3.5, -0.1, 'SwissProt Only', ha='center', va='top', fontsize=10, fontweight='bold', transform=ax1.transData)
    
    plt.tight_layout()
    plt.show()




    ###################################
    # ONE-HOT MODELS
    ###################################

    # Plot 3: One-Hot D1 Model
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    modelOH_d1 = ["D 1"]
    precision_scoresOH_d1 = [oh1d_prec]
    recall_scoresOH_d1 = [oh1d_recall]
    
    x3 = np.arange(len(modelOH_d1))
    
    ax3.bar(x3 - width*1, precision_scoresOH_d1, width, label='Precision', color=colors[0], alpha=0.7)
    ax3.bar(x3 + width*1, recall_scoresOH_d1, width, label='Recall', color=colors[1], alpha=0.7)
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Score')
    # ax3.set_title('One-Hot D1 Model Training Results')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(modelOH_d1)
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (p, r) in enumerate(zip(precision_scoresOH_d1, recall_scoresOH_d1)):
        ax3.text(i - width*1, p + 0.02, f'{p:.4f}', ha='center', va='bottom', fontsize=9)
        ax3.text(i + width*1, r + 0.02, f'{r:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

    # Plot 4: One-Hot D2 Model
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    modelOH_d2 = ["D 2"]
    precision_scoresOH_d2 = [oh2d_prec]
    recall_scoresOH_d2 = [oh2d_recall]
    
    x4 = np.arange(len(modelOH_d2))
    
    ax4.bar(x4 - width*1, precision_scoresOH_d2, width, label='Precision', color=colors[0], alpha=0.7)
    ax4.bar(x4 + width*1, recall_scoresOH_d2, width, label='Recall', color=colors[1], alpha=0.7)
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Score')
    # ax4.set_title('One-Hot D2 Model Training Results')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(modelOH_d2)
    ax4.legend()
    ax4.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (p, r) in enumerate(zip(precision_scoresOH_d2, recall_scoresOH_d2)):
        ax4.text(i - width*1, p + 0.02, f'{p:.4f}', ha='center', va='bottom', fontsize=9)
        ax4.text(i + width*1, r + 0.02, f'{r:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()



    ###################################
    # 1000d EVAL HIST PLOT
    ###################################



    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/1000d_uncut_ALL"
    reader = SummaryReader(log_dir)
    df = reader.scalars
    # print('--- df.shape', df.shape)
    # print(df.head(10))
    # print('--- columns', df.columns)
    tags = df["tag"].unique()
    # print('--- all tags', tags)

    import seaborn as sns
    
    all_prec = []
    for tag in tags:
        if tag.startswith("Precision/"):
            values = df[df["tag"] == tag]["value"].values
            all_prec.extend(values)
            # print(f"Processing tag: {tag}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(all_prec, kde=False, bins=100, alpha=0.7,stat="proportion",color=colors[0])
    plt.title("Distribution of Precision for 1000 D Model (SwissProt + TrEMBL)")
    plt.xlabel("Precision")
    plt.ylabel("Fraction")
    plt.show()

    all_rec = []
    for tag in tags:
        if tag.startswith("Recall/"):
            values = df[df["tag"] == tag]["value"].values
            all_rec.extend(values)  # Changed from all_prec to all_rec
            # print(f"Processing tag: {tag}")

    plt.figure(figsize=(10, 6))
    sns.histplot(all_rec, kde=False, bins=100, alpha=0.7,stat="proportion",color=colors[1])
    plt.title("Distribution of Recall for 1000 D Model (SwissProt + TrEMBL)")
    plt.xlabel("Recall")
    plt.ylabel("Fraction")
    plt.show()

##############################################################################################

if __name__ == "__main__":
    plotter()
