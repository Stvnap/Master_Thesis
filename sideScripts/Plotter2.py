import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tbparse import SummaryReader


def plotter():

    ###################################
    # TRAINING VALUES
    ###################################

    # get the values from the training logs logged by tensorboard


    colors = cm.get_cmap('tab10').colors


    # manually entered values from d2 and d10 model
    d2_prec = 0.9998,1
    d2_recall = 1,1

    d10_prec = 0.9984, 0.9993, 0.9999, 0.9997, 0.9947, 0.9953, 0.9885, 0.9659, 0.8717, 0.9024
    d10_recall = 0.997, 0.9994, 0.9997, 0.9996, 0.9805, 0.9957, 0.9839, 0.9956, 0.9984, 0.9896


    # log dir 2d
    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/t33_ALL_2d/"
    reader = SummaryReader(log_dir,extra_columns={"dir_name"})
    df = reader.scalars
    tags = df["tag"].unique()

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_prec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_2d_final/version_0"]  
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Precision for", class_tag, ":", np.max(values))
                prec_vals.append(np.max(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_rec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_2d_final/version_0"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Recall for", class_tag, ":", np.max(values))
                rec_vals.append(np.max(values))
    
    # remove last entry has it is the average over all classes
    prec_vals = prec_vals[:-1]
    rec_vals = rec_vals[:-1]

    print("Len prec_vals:",len(prec_vals))

    d2_prec = prec_vals
    d2_recall = rec_vals   


    # -----------------------------

    # log dir 10d
    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/t33_ALL_10d/"
    reader = SummaryReader(log_dir,extra_columns={"dir_name"})
    df = reader.scalars
    tags = df["tag"].unique()

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_prec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_10d_final/version_3"]  
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Precision for", class_tag, ":", np.max(values))
                prec_vals.append(np.max(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_rec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_10d_final/version_3"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Recall for", class_tag, ":", np.max(values))
                rec_vals.append(np.max(values))
    
    # remove last entry has it is the average over all classes
    prec_vals = prec_vals[:-1]
    rec_vals = rec_vals[:-1]


    d10_prec = prec_vals
    d10_recall = rec_vals   

    # -----------------------------

    # log dir thio
    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/t33_ALL_10d_THIO/"
    reader = SummaryReader(log_dir,extra_columns={"dir_name"})
    df = reader.scalars
    tags = df["tag"].unique()

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_prec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_10d_THIO_final/version_0"]  
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Precision for", class_tag, ":", np.max(values))
                prec_vals.append(np.max(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_rec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_10d_THIO_final/version_0"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Recall for", class_tag, ":", np.max(values))
                rec_vals.append(np.max(values))
    
    # remove last entry has it is the average over all classes
    prec_vals = prec_vals[:-1]
    rec_vals = rec_vals[:-1]

    d10_THIO_prec = prec_vals
    d10_THIO_recall = rec_vals

    # -----------------------------
    # log dir 100d
    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/t33_ALL_100d/"
    reader = SummaryReader(log_dir,extra_columns={"dir_name"})
    df = reader.scalars
    tags = df["tag"].unique()
    # print("Tags found:",tags)

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_prec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_100d_final/version_1"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Precision for", class_tag, ":", np.max(values))
                prec_vals.append(np.max(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_rec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_100d_final/version_1"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Recall for", class_tag, ":", np.max(values))
                rec_vals.append(np.max(values))

    # remove last entry has it is the average over all classes
    prec_vals = prec_vals[:-1]
    rec_vals = rec_vals[:-1]

    d100_prec = prec_vals                        # 0.9959  - old values from only SwissProt
    d100_recall = rec_vals                      # 0.9972

    # -----------------------------

    # log dir 1000d
    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/t33_ALL_1000d/"
    reader = SummaryReader(log_dir,extra_columns={"dir_name"})
    df = reader.scalars
    tags = df["tag"].unique()

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_prec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_1000d_final/version_1"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Precision for", class_tag, ":", np.max(values))
                prec_vals.append(np.max(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("val_rec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_1000d_final/version_1"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Recall for", class_tag, ":", np.max(values))
                rec_vals.append(np.max(values))


    # remove last entry has it is the average over all classes
    prec_vals = prec_vals[:-1]
    rec_vals = rec_vals[:-1]

    d1000_prec = prec_vals                      # 0.9909  - old values from only SwissProt
    d1000_recall = rec_vals                    #0.9952


    # -----------------------------
    # log dir 24380d

    # log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/t33_ALL_24380d/"
    # reader = SummaryReader(log_dir,extra_columns={"dir_name"})
    # df = reader.scalars
    # tags = df["tag"].unique()

    # prec_vals = []
    # for class_tag in tags:
    #     if class_tag.startswith("val_prec"):  
    #         run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_24380d_final/version_1"]  
    #         if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
    #             values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
    #             # print("Max Precision for", class_tag, ":", np.max(values))
    #             prec_vals.append(np.max(values))

    # rec_vals = []
    # for class_tag in tags:
    #     if class_tag.startswith("val_rec"):  
    #         run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_24380d_final/version_1"]  
    #         if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
    #             values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
    #             # print("Max Recall for", class_tag, ":", np.max(values))
    #             rec_vals.append(np.max(values))


    # # remove last entry has it is the average over all classes
    # prec_vals = prec_vals[:-1]
    # rec_vals = rec_vals[:-1]

    # dmax_prec =
    # dmax_recall =

    # -----------------------------

    models = ["D 2", "D 10", "D 10 Thiolase", "D 100", "D 1000"]
    precision_data = [d2_prec, d10_prec, d10_THIO_prec, d100_prec, d1000_prec]
    recall_data = [d2_recall, d10_recall, d10_THIO_recall, d100_recall, d1000_recall]


    # Violin plot for Precision
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    violin1 = ax1.violinplot(precision_data, positions=range(len(models)), showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin1['bodies']):
        pc.set_facecolor(colors[0])
        pc.set_alpha(0.7)

    # Change antenna colors to black
    for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
        if part in violin1:
            violin1[part].set_color('dimgray')

    # Add mean values as text with better visibility
    for i, data in enumerate(precision_data):
        mean_val = np.mean(data)
        ax1.text(i,0.45 + (1.01-0.45)/2, f'{mean_val:.3f}', ha='center', va='bottom', 
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', alpha=0.8))

    ax1.set_xlabel('Models')
    ax1.set_ylabel('Precision')
    ax1.set_title('ESM Embeddings Training Results - Precision Distribution')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models)
    ax1.set_ylim(0.45, 1.01)

    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/ESM Embeddings Training Results - Precision Distribution.png', dpi=600)

    plt.show()


    # Violin plot for Recall
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    violin2 = ax2.violinplot(recall_data, positions=range(len(models)), showmeans=True, showmedians=True)
    

    # Change antenna colors to black
    for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
        if part in violin2:
            violin2[part].set_color('dimgray')


    for i, pc in enumerate(violin2['bodies']):
        pc.set_facecolor(colors[1])
        pc.set_alpha(0.7)

    # Add mean values as text with better visibility
    for i, data in enumerate(recall_data):
        mean_val = np.mean(data)
        ax2.text(i, 0.75 + (1.01-0.75)/2, f'{mean_val:.3f}', ha='center', va='top', 
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', alpha=0.8))

    ax2.set_xlabel('Models')
    ax2.set_ylabel('Recall')
    ax2.set_title('ESM Embeddings Training Results - Recall Distribution')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models)
    ax2.set_ylim(0.75, 1.01)


    
    # ax1.axvline(x=2.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    # ax1.text(0.95, -0.1, 'Complete UniProt', ha='center', va='top', fontsize=10, fontweight='bold', transform=ax1.transData)
    # ax1.text(3.5, -0.1, 'SwissProt Only', ha='center', va='top', fontsize=10, fontweight='bold', transform=ax1.transData)


    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/ESM Embeddings Training Results - Recall Distribution.png', dpi=600)

    plt.show()


    ###################################
    # EVAL VALUES
    ###################################


    # d2_prec = 0.9995
    # d2_recall = 0.9652,0.9839

    # d10_prec = 0.9991,0.9999,0.9999,0.9991,0.9938,0.9941,0.9575,0.8392,0.7837,0.9176
    # d10_recall = 0.9554,0.9749,0.5508,0.9677,0.9464,0.8879,0.9767,0.6595,0.1863,0.4701

    # d10_THIO_prec = 0.9698,0.9479,0.7208,0.7972,0.5258,0.9869,0.9090,0.6123,0.3417,0.3489
    # d10_THIO_recall = 0.9915,0.9117,0.9256,0.8947,0.8167,0.7951,0.9114,0.2781,0.4571,0.0986


    # -----------------------------

    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/2d_uncut_ALL/"
    reader = SummaryReader(log_dir)
    df = reader.scalars
    tags = df["tag"].unique()
    print(df)
    print("Tags found:",tags)

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Precision/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Precision for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                prec_vals.append(values[0] if len(values) == 1 else np.mean(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Recall/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Recall for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                rec_vals.append(values[0] if len(values) == 1 else np.mean(values))

    print("Len prec_vals:",len(prec_vals))
    print("Len rec_vals:",len(rec_vals))
    d2_prec = prec_vals
    d2_recall = rec_vals


    # -----------------------------

    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/10d_uncut_ALL/"
    reader = SummaryReader(log_dir)
    df = reader.scalars
    tags = df["tag"].unique()
    print(df)
    print("Tags found:",tags)

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Precision/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Precision for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                prec_vals.append(values[0] if len(values) == 1 else np.mean(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Recall/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Recall for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                rec_vals.append(values[0] if len(values) == 1 else np.mean(values))

    print("Len prec_vals:",len(prec_vals))
    print("Len rec_vals:",len(rec_vals))

    d10_prec = prec_vals
    d10_recall = rec_vals

    # -----------------------------

    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/10d_uncut_ALL_THIO/"
    reader = SummaryReader(log_dir)
    df = reader.scalars
    tags = df["tag"].unique()
    print(df)
    print("Tags found:",tags)

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Precision/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Precision for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                prec_vals.append(values[0] if len(values) == 1 else np.mean(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Recall/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Recall for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                rec_vals.append(values[0] if len(values) == 1 else np.mean(values))

    print("Len prec_vals:",len(prec_vals))
    print("Len rec_vals:",len(rec_vals))

    d10_THIO_prec = prec_vals
    d10_THIO_recall = rec_vals


    # -----------------------------
    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/100d_uncut_ALL/"
    reader = SummaryReader(log_dir)
    df = reader.scalars
    tags = df["tag"].unique()
    print(df)
    print("Tags found:",tags)

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Precision/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Precision for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                prec_vals.append(values[0] if len(values) == 1 else np.mean(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Recall/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Recall for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                rec_vals.append(values[0] if len(values) == 1 else np.mean(values))
            
    print("Len prec_vals:",len(prec_vals))
    print("Len rec_vals:",len(rec_vals))
    print("Precision values:",prec_vals)
    print("Recall values:",rec_vals)
    d100_prec = prec_vals                         # 0.9429
    d100_recall = rec_vals                       # 0.9154
    
    # -----------------------------



    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/1000d_uncut_ALL/"
    reader = SummaryReader(log_dir)
    df = reader.scalars
    tags = df["tag"].unique()
    print(df)
    print("Tags found:",tags)

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Precision/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Precision for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                prec_vals.append(values[0] if len(values) == 1 else np.mean(values))

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("Recall/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Recall for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                rec_vals.append(values[0] if len(values) == 1 else np.mean(values))

            
    print("Len prec_vals:",len(prec_vals))
    print("Len rec_vals:",len(rec_vals))
    d1000_prec = prec_vals                        # 0.9241
    d1000_recall = rec_vals                      # 0.8474

    # -----------------------------
    # log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/24380d_uncut_ALL/"
    # reader = SummaryReader(log_dir)
    # df = reader.scalars
    # tags = df["tag"].unique()
    # print(tags)

    # prec_vals = []
    # for class_tag in tags:
    #     if class_tag.startswith("Precision/"):  
    #         values = df[df["tag"] == class_tag]["value"].values
    #         # print("Max Precision for", class_tag, ":", values)
    #         prec_vals.append(values)

    # rec_vals = []
    # for class_tag in tags:
    #     if class_tag.startswith("Recall/"):  
    #         values = df[df["tag"] == class_tag]["value"].values
    #         # print("Max Recall for", class_tag, ":", values)
    #         rec_vals.append(values)
 
    # dmax_prec = 
    # dmax_recall = 

    # -----------------------------


    # Fixed: Use consistent model names and data
    models = ["D 2", "D 10", "D 10 Thiolase", "D 100", "D 1000"]
    precision_data = [d2_prec, d10_prec, d10_THIO_prec, d100_prec, d1000_prec]
    recall_data = [d2_recall, d10_recall, d10_THIO_recall, d100_recall, d1000_recall]

    colors = cm.get_cmap('tab10').colors

    # Violin plot for Precision
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    violin1 = ax1.violinplot(precision_data, positions=range(len(models)), showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin1['bodies']):
        pc.set_facecolor(colors[0])
        pc.set_alpha(0.7)

    for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
        if part in violin1:
            violin1[part].set_color('dimgray')

    # Add mean values as text with better visibility
    for i, data in enumerate(precision_data):
        mean_val = np.mean(data)
        ax1.text(i, (1.02/2), f'{mean_val:.3f}', ha='center', va='bottom', 
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', alpha=0.8))

    ax1.set_xlabel('Models')
    ax1.set_ylabel('Precision')
    ax1.set_title('ESM Embeddings Evaluation Results - Precision Distribution')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models)
    ax1.set_ylim(-0.01, 1.01)

    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/ESM Embeddings Evaluation Results - Precision Distribution.png', dpi=600)

    plt.show()

    # Violin plot for Recall
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    violin2 = ax2.violinplot(recall_data, positions=range(len(models)), showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin2['bodies']):
        pc.set_facecolor(colors[1])
        pc.set_alpha(0.7)

    for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
        if part in violin2:
            violin2[part].set_color('dimgray')

    # Add mean values as text with better visibility
    for i, data in enumerate(recall_data):
        mean_val = np.mean(data)
        ax2.text(i, (1.02/2), f'{mean_val:.3f}', ha='center', va='top', 
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', alpha=0.8))

    ax2.set_xlabel('Models')
    ax2.set_ylabel('Recall')
    ax2.set_title('ESM Embeddings Evaluation Results - Recall Distribution')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models)
    ax2.set_ylim(-0.01, 1.01)

    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/ESM Embeddings Evaluation Results - Recall Distribution.png', dpi=600)

    plt.show()




    ###################################
    # ONE-HOT MODELS
    ###################################


    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logshp/tb_1d_combined/tb_20250406_145252"
    reader = SummaryReader(log_dir, extra_columns={"dir_name"} )
    df = reader.tensors
    # print(df)
    tags = df["tag"].unique()
    # print("Tags found:",tags)

    prec_vals = []
    for class_tag in tags:
        if class_tag.startswith("epoch_precision"):  
            tag_df = df[df["tag"] == class_tag]
            for idx, row in tag_df.iterrows():
                dir_name = row["dir_name"]
                if "validation" in dir_name:  # Changed from endswith to 'in'
                    value = row["value"]
                    if isinstance(value, (list, np.ndarray)):
                        prec_vals.extend(value)
                    else:
                        prec_vals.append(value)

    rec_vals = []
    for class_tag in tags:
        if class_tag.startswith("epoch_recall"):  
            tag_df = df[df["tag"] == class_tag]
            # print(tag_df)
            for idx, row in tag_df.iterrows():
                dir_name = row["dir_name"]
                # print(dir_name)
                if "validation" in dir_name:  # Changed from endswith to 'in'
                    value = row["value"]
                    if isinstance(value, (list, np.ndarray)):
                        rec_vals.extend(value)
                    else:
                        rec_vals.append(value)


    print("Len prec_vals:",len(prec_vals))
    # print(prec_vals)
    print("Len rec_vals:",len(rec_vals))


    oh1d_prec = prec_vals                    # 0.9973
    oh1d_recall = rec_vals                 #1



    ##################

    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logshp/HPsearch2d_Home"
    reader = SummaryReader(log_dir, extra_columns={"dir_name"} )
    df = reader.tensors
    tags = df["tag"].unique()
    print(df)
    print("Tags found:",tags)
    


    ## !!!!!!!!! SINLGE OUT EVALUATION VALUES ONLY !!!!!!!!!!

    prec_vals_1d = []
    for class_tag in tags:
        if class_tag.startswith("epoch_prec_1"):
            tag_df = df[df["tag"] == class_tag]
            for idx, row in tag_df.iterrows():
                dir_name = row["dir_name"]
                if "validation" in dir_name:  # Changed to check if 'validation' is in dir_name
                    value = row["value"]
                    if isinstance(value, (list, np.ndarray)):
                        prec_vals_1d.extend(value)
                    else:
                        prec_vals_1d.append(value)

    rec_vals_1d = []
    for class_tag in tags:
        if class_tag.startswith("epoch_rec_1"):  
            tag_df = df[df["tag"] == class_tag]
            for idx, row in tag_df.iterrows():
                dir_name = row["dir_name"]
                if "validation" in dir_name:
                    value = row["value"]
                    if isinstance(value, (list, np.ndarray)):
                        rec_vals_1d.extend(value)
                    else:
                        rec_vals_1d.append(value)

    prec_vals_2d = []
    for class_tag in tags:
        if class_tag.startswith("epoch_prec_2"):  
            tag_df = df[df["tag"] == class_tag]
            for idx, row in tag_df.iterrows():
                dir_name = row["dir_name"]
                if "validation" in dir_name:
                    value = row["value"]
                    if isinstance(value, (list, np.ndarray)):
                        prec_vals_2d.extend(value)
                    else:
                        prec_vals_2d.append(value)

    rec_vals_2d = []
    for class_tag in tags:
        if class_tag.startswith("epoch_rec_2"):  
            tag_df = df[df["tag"] == class_tag]
            for idx, row in tag_df.iterrows():
                dir_name = row["dir_name"]
                if "validation" in dir_name:
                    value = row["value"]
                    if isinstance(value, (list, np.ndarray)):
                        rec_vals_2d.extend(value)
                    else:
                        rec_vals_2d.append(value)


    print("Len prec_vals:",len(prec_vals_1d))
    print("Len rec_vals:",len(rec_vals_1d))

    print("Len prec_vals:",len(prec_vals_2d))
    print("Len rec_vals:",len(rec_vals_2d))

    oh2d_prec = prec_vals                        #0.0735   
    oh2d_recall = rec_vals                      # 0.1021

    width = 0.35

    # Plot 3: One-Hot D1 Model (Violin Plot)
    fig3 = plt.figure(figsize=(8, 5))
    ax3 = fig3.add_subplot(111)

    
    # Combine precision and recall data for violin plot
    combined_data = [oh1d_prec, oh1d_recall]
    labels = ['Precision', 'Recall']
    
    violin3 = ax3.violinplot(combined_data, positions=range(len(labels)), showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin3['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Change antenna colors to black
    for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
        if part in violin3:
            violin3[part].set_color('dimgray')

    # Add mean values as text
    for i, data in enumerate(combined_data):
        mean_val = np.mean(data)
        ylim = ax3.get_ylim()
        y_middle = (ylim[0] + ylim[1]) / 2
        ax3.text(i, y_middle, f'{mean_val:.3f}', ha='center', va='bottom', 
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', alpha=0.8))

    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('One-Hot D1 Model Training Results')
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels)
    
    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/One-Hot D1 Model Training Results.png', dpi=600)

    plt.show()
    # Plot 4: One-Hot D2 Model (Violin Plot)
    fig4 = plt.figure(figsize=(8, 5))
    ax4 = fig4.add_subplot(111)
    
    # Separate precision and recall data for violin plot
    combined_data = [prec_vals_1d, rec_vals_1d, prec_vals_2d, rec_vals_2d]
    labels = ['Precision', 'Recall', 'Precision', 'Recall']
    
    violin4 = ax4.violinplot(combined_data, positions=range(len(labels)), showmeans=True, showmedians=True)
    
    for i, pc in enumerate(violin4['bodies']):
        if i % 2 == 0:
            pc.set_facecolor(colors[0])
            pc.set_alpha(0.7)
        else:
            pc.set_facecolor(colors[1])
            pc.set_alpha(0.7)

    # Change antenna colors to black
    for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
        if part in violin4:
            violin4[part].set_color('dimgray')

    # Add mean values as text
    for i, data in enumerate(combined_data):
        mean_val = np.mean(data)
        ylim = ax4.get_ylim()
        y_middle = (ylim[0] + ylim[1]) / 2
        ax4.text(i, y_middle, f'{mean_val:.3f}', ha='center', va='bottom', 
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', alpha=0.8))

    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Score')
    ax4.set_title('One-Hot Model Training Results - D1 vs D2')
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels)
    # Add a second row of x-axis labels for D1 and D2 classes
    ax4.text(0.5, ax4.get_ylim()[0] - 0.1 * (ax4.get_ylim()[1] - ax4.get_ylim()[0]), 
             'Class 1', ha='center', va='top', fontsize=10, 
             transform=ax4.transData)
    ax4.text(2.5, ax4.get_ylim()[0] - 0.1 * (ax4.get_ylim()[1] - ax4.get_ylim()[0]), 
             'Class 2', ha='center', va='top', fontsize=10, 
             transform=ax4.transData)
    
    ax4.axvline(x=1.5, color='black', linestyle='--', alpha=0.7, linewidth=1.5)

    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/One-Hot Model Training Results - D1 vs D2.png', dpi=600)

    plt.show()



    ###################################
    # 1000d EVAL HIST PLOT
    ###################################



    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/1000d_uncut_ALL"
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
    
    plt.figure(figsize=(8, 5))
    sns.histplot(all_prec, kde=False, bins=100, alpha=0.7,stat="proportion",color=colors[0])
    plt.title("Distribution of Precision for 1000 D Model (SwissProt + TrEMBL)")
    plt.xlabel("Precision")
    plt.ylabel("Fraction")
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Distribution of Precision for 1000 D Model (SwissProt + TrEMBL).png', dpi=600)

    plt.show()

    all_rec = []
    for tag in tags:
        if tag.startswith("Recall/"):
            values = df[df["tag"] == tag]["value"].values
            all_rec.extend(values)  # Changed from all_prec to all_rec
            # print(f"Processing tag: {tag}")

    plt.figure(figsize=(8, 5))
    sns.histplot(all_rec, kde=False, bins=100, alpha=0.7,stat="proportion",color=colors[1])
    plt.title("Distribution of Recall for 1000 D Model (SwissProt + TrEMBL)")
    plt.xlabel("Recall")
    plt.ylabel("Fraction")
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Distribution of Recall for 1000 D Model (SwissProt + TrEMBL).png', dpi=600)

    plt.show()

##############################################################################################


    # -------------------------
    # Final Model Plots
    # -------------------------



    # trainings plot vs Evaluation plot 

    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/t33_ALL_23480d/"
    reader = SummaryReader(log_dir,extra_columns={"dir_name"})
    df = reader.scalars
    tags = df["tag"].unique()

    # train values
    prec_vals_train = []
    for class_tag in tags:
        if class_tag.startswith("val_prec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_24380d_final/version_0"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Precision for", class_tag, ":", np.max(values))
                prec_vals_train.append(np.max(values))
 
    rec_vals_train = []
    for class_tag in tags:
        if class_tag.startswith("val_rec"):  
            run_filtered_df = df[df["dir_name"] == "tensorboard/t33_ALL_24380d_final/version_0"]
            if run_filtered_df[run_filtered_df["tag"] == class_tag].shape[0] > 0:
                values = run_filtered_df[run_filtered_df["tag"] == class_tag]["value"].values
                # print("Max Recall for", class_tag, ":", np.max(values))
                rec_vals_train.append(np.max(values))

    # remove last entry has it is the average over all classes
    prec_vals_train = rec_vals_train[:-1]
    rec_vals_train = rec_vals_train[:-1]


    # Eval values

    log_dir = "/global/research/students/sapelt/Masters/MasterThesis/models/FINAL/24380d_uncut_ALL/"
    reader = SummaryReader(log_dir)
    df = reader.scalars
    tags = df["tag"].unique()
    print(df)
    print("Tags found:",tags)

    prec_vals_eval = []
    for class_tag in tags:
        if class_tag.startswith("Precision/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Precision for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                prec_vals_eval.append(values[0] if len(values) == 1 else np.mean(values))

    rec_vals_eval = []
    for class_tag in tags:
        if class_tag.startswith("Recall/"):  
            values = df[df["tag"] == class_tag]["value"].values
            # print("Max Recall for", class_tag, ":", values)
            # Extract single value or take mean if multiple values
            if len(values) > 0:
                rec_vals_eval.append(values[0] if len(values) == 1 else np.mean(values))


    # Violin plot comparing Training vs Evaluation for Full Model
    models_comp = ["Training", "Evaluation"]
    precision_data_comp = [prec_vals_train, prec_vals_eval]
    recall_data_comp = [rec_vals_train, rec_vals_eval]

    # Violin plot for Precision comparison
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    violin5 = ax5.violinplot(precision_data_comp, positions=range(len(models_comp)), showmeans=True, showmedians=True)

    for i, pc in enumerate(violin5['bodies']):
        pc.set_facecolor(colors[0])
        pc.set_alpha(0.7)

    # Change antenna colors to black
    for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
        if part in violin5:
            violin5[part].set_color('dimgray')


    # Add mean values as text
    for i, data in enumerate(precision_data_comp):
        mean_val = np.mean(data)
        ax5.text(i, 0.5, f'{mean_val:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', alpha=0.8))

    ax5.set_xlabel('Phase')
    ax5.set_ylabel('Precision')
    ax5.set_title('Full Model (D 24380) - Training vs Evaluation Precision')
    ax5.set_xticks(range(len(models_comp)))
    ax5.set_xticklabels(models_comp)
    ax5.set_ylim(0, 1.01)

    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Full Model (D 24380) - Training vs Evaluation Precision.png', dpi=600)

    plt.show()

    # Violin plot for Recall comparison
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    violin6 = ax6.violinplot(recall_data_comp, positions=range(len(models_comp)), showmeans=True, showmedians=True)

    # Change antenna colors to black
    for part in ['cmaxes', 'cmins', 'cbars','cmeans','cmedians']:
        if part in violin6:
            violin6[part].set_color('dimgray')

    for i, pc in enumerate(violin6['bodies']):
        pc.set_facecolor(colors[1])
        pc.set_alpha(0.7)

    # Add mean values as text
    for i, data in enumerate(recall_data_comp):
        mean_val = np.mean(data)
        ax6.text(i, 0.5, f'{mean_val:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='black', alpha=0.8))

    ax6.set_xlabel('Phase')
    ax6.set_ylabel('Recall')
    ax6.set_title('Full Model (D 24380) - Training vs Evaluation Recall')
    ax6.set_xticks(range(len(models_comp)))
    ax6.set_xticklabels(models_comp)
    ax6.set_ylim(0, 1.01)

    plt.tight_layout()
    plt.savefig('/home/sapelt/Documents/Master/FINAL/Full Model (D 24380) - Training vs Evaluation Recall.png', dpi=600)

    plt.show()




# def transformer_plotter():
#     log_dir = "/global/research/students/sapelt/Masters/MasterThesis/logs/FINAL/Optuna_uncut_t33_domains_boundary"
#     reader = SummaryReader(log_dir)
#     df = reader.scalars
#     tags = df["tag"].unique()
#     print(df)
#     print("Tags found:",tags)

#     all



##############################################################################################

if __name__ == "__main__":
    plotter()
    # transformer_plotter()
