import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open ("/global/research/students/sapelt/Masters/Pfam-A.hmm.dat", "r") as f:
    pfam_data = f.read()
    line_count = 0
    repeat_count = 0
    all_ml = []
    pfam_data_list = pfam_data.split("//\n")
    # print(len(pfam_data_list))
    # print(pfam_data_list[0])
    for idx,entry in enumerate(pfam_data_list):
        line_count += 1
        for idx2,line in enumerate(entry.splitlines()):
            # if idx < 5:
                # print(line)
                # print("\n")
            if line.startswith("#=GF TP   Repeat"):
                # print("check")
                repeat_count += 1
                type = line.split()[2]
                ml = pfam_data_list[idx].splitlines()
                ml = ml[6].split()[2]
                # print("ml:",ml)
                # print("type",type)
                all_ml.append(int(ml))
                break

print ("Total entries:", line_count)
print ("Total repeat entries:", repeat_count)
print ("Fraction of repeat entries:", repeat_count/line_count)
print("Average ML of repeat entries:", np.mean(all_ml),'\n')

# under 0.01
rec_under = ["26",
"42",
"57",
"78",
"94",
"101",
"102",
"105",
"109",
"115",
"118",
"124",
"134",
"138",
"140",
"145",
"157",
"173",
"176",
"180",
"182",
"207",
"213",
"221",
"229",
"231",
"250",
"253",
"270",
"289",
"298",
"302",
"323",
"330",
"335",
"348",
"359",
"364",
"370",
"371",
"394",
"409",
"419",
"452",
"455",
"468",
"484",
"486",
"493",
"497",
"503",
"504",
"507",
"511",
"516",
"517",
"518",
"523",
"525",
"531",
"536",
"545",
"566",
"576",
"595",
"611",
"616",
"624",
"651",
"695",
"698",
"705",
"715",
"721",
"725",
"732",
"756",
"757",
"771",
"789",
"792",
"810",
"811",
"831",
"845",
"846",
"854",
"862",
"867",
"871",
"883",
"934",
"946",
"954",
"956",
"967"]



with open ("/global/research/students/sapelt/Masters/pfamdict1000.txt", "r") as f:
    pfamdict_data = f.read()
    pfamdict = []
    pfamdict.append("DUMMY")
    pfamdict_lines = pfamdict_data.splitlines()
    for line in pfamdict_lines:
        # print(line)
        pfamdict.append(line.split()[0])

# print(pfamdict[1])
# print(pfamdict[1000])

pfamID_found = []

for rec in rec_under:
    # print(f"Class {rec}:", pfamdict[int(rec)])
    pfamID_found.append(pfamdict[int(rec)])

# print(pfamID_found)


# lookup in hmm.dat and check if type is actually repeat

number_of_found_repeats = 0
ml_list = []
for pfamID in pfamID_found:
    for idx,entry in enumerate(pfam_data_list):
        # print (entry)
        for line in entry.splitlines():
            # print(line,"\n")
            if line.startswith(f"#=GF AC   {pfamID.split('.')[0]}"):
                id = line.split()[2]
                id = id.split('.')[0]
                # print("Found ID:",id)
                # print(line)
                # print(entry)
                if entry.splitlines()[5].startswith("#=GF TP   Repeat"):
                    type = entry.splitlines()[5].split()[2]
                    # print(f"ID {id} is of type {type} and is a repeat.")
                    number_of_found_repeats += 1
                    ml = entry.splitlines()[6].split()[2]
                    ml_list.append(int(ml))


print("\nNumber of found repeats:", number_of_found_repeats)
print("Total Pfam IDs searched:", len(pfamID_found))
print("Fraction of found repeats:", number_of_found_repeats/len(pfamID_found))
print("Average ML of found repeats:", np.mean(ml_list))

###################################

# Create a bar plot showing both fractions
fractions = [repeat_count/line_count, number_of_found_repeats/len(pfamID_found)]
labels = ['Overall Pfam Repeat Fraction', 'Found Repeats Fraction < 0.05 recall']
width = 0.35

colors = cm.get_cmap('tab10').colors
plt.figure(figsize=(10, 6))
plt.bar(labels,fractions, color=colors[:2], alpha=0.7)
plt.ylabel('Fraction')
plt.title('Comparison of Repeat Fractions in 1000d Model')
plt.ylim(0, 1)

# Add value labels on top of bars
for i, v in enumerate(fractions):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# Create a violin plot showing both ML distributions (without outliers)
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [x for x in data if lower_bound <= x <= upper_bound]

# Remove outliers from both datasets
all_ml_clean = remove_outliers(all_ml)
ml_list_clean = remove_outliers(ml_list)

ml_data = [all_ml_clean, ml_list_clean]
ml_labels = ['All Pfam Repeats', 'Found Repeats < 0.05 recall']

plt.figure(figsize=(10, 6))
parts = plt.violinplot(ml_data, positions=[1, 2], showmeans=True, showmedians=True)

# Color the violin plots
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)

# Add mean values as text labels
means = [np.mean(data) for data in ml_data]
for i, mean_val in enumerate(means):
    plt.text(i+1, mean_val, f'Mean: {mean_val:.1f}', ha='center', va='bottom', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.xticks([1, 2], ml_labels)
plt.ylabel('Model Length (ML)')
plt.title('Distribution of HHM Lengths in 1000d Model (Outliers Removed)')

plt.tight_layout()
plt.show()
