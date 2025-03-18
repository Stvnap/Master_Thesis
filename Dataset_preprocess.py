###################################################################################################################################
# This scipt creates a final dataset ready for training the model.
# all input files need to be preprocessed annotated by InterproScan & striped by sequence-fairies-extractDomains
# it uses one positive Domain file that is targeted to be classified, negative domains (4 in this test case)
# and random protein sequences from swissprot and trembl (2.5 M)
# the input length is determined by the positive datasets 0.65 quantile
# all sequences above this length are cut into the targeted length using a sliding window apporoach with overlap
# all sequences below this length are padded with 'X' to the targeted length (x=unknown amino acid)
# the final dataset is saved as a numpy array for further use
###################################################################################################################################

import re
from scipy import stats
import numpy as np
import pandas as pd


class DomainProcessing:
    def __init__(self, domain_path):
        self.domain_path = domain_path
        self.boundaries_all = []
        self.sequence_all = []
        self.id_all = []
        self._opener()

    def _opener(self):
        with open(self.domain_path, "r") as file:
            sequence = []

            for line in file:
                line = line.strip()
                id_whole = line
                # print(line)
                # print('\n')
                if line.startswith(">"):
                    sequence_joined = "".join(sequence)
                    self.sequence_all.append(sequence_joined)

                    try:
                        boundaries = re.search(r"_(\d+-\d+)", line)
                        boundaries = boundaries.group(1)
                        self.boundaries_all.append(boundaries)
                    except:
                        continue

                    self.id_all.append(id_whole)

                    self.id_whole = line
                    # print(id_whole)

                    sequence = []
                # print(boundaries_all[0])
                else:
                    sequence.append(line)
            # print(id_all)

            # print(sequence_all)

            # print(len(id_all))
            # print(len(sequence_all))
            # print(len(boundaries_all))

    def seq_array_returner(self):
        self.sequence_all = pd.DataFrame(self.sequence_all)
        self.sequence_all.columns = ["Sequences"]
        return self.sequence_all

    def len_finder(self):
        return [len(seq) for seq in self.sequence_all]

    def distribution_finder_and_cleaner(self, seqlen):
        seqarraylen = np.array(seqlen)
        shapiro = stats.shapiro(seqarraylen)
        # return shapiro
        seqarraylen_clean = seqarraylen  # [(seqarraylen>=np.quantile(seqarraylen,0.125/2)) & (seqarraylen<=np.quantile(seqarraylen,0.875))]
        # print(seqarraylen_clean)

        seqarray = pd.DataFrame(self.sequence_all)
        seqarray_clean = seqarray  # [(np.char.str_len(seqarray)>=np.quantile(seqarraylen,0.125/2)) & (np.char.str_len(seqarray)<=np.quantile(seqarraylen,0.875))]

        shapiro = stats.shapiro(seqarraylen_clean)

        return seqarray_clean, seqarraylen_clean, shapiro

    def dimension_finder(self, seqarray_len):
        # print(seqarray_len)
        seqarray_len_clean = int(np.quantile(seqarray_len, 0.65))
        return seqarray_len_clean


fasta = DomainProcessing("/global/research/students/sapelt/Masters/domains_PF00177.fa")


################################################################### POSITIVE PF00177 #########################################################################


seqlen = fasta.len_finder()
# print(seqlen)

seqarray = fasta.seq_array_returner()
# print(seqarray)

seqarray_clean, seqarraylen_clean, normaltest = fasta.distribution_finder_and_cleaner(
    seqlen
)
# print(seqarray_clean)
print("len:", len(seqarray_clean), "max:", max(seqarraylen_clean))


# plt.hist(seqlen, bins=100)
# plt.show()


# plt.hist(seqarraylen_clean, bins=100)
# plt.show()
print(normaltest)


dimension_positive = fasta.dimension_finder(seqarraylen_clean)
print("targeted dimension", dimension_positive)

# max length is now 155, needs to be padded to len(seqarraylen_cleaned)

############################################# creation of negative dataset only containing other domains####################################################

################ PF00079 ########################

fasta = DomainProcessing("/global/research/students/sapelt/Masters/domains_PF00079.fa")

seqlen_PF00079 = fasta.len_finder()
seqarray_PF00079 = fasta.seq_array_returner()
seqarray_clean_PF00079, seqarraylen_clean_PF00079, normaltest_PF00079 = (
    fasta.distribution_finder_and_cleaner(seqlen_PF00079)
)

# print(seqarray_clean_PF00079)
print("len:", len(seqarray_clean_PF00079), "max:", max(seqarraylen_clean_PF00079))


# plt.hist(seqlen_PF00079, bins=100)
# plt.show()

# plt.hist(seqarraylen_clean_PF00079, bins=100)
# plt.show()
print(normaltest_PF00079)

dimension_PF00079 = fasta.dimension_finder(seqarraylen_clean_PF00079)
print("targeted dimension", dimension_PF00079)

print("\n")
################ PF00080 ########################

fasta = DomainProcessing("/global/research/students/sapelt/Masters/domains_PF00080.fa")

seqlen_PF00080 = fasta.len_finder()
seqarray_PF00080 = fasta.seq_array_returner()
seqarray_clean_PF00080, seqarraylen_clean_PF00080, normaltest_PF00080 = (
    fasta.distribution_finder_and_cleaner(seqlen_PF00080)
)

# print(seqarray_clean_PF00080)
print("len:", len(seqarray_clean_PF00080), "max:", max(seqarraylen_clean_PF00080))

# plt.hist(seqlen_PF00080, bins=100)
# plt.show()

# plt.hist(seqarraylen_clean_PF00080, bins=100)
# plt.show()
print(normaltest_PF00080)

dimension_PF00080 = fasta.dimension_finder(seqarraylen_clean_PF00080)
print("targeted dimension", dimension_PF00080)


print("\n")

# ################ PF00082 ########################
# fasta=DomainProcessing('/global/research/students/sapelt/Masters/domains_PF00082.fa')

# seqlen_PF00082=fasta.len_finder()
# seqarray_PF00082=fasta.seq_array_returner()
# seqarray_clean_PF00082,seqarraylen_clean_PF00082, normaltest_PF00082=fasta.distribution_finder_and_cleaner(seqlen_PF00082)

# print(seqarray_clean_PF00082)
# print(len(seqarray_clean_PF00082),max(seqarraylen_clean_PF00082))

# plt.hist(seqlen_PF00082, bins=100)
# plt.show()

# plt.hist(seqarraylen_clean_PF00082, bins=100)
# plt.show()
# print(normaltest_PF00082)

################ PF00118 ########################
fasta = DomainProcessing("/global/research/students/sapelt/Masters/domains_PF00118.fa")

seqlen_PF00118 = fasta.len_finder()
seqarray_PF00118 = fasta.seq_array_returner()
seqarray_clean_PF00118, seqarraylen_clean_PF00118, normaltest_PF00118 = (
    fasta.distribution_finder_and_cleaner(seqlen_PF00118)
)

# print(seqarray_clean_PF00118)
print("len:", len(seqarray_clean_PF00118), "max:", max(seqarraylen_clean_PF00118))

# plt.hist(seqlen_PF00118, bins=100)
# plt.show()

# plt.hist(seqarraylen_clean_PF00118, bins=100)
# plt.show()
print(normaltest_PF00118)

dimension_PF00118 = fasta.dimension_finder(seqarraylen_clean_PF00118)
print("targeted dimension", dimension_PF00118)


print("\n")

################ PF00162 ########################
fasta = DomainProcessing("/global/research/students/sapelt/Masters/domains_PF00162.fa")

seqlen_PF00162 = fasta.len_finder()
seqarray_PF00162 = fasta.seq_array_returner()
seqarray_clean_PF00162, seqarraylen_clean_PF00162, normaltest_PF00162 = (
    fasta.distribution_finder_and_cleaner(seqlen_PF00162)
)

# print(seqarray_clean_PF00162)
print("len:", len(seqarray_clean_PF00162), "max:", max(seqarraylen_clean_PF00162))

# plt.hist(seqlen_PF00162, bins=100)
# plt.show()

# plt.hist(seqarraylen_clean_PF00162, bins=100)
# plt.show()
print(normaltest_PF00162)

dimension_PF00162 = fasta.dimension_finder(seqarraylen_clean_PF00162)
print("targeted dimension", dimension_PF00162)

print("\n", "negative rnd:", "\n")


################################################### random protein sequences from swiss prot #######################################################

fasta = DomainProcessing(
    "/global/research/students/sapelt/Masters/alluniprot/sprot_domains.fa"
)

seqlen_rnd_sprot = fasta.len_finder()
seqarray_rnd_sprot = fasta.seq_array_returner()
seqarray_clean_rnd_sprot, seqarraylen_clean_rnd_sprot, normaltest_rnd_sprot = (
    fasta.distribution_finder_and_cleaner(seqlen_rnd_sprot)
)

# print(seqarray_clean_rnd_sprot)
print(len(seqarray_clean_rnd_sprot), max(seqarraylen_clean_rnd_sprot))

# plt.hist(seqlen_rnd_sprot, bins=100)
# plt.show()

print(normaltest_rnd_sprot)

dimension_rnd_sprot = fasta.dimension_finder(seqarraylen_clean_rnd_sprot)
print(dimension_rnd_sprot)

print("\n")

print(len(seqarray_clean_rnd_sprot))

print("\n")

# fasta=DomainProcessing('/global/research/students/sapelt/Masters/alluniprot/uniprot_trembl.fasta')

# seqlen_rnd_trembl=fasta.len_finder()
# seqarray_rnd_trembl=fasta.seq_array_returner()
# seqarray_clean_rnd_trembl,seqarraylen_clean_rnd_trembl, normaltest_rnd_trembl=fasta.distribution_finder_and_cleaner(seqlen_rnd_trembl)

# # print(seqarray_clean_rnd_trembl)
# print(len(seqarray_clean_rnd_trembl),max(seqarraylen_clean_rnd_trembl))

# # plt.hist(seqlen_rnd_trembl, bins=100)
# # plt.show()

# print(normaltest_rnd_trembl)

# dimension_rnd_trembl=fasta.dimension_finder(seqarraylen_clean_rnd_trembl)
# print(dimension_rnd_trembl)

# print('\n')


# # combine the random protein sequences from swissprot and trembl

# seqarray_clean_rnd=np.concatenate(seqarray_clean_rnd_trembl,seqarray_clean_rnd_sprot)


######################################################################################################################################################
############################################# deletion of the used domains out of the whole dataset ############################################
######################################################################################################################################################

# create a list of labels for the positive dataset, negative domains and random protein sequences


seq_labels_positive = seqarray_clean[1:]
seq_labels_positive["categories"] = 0
# print(seq_labels_positive)


# print(seq_labels_positive)


# print(len(seqarray_clean))
# print(len(labels_negative_domains))

# combining all domain sequences into one array


seq_labels_negative_domains = pd.concat(
    (
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
    )
)


seq_labels_negative_domains = seq_labels_negative_domains[1:]
seq_labels_negative_domains["categories"] = 1
# print(seq_labels_negative_domains)


# print('\n',len(seqarray_all_domains))

# print(len(seqarraylen_clean_rnd_sprot))

# delete the used domains out of the whole dataset

# seqarray_clean_rnd_without_domains = np.setdiff1d(seqarray_clean_rnd, seqarray_all_domains)


seqarray_clean_rnd_sprot = seqarray_clean_rnd_sprot[1:]

seqarray_clean_rnd_sprot["categories"] = 2

# print(seqarray_clean_rnd_sprot)


seq_labels_all_domains = pd.concat([seq_labels_positive, seq_labels_negative_domains])
# print(seq_labels_all_domains)


# print(seqarray_clean_rnd_sprot)


# print(seqarray_clean_rnd_sprot[seqarray_clean_rnd_sprot["categories"]==0])

seqarray_clean_rnd_without_double_domains = seqarray_clean_rnd_sprot[
    ~seqarray_clean_rnd_sprot["Sequences"].isin(seq_labels_all_domains["Sequences"])
]


seqarray_full = pd.concat(
    [seqarray_clean_rnd_without_double_domains, seq_labels_all_domains]
)

# print(seqarray_clean_rnd_sprot[seqarray_clean_rnd_sprot["categories"]==0])


# print(seqarray_full[seqarray_full["categories"] == 0])


print("full df:", seqarray_full)






# ratio control




######################################################################################################################################################
################################################### Creating the slinding window approach ############################################################
########################################## using the target dimension of the positive dataset ########################################################
######################################################################################################################################################

# print(dimension_positive)

examplearray = np.array(
    [
        "1",
        "12",
        "123",
        "1234",
        "12345",
        "123456",
        "1234567",
        "12345678",
        "123456789",
        "1234567890",
        "12345678901",
        "123456789012",
    ]
)



examplearray = pd.DataFrame(
    [
        "1",
        "12",
        "123",
        "1234",
        "12345",
        "123456",
        "1234567",
        "12345678",
        "123456789",
        "1234567890",
        "12345678901",
        "123456789012",
    ]
)


examplearray.columns=['Sequences']


examplearray=pd.concat([examplearray,examplearray])
examplearray["categories"] = 0




exampledimension = 5


print("seqarray full:", (seqarray_full))
print("examplearray full:", (examplearray))
# print("seqarray subset:", seqarray_full[3])
# print("examplearray subset:", examplearray[3], "\n")
print("\n")

# for seq in examplearray['Sequences']:
#     print(seq)





def sliding_window_test(seq, dimension, stepsize=1):
    seqarray_sliding = []
    if len(seq) > dimension:
        seq_slice = [
            seq[i : i + dimension]
            for i in range(0, len(seq), stepsize)
            if len(seq[i : i + dimension]) == dimension
        ]
        if len(seq) % dimension != 0:
            # if seq[-dimension:] not in seq_slice:
            seq_slice.append(seq[-dimension:])

        seqarray_sliding.append(seq_slice)
    else:
        seqarray_sliding.append(seq)
        # if len(seqarray_sliding)== 1:
        #     print('wrong')
        #     print(seqarray_sliding)
        #     print(len(seqarray_sliding[0][0]))
        #     # break
        # else:
        #     # i+=1
        #     # print(i)
        #     continue
    return seqarray_sliding


# # sliding=sliding_window_test(seqarray_full,dimension_positive,(dimension_positive-10))


def test(seqarray_full):
    seqarray_window = []
    i = 0
    for sequence in seqarray_full['Sequences']:
        # print(sequence,'\n')
        sliding = sliding_window_test(
            sequence, dimension_positive, (dimension_positive - 10)
        )
        seqarray_window.append(sliding)
        if len(sliding) > 1:
            i += 1
            print(sliding)
            print(i)
            break
        else:
            # print(sliding)
            continue

    # print(sequence)
    # print(sliding)

    return len(seqarray_window)



testarray=test(seqarray_full)




# sliding window problem: first entry is empty
def sliding_window(seqarray, dimension, stepsize=1):
    seqarray_sliding = []
    for seq in seqarray['Sequences']:
        seq_slice = []
        if len(seq) > dimension:
            seq_slice = [
                seq[i : i + dimension]
                for i in range(0, len(seq), stepsize)
                if len(seq[i : i + dimension]) == dimension
            ]
            if len(seq) % dimension != 0:
                if seq[-dimension:] not in seq_slice:
                    seq_slice.append(seq[-dimension:])
            seqarray_sliding.append(seq_slice)
        else:
            seqarray_sliding.append([seq])
    return seqarray_sliding






example_sliding = sliding_window(examplearray, exampledimension, (exampledimension - 2))


sliding = sliding_window(seqarray_full, dimension_positive, (dimension_positive - 10))




# print('exapmple sliding', example_sliding,'\n')



print('Len Sliding', len(sliding))
print('Len seqall',(seqarray_full.shape))

sliding=pd.Series(sliding)

def multiplier (seqarray_full,sliding):
    sliding_df=pd.DataFrame(columns=['Sequences','categories'])
    category_index=0
    for nested_list in sliding:
        nested_list=pd.DataFrame(nested_list,columns=['Sequences'])

        # print(nested_list)
        
        categories = seqarray_full.iloc[category_index]['categories']
        # print('cat',categories)

        nested_list["categories"] = categories


        # print(nested_list)

        category_index+=1

        temp_df = pd.DataFrame({'Sequences': nested_list['Sequences'], 'categories': nested_list['categories']})
        sliding_df = pd.concat([sliding_df, temp_df], ignore_index=True)

        if category_index % 1000 == 0:
            print('iteration:',category_index)

            
    print(sliding_df)
    return sliding_df
    



# multiplier(examplearray,example_sliding)

seqarray_final=multiplier(seqarray_full,sliding)


print('Final array:',seqarray_final)

seqarray_final.to_csv('datatest2.csv', index=False) 



# ######################################################################################################################################################
# ################################################### Saving the final datasets ############################################################################
# ######################################################################################################################################################


