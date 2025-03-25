import Dataset_preprocess_v2
import DNN_pipeline

#################################################################################################

if __name__ == "__main__":
    fasta = Dataset_preprocess_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00177.fa"
    )
    seqarray_clean, seqarraylen_clean, normaltest = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    dimension_positive = fasta.dimension_finder(seqarraylen_clean)
    # print("targeted dimension", dimension_positive)

    # negative Domains:
    fasta = Dataset_preprocess_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00079.fa"
    )
    seqarray_clean_PF00079, seqarraylen_clean_PF00079, normaltest_PF00079 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = Dataset_preprocess_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00080.fa"
    )
    seqarray_clean_PF00080, seqarraylen_clean_PF00080, normaltest_PF00080 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = Dataset_preprocess_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00118.fa"
    )
    seqarray_clean_PF00118, seqarraylen_clean_PF00118, normaltest_PF00118 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = Dataset_preprocess_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00162.fa"
    )
    seqarray_clean_PF00162, seqarraylen_clean_PF00162, normaltest_PF00162 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )

    # load in swissprot and trembl
    fasta = Dataset_preprocess_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/alluniprot/sprot_domains.fa"
    )
    seqarray_clean_rnd_sprot = fasta._load_in_SwissProt()

    ################### Data creation ########################
    dataset = Dataset_preprocess_v2.databaseCreater(
        seqarray_clean,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        dimension_positive,
        0,
    )
    #################################################################################################

    run = DNN_pipeline.Starter(
        "/global/research/students/sapelt/Masters/MasterThesis/datatestSwissProt.csv"
    )

    # run = Starter("/global/research/students/sapelt/Masters/MasterThesis/datatest1.csv")

    run.tuner()
