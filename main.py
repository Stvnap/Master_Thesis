import Dataset_preprocess_TRAIN_v2
import Dataset_preprocess_EVAL_v2
import DNN_pipeline
import Testrunner
import Predictor

#################################################################################################

if __name__ == "__main__":

#################################################################################################

    ### TRAINING SET CREATION ###

    fasta = Dataset_preprocess_TRAIN_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00177.fa"
    )
    seqarray_clean, seqarraylen_clean, normaltest = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    dimension_positive = fasta.dimension_finder(seqarraylen_clean)
    # print("targeted dimension", dimension_positive)

    # negative Domains:
    fasta = Dataset_preprocess_TRAIN_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00079.fa"
    )
    seqarray_clean_PF00079, seqarraylen_clean_PF00079, normaltest_PF00079 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = Dataset_preprocess_TRAIN_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00080.fa"
    )
    seqarray_clean_PF00080, seqarraylen_clean_PF00080, normaltest_PF00080 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = Dataset_preprocess_TRAIN_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00118.fa"
    )
    seqarray_clean_PF00118, seqarraylen_clean_PF00118, normaltest_PF00118 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    fasta = Dataset_preprocess_TRAIN_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/domains_PF00162.fa"
    )
    seqarray_clean_PF00162, seqarraylen_clean_PF00162, normaltest_PF00162 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )

    # load in swissprot and trembl
    fasta = Dataset_preprocess_TRAIN_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/alluniprot/sprot_domains.fa"
    )
    seqarray_clean_rnd_sprot = fasta._load_in_SwissProt()

    ################### Data creation ########################
    dataset = Dataset_preprocess_TRAIN_v2.databaseCreater(
        seqarray_clean,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        dimension_positive,
        10,
    )
    #################################################################################################
    
    ### EVAL SET CREATION ###

    # positive Domain PF00177
    print("Loading positive domain PF00177")
    fasta = Dataset_preprocess_EVAL_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00177.fasta"
    )
    seqarray_clean, boundaries_allPF00177 = fasta.distribution_finder_and_cleaner(
        fasta.len_finder()
    )
    dimension_positive = fasta.dimension_finder(fasta.len_finder())
    print("targeted dimension", dimension_positive)

    # negative Domains:
    print("Loading negative PF00079")
    fasta = Dataset_preprocess_EVAL_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00079.fasta"
    )
    seqarray_clean_PF00079, boundaries_allPF00079 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    print("Loading negative PF00080")
    fasta = Dataset_preprocess_EVAL_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00080.fasta"
    )
    seqarray_clean_PF00080, boundaries_allPF00080 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    print("Loading negative PF00118")
    fasta = Dataset_preprocess_EVAL_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00118.fasta"
    )
    seqarray_clean_PF00118, boundaries_allPF00118 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )
    print("Loading negative PF00162")
    fasta = Dataset_preprocess_EVAL_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/rawPF00162.fasta"
    )
    seqarray_clean_PF00162, boundaries_allPF00162 = (
        fasta.distribution_finder_and_cleaner(fasta.len_finder())
    )

    # load in swissprot and trembl

    fasta = Dataset_preprocess_EVAL_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/rawuniprot_sprot.fasta"
    )
    seqarray_clean_rnd_sprot = fasta._load_in_SwissProt()

    print("Loading trembl")
    fasta = Dataset_preprocess_EVAL_v2.DomainProcessing(
        "/global/research/students/sapelt/Masters/rawuniprot_trembl.fasta"
    )
    seqarray_clean_rnd_trembl = fasta._load_in_Trembl()

    boundaries_all = [
        boundaries_allPF00177,
        boundaries_allPF00079,
        boundaries_allPF00080,
        boundaries_allPF00118,
        boundaries_allPF00162,
    ]
    boundaries_all = [item for sublist in boundaries_all for item in sublist]

    ################### Data creation ########################
    print("Starting data creation for SwissProt validation set")
    dataset = Dataset_preprocess_EVAL_v2.databaseCreater(
        seqarray_clean,
        seqarray_clean_PF00079,
        seqarray_clean_PF00080,
        seqarray_clean_PF00118,
        seqarray_clean_PF00162,
        seqarray_clean_rnd_sprot,
        seqarray_clean_rnd_trembl,
        148,  # HARDCODED change if dimension of positive domain changes
        0,
        boundaries_all,
    )
    
    print("All done creating evaluation dataset with full sequences")
    
    
    #################################################################################################

    ### HYPERPARAMETER TUNING ###

    run = DNN_pipeline.Starter(
        "/global/research/students/sapelt/Masters/MasterThesis/datatestSwissProt.csv"
    )

    # run = Starter("/global/research/students/sapelt/Masters/MasterThesis/datatest1.csv")

    run.tuner()


    ##################################################################################################

    ### FINAL TRAINING ###

    Testrun = Testrunner.Testrunning(
        "/global/research/students/sapelt/Masters/MasterThesis/logshp/test1_palma/trial.json",
        "/global/research/students/sapelt/Masters/MasterThesis/logshp/test1_palma/checkpoint.weights.h5",
    )

    Testrun.trainer()

    ##################################################################################################

    ### EVALUATION ON FULL SEQUENCES ###

    Predictor.predicting("./models/my_modelnewlabeling.keras", "./DataEvalSwissProt.csv")

