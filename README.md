# Constrained pseudo-time ordering for clinical transcriptomics data 
The pseudo ordering algorithm orders bulk-RNASeq samples based on gene expression and clinical information. It uses the chronology of sample collection within a patient for the ordering, thereby obtaining the progression of biological mechanisms with respect to time.
Polynomials are used to represent expression over the duration of the study and an EM algorithm to determine parameters and locations for samples along the gene curves. It works best for chronic diseases such as Asthma, Psoriasis, Ulcerative Colitis and vaccine treatments.

The implemetation consists of the atopic dermatitis dataset () along with clinical scores.

## Running the code
### Setup:
pip install -r requirements.txt

### Notebook
notebooks/EM.ipynb has all the code to run the curve fitting method and visualize the pseudo ordering. The method can be run using gene expression and clinical data or only using gene expression.

## Input Data:
### Gene Expression File
Format: Tab delimited data, with the first column containing "Geneid" and subsequent columns should include the normalized gene expression values. The column names should be of the format <patient_id>_<visit_number>. It is recommended that the samples are sorted on their visit in ascending order/starting with the first visit. Sample dataset - input_output/dataset_forPseudoOrdering.tsv is from study GSE193309 - https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193309. Sample filtering is described in the Supplementary file.

### Sample Naming Convention
The samples are named with the patient ID preceeing the visit information delimited by an underscore. For example, sample for visit 1 for patient XYZ must be XYZ_visit1. The patient ID may contain underscore (X_YZ) in which case the location of the delimiter must be specified in the patient_index function. The visit information must not contain an underscore and must be sequentially ordered such as vis0, vis1, vis2, vis4, vis8.

### Clinical Scores
Format - Tab delimited, with first column labeled as 'smp_name' and should contain the sample names.Sample clinical scores are in gse193309_clinicaldata.tsv.
These should match the columns of the gene expression file that contain sample data. The second column should be named 'clinical_score' and should contain the clinical scores of the sample.
This is an optional file. The algorithm can be run exclusively using gene expression data.

## Output
2 columns containing pseudo vector and sample names

# NOTES
1. Pseudo ordering method does not perform normalization or batch correction, so it is recommended that all pre-processing of data be done before using the algorithm.
2. The algorithm is stochastic in nature and the result dependes on the initialization. It is recommended the number of initializations is set to between 15-30 and iterations between 20-30. Depending on the diagnostic plots, they can be reduced.


## Copyright Notice
Copyright Notice: Permission is hereby granted, free of charge, for academic research purpose only and for non-commercial use only, to any person from academic research or non-profit organization obtaining a copy of this software and associated documentation files (the "Software"), to use, copy, modify, or merge the Software, subject to the following conditions: this permission notice shall be included in all copies or substantial portions of the Software. All other rights are reserved. The Software is provided 'as is", without warranty of any kind, express or implied, including the warranties of noninfringement.
