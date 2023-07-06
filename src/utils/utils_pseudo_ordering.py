"""
Class consists of methods that performs pseudo ordering of samples
"""
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


class pseudo_ordering:
    """
    Consists of methods for building data structures and performing pseudo ordering
    """

    def __init__(self):
        # Empty
        pass

    def compute_ys(self, smp_order_df, df):
        """
        Computes a 2d array from the dataframe by ordering the samples
         according to Z from the input dataframe
        Input:
            - sample order dataframe
            - Input clusters (DataFrame)
        Output:
            - Ordered Gene expression 2d array
        """
        smp_order = list(smp_order_df["samples"])
        # print(smp_order)
        df = df.reindex(columns=smp_order)
        res = df.values
        return res

    def compute_new_z_withcorr(
        self,
        start,
        end,
        z_dash_vec,
        lkl_norm_vec,
        clinical_sd_norm_vec,
        corr_norm_vec,
        lkl_weight,
        clin_sd_weight,
        clin_cor_weight,
    ):
        """
        Given the upper and lower bounds for z_dash, the function computes the new value of Z for the sample
        Input:
          - Start Index
          - End Index
          - z_dash vector
          - likelihood vector
          - Clinical score vector
          - Likelihood weight
          - Clinical score weight
        Output:
          - Z-value
        """
        new_zval = -1
        df = pd.DataFrame(
            {
                "ind": list(range(len(clinical_sd_norm_vec))),
                "clinical_sd_norm": clinical_sd_norm_vec,
                "lkl_norm": lkl_norm_vec,
                "corr_norm": corr_norm_vec,
            }
        )
        if end - start > 1:
            df = df[(df["ind"] >= start) & (df["ind"] <= end + 1)]
        if len(df) > 1 and end - start > 1:
            df["lkl_sc"] = lkl_weight * df["lkl_norm"]
            df["sd_sc"] = clin_sd_weight * (1 - df["clinical_sd_norm"])
            df["cor_sc"] = clin_cor_weight * df["corr_norm"]
            df["score"] = df["lkl_sc"] + df["sd_sc"] + df["cor_sc"]
            max_index = df[df["score"] == df["score"].max()]["ind"].values[0]
            new_zval = z_dash_vec[max_index]

        return new_zval

    def compute_new_z_onlygex(self, start, end, z_dash_vec, lkl_vec):
        """
        Given the upper and lower bounds for z_dash, the function
        computes the new value of Z for the sample
        Input:
          - Start Index
          - End Index
          - z_dash vector
          - likelihood vector
        Output:
          - Z-value
        """

        new_zval = -1
        df = pd.DataFrame({"ind": list(range(len(lkl_vec))), "lkl": lkl_vec})
        if end - start > 1:
            df = df[(df["ind"] >= start) & (df["ind"] <= end + 1)]
        if len(df) > 1 and end - start > 1:
            max_index = df[df["lkl"] == df["lkl"].max()]["ind"].values[0]
            new_zval = z_dash_vec[max_index]

        return new_zval

    def z_dash_range(self, z_dash_vec, z_vec, ll, ul):
        """
        Extracts the range of z_dash values that lie within the lower and upper bounds of z_vector
        Input:
            - Z-dash vector
            - Z-vector
            - Lower Limit
            - Upper Limit
        Output:
          - Lower and Upper limits of Z-dash
        """

        try:
            start = np.where(z_dash_vec > z_vec[ll])[0][0]
            end = np.where(z_dash_vec < z_vec[ul])[0][-1]
        except Exception as e:
            start = 0
            end = 0
            # print("Error:", e)

        return start, end

    def extract_index_limits(self, sample_index, ante_index, con_index, z_vec):
        """
        For a given sample, extracts the upper and lower index between which it can be moved
        Input:
            - Index of the sample
            - Antecedent Dictionary
            - Consequent Dictionary
            - Z vector
        Output:
            - Lower limit
            - Upper Limit
        """
        ll = []
        ul = []
        if sample_index in ante_index.keys():
            ante = ante_index[sample_index]
            ll = max(ante)
        else:
            ll = 0

        if sample_index in con_index.keys():
            cons = con_index[sample_index]
            ul = min(cons)
        else:
            ul = len(z_vec) - 1

        return ll, ul

    # Normalize function
    def normalize_0to1(self, arr):
        """
        Normalizes values of an array between 0 to 1 - (Min-Max normalization)
        Input:
            - Array of numbers
        Output
            - Normalized Array
        """
        min_ele = min(arr)
        max_ele = max(arr)
        norm_vals = [(x - min_ele) / (max_ele - min_ele) for x in arr]
        return norm_vals

    def compute_clinical_stdvs(self, zs_clinical_vec, score):
        """
        Computes Clinical scores (Standard Deviation) with the score from the sample
        Input:
            -
        """
        zs_clinical_curr = [np.append(x, score) for x in zs_clinical_vec]
        zs_clinical_vals = [np.std(x) for x in zs_clinical_curr]
        return zs_clinical_vals

    def calculate_clinical_correlation(self, clinical_df):
        """
        calculates the absolute correlation between Z-value and act scores
        """

        clinical_df = clinical_df[["z", "clinical_score"]].dropna(axis=0)
        return abs(np.corrcoef(clinical_df["z"], clinical_df["clinical_score"])[0, 1])

    def compute_clinical_correlations(self, clinical_df, z_dash_vec, sample_index):
        """
        Computes the correlation for the sample index wrt z_dash
        """

        z_vec = np.array(clinical_df["z"])
        clinical_vec = np.array(clinical_df["clinical_score"])
        z_mat = np.tile(z_vec, (len(z_dash_vec), 1))

        z_mat[:, sample_index] = z_dash_vec

        return np.corrcoef(z_mat, clinical_vec)[:-1, -1]

    def compute_likelihood(self, y_vec, sample_index, z_mean, z_sd):
        """
        Computes the log-likelihood of a sample coming from each index in z_dash
        Input:
          - Y vector
          - Index of the sample
          - Mean of z_dash - {DataFrame}
          - SD of z_dash - {DataFrame}
        Output:
          - Likelihood of the sample from all values in z-dash
        """

        # Compute likelihood of the sample at each position
        # lkl_smp_mat = np.empty([z_mean.shape[0], z_mean.shape[1]]) # Likelihood of the datapoint coming from each curve
        lkl_smp = np.empty(z_mean.shape[1])
        for i in range(0, z_mean.shape[1]):
            lkl_vec = norm(z_mean[:, i], z_sd[:, i]).pdf(y_vec[:, sample_index])
            # Set any value greater than 1 to 1
            lkl_vec[lkl_vec > 1] = 1
            # Ignore very low probabilities, extract > 20 percentile values
            lkl_smp[i] = np.sum(np.log(lkl_vec[lkl_vec > np.percentile(lkl_vec, 20)]))

        return lkl_smp

    def sample_ancestor_decendents(self, ind, ante, desc):
        """
        Extracts antecedents and consequents of a sample and returns the indicies including of the sample
        Input:
            - IOndex of the sample
            - Dictionary of Antecedents
            - Dictionary of decendents
        Output:
            - list of samples indicies
        """
        smp_inds = []
        smp_inds.append(ind)
        if ind in ante.keys():
            smp_inds = smp_inds + ante[ind]
        if ind in desc.keys():
            smp_inds = smp_inds + desc[ind]

        return smp_inds

    def parameter_estimation(
        self, z_dash_vec, coefs, err_vec, clinical_df, sample_index_val=None
    ):
        """
        Estimates the mean and standard deviation of each z using the curve coefficients
        mean = f(zi), SD = Mean deviation of points

        Input:
          - Z positions dataframe
          - Vector of Z that was used for curve construction
          - Curves coefficient set
          - Errors or deviations of sample points from the curve(s)
          - Clinical scores dataframe
          - sample index value (optional)

        Output:
          - Mean estimates of Z positions
          - SD estimates of Z positions
          - Clinical scores
        """
        # Calculate Mean and SD of the positions
        zs_mean = np.empty([len(coefs), len(z_dash_vec)])
        # Mean
        for i in range(0, len(coefs)):
            zs_mean[i] = np.polyval(coefs[i], z_dash_vec)

        # SD
        sq_err = np.square(err_vec)
        sigma = np.sqrt(np.sum(sq_err, axis=1) / err_vec.shape[1])
        zs_sd = np.tile(sigma, (len(z_dash_vec), 1)).transpose()

        # ACT Scores
        # remove the ACT score of the sample
        clinical_sc_df = clinical_df.copy()
        if sample_index_val != None:
            print("sample_index_val", sample_index_val)
            index_sample = clinical_sc_df[
                clinical_sc_df["ind"] == sample_index_val
            ].index
            clinical_sc_df.drop(index_sample, inplace=True)

        zs_act = []
        for i in range(0, len(z_dash_vec)):  # For every location in z_dash
            # Indicies that are close to Z-dash index. Get values of the range - lower and upper bounds of search space from Z-dash
            l1 = z_dash_vec[i] - 0.05
            l2 = z_dash_vec[i] + 0.05
            clinical_scores = clinical_sc_df[
                (clinical_sc_df["z"] >= l1) & (clinical_sc_df["z"] <= l2)
            ]["clinical_score"].values
            if len(clinical_scores) == 0:
                zs_act.append(clinical_sc_df["clinical_score"].to_list())
            else:
                zs_act.append(clinical_scores)

        return zs_mean, zs_sd, zs_act

    def parameter_estimation_onlygex(self, z_dash_vec, coefs, err_vec):
        """
        Estimates the mean and standard deviation of each z using the curve coefficients
        mean = f(zi), SD = Mean deviation of points

        Input:
          - Z positions dataframe
          - Vector of Z that was used for curve construction
          - Curves coefficient set
          - Errors or deviations of sample points from the curve(s)

        Output:
          - Mean estimates of Z positions
          - SD estimates of Z positions
        """
        # Calculate Mean and SD of the positions
        zs_mean = np.empty([len(coefs), len(z_dash_vec)])
        # Mean
        for i in range(0, len(coefs)):
            zs_mean[i] = np.polyval(coefs[i], z_dash_vec)

        # SD
        sq_err = np.square(err_vec)
        sigma = np.sqrt(np.sum(sq_err, axis=1) / err_vec.shape[1])
        zs_sd = np.tile(sigma, (len(z_dash_vec), 1)).transpose()

        return zs_mean, zs_sd

    def compute_indexes(self, sample_to_index, ante_patient, con_patient):
        """
        Computes antedent, consequent indicies from ante_patient, con_patient and sample index
        Input:
            - Dictionaries relating to sample_to_index (changes every iteration), ante_patient, con_patient
        Output:
            - Dictionaries with the index - antecedent_index, consequent_index
        """

        # Antecedent index
        antecedent_index = {}
        for key, val in ante_patient.items():
            k = sample_to_index[key]
            # print(key,val)
            v = []
            for a in val:
                v.append(sample_to_index[a])
                # print(k,v)
            antecedent_index[k] = v

        # Consequent Index
        consequent_index = {}
        for key, val in con_patient.items():
            k = sample_to_index[key]
            v = []
            for a in val:
                v.append(sample_to_index[a])
            consequent_index[k] = v

        return antecedent_index, consequent_index

    def calculate_clinical_sd(self, clinical_df):
        """
        Calculates the standard deviation in ACT scores for the sample ordering
        """
        df = clinical_df.copy()
        df = df.sort_values(by=["ind"]).reset_index(drop=True)
        points = df["z"].copy()

        # Clustering 1-D data
        clusters = []
        eps = 0.04
        points_sorted = sorted(points)
        curr_point = points_sorted[0]
        curr_cluster = [curr_point]
        for point in points_sorted[1:]:
            if point <= curr_point + eps:
                curr_cluster.append(point)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)

        groups = []
        cnt = 0
        for i in range(0, len(clusters)):
            a = len(clusters[i])
            groups = groups + [cnt] * a
            cnt = cnt + 1

        df["groups"] = groups
        t_sd = df[["clinical_score", "groups"]].groupby("groups").std()
        t_c = df[["groups", "samples"]].groupby("groups").count()
        t = pd.merge(t_c, t_sd, on="groups")
        dfr = t.rename(columns={"clinical_score": "sd", "samples": "num_samples"})
        wa = self.weighted_average(
            dfr.dropna()["sd"].to_list(), dfr.dropna()["num_samples"].to_list()
        )

        return wa

    def rmse(self, a, b):
        """
        Calculates RMSE between 2 2d arrays
        """
        val = []
        for i in range(0, len(a)):
            t = np.sqrt(np.mean(np.square(a[i] - b[i])))
            val.append(t)
        return np.mean(val)

    def fit_polynomial(self, a, b, poly_deg=3):
        """
        Fits polynomial. Outputs the residuals at each index
        Input:
            - Z value {1d array}
            - Y values (2d array)
        Output:
            - Fitted values (2d array), Residuals (2d array)
        """
        # Fit polynomial
        coef = []
        for i in range(0, len(b)):
            p = np.polyfit(a, b[i], deg=poly_deg)
            p = p.round(3)
            coef.append(p)

        fitted_vals = np.empty([b.shape[0], b.shape[1]])
        err_vals = np.empty([b.shape[0], b.shape[1]])
        for i in range(0, len(coef)):
            fitted_vals[i] = np.polyval(coef[i], a)
            err_vals[i] = abs(fitted_vals[i] - b[i])

        return coef, fitted_vals, err_vals

    def initialize_z(self, s_ind, num_samp):
        """
        Initialize Z and construct the sample - index - z dataframe
        Input:
          - DataFrame containting the sample, index
          - Number of samples
        Output:
          - Dataframe that contains the latent vector
        """
        # Sorted Z to reflect index ordering
        # smp_to_index = dict(zip(s_ind['sample'], s_ind['index']))
        z = np.sort(np.random.uniform(0, 1, num_samp))
        sample_index_0 = pd.DataFrame(
            {"samples": s_ind["sample"], "ind": s_ind["index"], "z": z}
        )

        return sample_index_0

    def weighted_average(self, distribution, weights):
        """
        Computes the weighted average of the distribution given the weights
        """

        numerator = sum(
            [distribution[i] * weights[i] for i in range(len(distribution))]
        )
        denominator = sum(weights)
        if denominator != 0:
            return round(numerator / denominator, 2)
        else:
            return -1

    def patient_index(self, smps, patient_delimter_loc=2):
        """
        Contruct the Patient Index using initial ordering
        Input:
          - List of samples
          - patient_delimter_loc is the location of the delimiter ("_") that separates the patient ID and the sample name
        Output:
          - Dictionaries of Antecedents and Decendents of all samples
        """

        def extract_patid(txt):
            """
            Extract patient identifiers
            """
            a = txt.split("_")
            if patient_delimter_loc == 1:
                pat_id = a[0]
            else:
                pat_id = "_".join(a[:patient_delimter_loc])

            return pat_id

        patient_id = [extract_patid(x) for x in smps]
        ind0 = pd.DataFrame(
            {"sample": smps, "patient": patient_id, "index": list(range(0, len(smps)))}
        )

        # Sample dictionary
        ind_to_smp = dict(zip(ind0["index"], ind0["sample"]))

        # Index of patient samples
        pt_dict = self.patient_sample_indexes(ind0)

        # Antecedent index
        ante_ind_0 = self.build_antecedent_index(pt_dict)

        # Consequent Index
        con_ind_0 = self.build_consequent_index(pt_dict)

        # Sample Order Dictionary
        ante_pt = {}
        for key, val in ante_ind_0.items():
            key_value = ind_to_smp[key]
            val_list = []
            for item in val:
                val_list.append(ind_to_smp[item])
            ante_pt[key_value] = val_list

        con_pt = {}
        for key, val in con_ind_0.items():
            key_value = ind_to_smp[key]
            val_list = []
            for item in val:
                val_list.append(ind_to_smp[item])
            con_pt[key_value] = val_list

        return ind0, ante_pt, con_pt

    def build_consequent_index(self, pt_dict):
        """
        Builds the consequents of all samples - indexes of all post visit samples for a given sample of a patient
        Input:
            - Patient - Sample Index dictionary
        Output:
            - Index of all consequents (post visit samples) of a sample
        """
        conseq = {}
        for key, val in pt_dict.items():
            pat_sample = val.copy()
            for i in range(0, len(pat_sample) - 1):
                for j in range(i + 1, len(pat_sample)):
                    if pat_sample[i] in conseq.keys():
                        conseq[pat_sample[i]] = conseq[pat_sample[i]] + [pat_sample[j]]
                    else:
                        conseq[pat_sample[i]] = [pat_sample[j]]
        return conseq

    def build_antecedent_index(self, pt_dict):
        """
        Builds the antecedents of all samples - indexes of all prior visit samples for a given sample of a patient
        Input:
            - Patient - Sample Index dictionary
        Output:
            - Index of all antecedents (prior visit samples) of a sample
        """
        ante = {}
        for key, val in pt_dict.items():
            pat_sample = val.copy()
            pat_sample.sort(reverse=True)
            for i in range(0, len(pat_sample) - 1):
                for j in range(i + 1, len(pat_sample)):
                    if pat_sample[i] in ante.keys():
                        ante[pat_sample[i]] = ante[pat_sample[i]] + [pat_sample[j]]
                    else:
                        ante[pat_sample[i]] = [pat_sample[j]]
        return ante

    def patient_sample_indexes(self, pt_indx):
        """
        Constructs patient-sample index dictionary given the sample index
        Input:
            - Sample Index dictionary
        Output
            - Patient - Samples Index dictionary
        """
        # Patient Dictionary
        ptid = list(pt_indx["patient"].unique())
        pt_dict = {}
        for item in ptid:
            vals = list(pt_indx[pt_indx["patient"] == item]["index"])
            if len(vals) > 1:
                pt_dict[item] = vals
        return pt_dict

    def em_with_clinical_correlation(
        self, indexes, num_samples, df_gex, act, weights, num_init=50, num_iter=20
    ):
        """
        Extracts curves using gene expression and clinical information
        Input:
            - Indexes Dictionary
                - Original (Ordered) Index of Samples: {DataFrame}
                - Dictionary of ancestors of samples
                - Dictionary of decendents of samples
            - Number of samples
            - Gene Expression DataFrame
            - Dictionary of gene expression and clinical score weight
            - Clinical Scores DataFrame {Sample-ACT Score}
            - Number of initializations (for z)
            - Number of iterations
        Output:
            - rmse_err_min - Minimum RMSE error from each initialization
            - rmse_err_list - All RMSE errors from each iteration and initialization
            - clin_sd_scores_min - Minimum clinical score SD from each initialization
            - clin_sd_scores_list - All clinical score SDs from each iteration and initialization
            - z_list - All Z (ordering) from each iteration and initialization
            - smp_ord_list - Sample order lists obtained from z_list
            - coefs_list - Polynomials of orderings at each iteration
            - rmse_selected - Selected RMSEs where the combined RMSE and clinical scores were minimum
            - overall_order_score - Combined RMSE and clinical scores
        """

        # unpacking Dictionaries
        ind0 = indexes["index_init"]
        ante_pt = indexes["index_antecedent"]
        con_pt = indexes["index_consequent"]

        # Gex and Clinical weights
        gene_expression_weight = weights["gene_expression_weight"]
        clinical_sd_weight = weights["clinical_sd_weight"]
        clinical_corr_weight = weights["clinical_corr_weight"]

        # List initialization to keep track of scoring parameters
        rmse_err_list = []
        rmse_err_min = []
        clin_sd_scores_list = []
        clin_sd_scores_min = []
        z_list = []
        smp_ord_list = []
        coefs_list = []
        rmse_selected = []
        overall_order_score = []

        for _ in range(num_init):
            # Lists for storing values after each iteration
            rmse_err = []
            clin_sd_scores = []
            z_iter = []
            smp_ord_iter = []
            coefs_iter = []
            clin_corr = []
            # INITIALIZATION -STEP
            # Initialize Latent Vector
            z_dash = np.linspace(0.0, 1, 101)
            # Initialize domain
            sample_index_var = self.initialize_z(ind0, num_samples)
            # Sample Indexes
            smp_to_index = dict(
                zip(sample_index_var["samples"], sample_index_var["ind"])
            )
            ante_ind, con_ind = self.compute_indexes(smp_to_index, ante_pt, con_pt)
            y = self.compute_ys(sample_index_var, df_gex)

            z = sample_index_var["z"].to_list()
            # clinical scores
            act = pd.merge(
                act[["samples", "clinical_score"]],
                sample_index_var[["samples", "z", "ind"]],
                on="samples",
            )
            act = act.sort_values(by=["z"]).reset_index(drop=True)

            # Initial Curve Fit
            coefficients, fitted_x, errors = self.fit_polynomial(z, y)

            # E-STEP
            for _ in range(num_iter):
                y = self.compute_ys(sample_index_var, df_gex)
                # RMSE Calculation
                new_rmse = self.rmse(fitted_x, y)
                # Clinical Score Calculation
                clin_sd_score = self.calculate_clinical_sd(act)
                # Clinical Correlation
                new_clin_corr = self.calculate_clinical_correlation(act)
                if clin_sd_score == -1:
                    continue
                # Appending values after each iteration
                rmse_err.append(new_rmse)
                clin_sd_scores.append(clin_sd_score)
                clin_corr.append(new_clin_corr)
                z_iter.append(z)
                smp_ord_iter.append(sample_index_var["samples"].to_list())
                coefs_iter.append(coefficients)
                # Calculate Z_dash Parameters ----- Mean and SD of the positions
                zs_mn, zs_sdv, zs_act = self.parameter_estimation(
                    z_dash, coefficients, errors, act
                )

                sample_list = list(df_gex.columns)
                while len(sample_list) > 0:
                    # Select Sample
                    sample = sample_list.pop(random.randrange(len(sample_list)))
                    smp_index = sample_index_var.loc[
                        sample_index_var["samples"] == sample, "ind"
                    ].values[0]
                    # print("sample, ind_select:",sample, ind_select)

                    lkl = self.compute_likelihood(y, smp_index, zs_mn, zs_sdv)

                    sc = act[act["ind"] == smp_index]["clinical_score"].values[0]
                    clinical_sd_scores = self.compute_clinical_stdvs(zs_act, sc)
                    clinical_corrs = self.compute_clinical_correlations(
                        act, z_dash, smp_index
                    )
                    lkl_norm = self.normalize_0to1(lkl)
                    clinical_sd_norm = self.normalize_0to1(clinical_sd_scores)
                    corrs_norm = self.normalize_0to1(abs(clinical_corrs))
                    lower_lim, upper_lim = self.extract_index_limits(
                        smp_index, ante_ind, con_ind, z
                    )
                    # print(lower_lim, upper_lim)
                    st, en = self.z_dash_range(z_dash, z, lower_lim, upper_lim)
                    # print("Sample:{}, z_Lower_lim:{}, z_Upper_lim:{}, z_dash_ll:{}, z_dash_ul:{}".format(sample, lower_lim, upper_lim, st, en))

                    new_z = self.compute_new_z_withcorr(
                        st,
                        en,
                        z_dash,
                        lkl_norm,
                        clinical_sd_norm,
                        corrs_norm,
                        lkl_weight=gene_expression_weight,
                        clin_sd_weight=clinical_sd_weight,
                        clin_cor_weight=clinical_corr_weight,
                    )
                    if new_z > -1:
                        # Assign New Z value
                        sample_index_var.loc[
                            sample_index_var["samples"] == sample, "z"
                        ] = new_z
                        # Re-construct sample index values with changed z
                        sample_index_var = sample_index_var.sort_values(
                            by=["z"]
                        ).reset_index(drop=True)
                        sample_index_var["ind"] = list(range(0, len(sample_index_var)))
                        # Reconstruct Indexes
                        smp_to_index = dict(
                            zip(sample_index_var["samples"], sample_index_var["ind"])
                        )
                        ante_ind, con_ind = self.compute_indexes(
                            smp_to_index, ante_pt, con_pt
                        )
                        # New Z
                        z = sample_index_var["z"].to_list()
                        # Re-order Ys
                        y = self.compute_ys(sample_index_var, df_gex)
                        # print("Order:\n",sample_index_var['samples'].to_list())
                        # Rearrange ACT DF
                        act = pd.merge(
                            act[["samples", "clinical_score"]],
                            sample_index_var[["samples", "z", "ind"]],
                            on="samples",
                        )
                        act = act.sort_values(by=["z"]).reset_index(drop=True)

                # End of While Loop

                # M-STEP
                y = self.compute_ys(sample_index_var, df_gex)
                z = sample_index_var["z"].to_list()
                # Curve Fit
                coefficients, fitted_x, errors = self.fit_polynomial(z, y)

            # End of Iteration loop

            # Getting only the best result wrt RMSE
            min_index_rmse = rmse_err.index(min(rmse_err))
            # Record values wrt minimum RMSE
            rmse_err_min.append(rmse_err[min_index_rmse])
            rmse_err_list.append(rmse_err)

            z_list.append(z_iter[min_index_rmse])
            smp_ord_list.append(smp_ord_iter[min_index_rmse])
            coefs_list.append(coefs_iter[min_index_rmse])
            # rmse_selected.append(rmse_err[max_sc_ind])
            # overall_order_score.append(sc)

        return (
            rmse_err_min,
            rmse_err_list,
            z_list,
            smp_ord_list,
            coefs_list,
        )

    def em_onlygex(self, indexes, num_samples, df_gex, num_init=20, num_iter=20):
        """
        Extracts curves using gene expression and clinical information
        Input:
            - Original (Ordered) Index of Samples: {DataFrame}
            - Dictionary of ancestors of samples
            - Dictionary of decendents of samples
            - Number of samples
            - Gene Expression DataFrame
        Output:
            - rmse_err_min
            - rmse_err_list
            - z_list
            - smp_ord_list
            - coefs_list
        """

        # unpacking Dictionaries
        ind0 = indexes["index_init"]
        ante_pt = indexes["index_antecedent"]
        con_pt = indexes["index_consequent"]

        # List initialization to keep track of scoring parameters
        rmse_err_list = []
        rmse_err_min = []
        z_list = []
        smp_ord_list = []
        coefs_list = []

        for j in range(0, num_init):
            # print("*************************************\n   Initialization:{} \n*********************************".format(j))
            # Storing values after each iteration
            rmse_err = []
            z_iter = []
            smp_ord_iter = []
            coefs_iter = []

            # INITIALIZATION

            # Initialize Latent Vector
            z_dash = np.linspace(0.0, 1, 101)
            # Initialization ----- Z
            sample_index_var = self.initialize_z(ind0, num_samples)
            smp_to_index = dict(
                zip(sample_index_var["samples"], sample_index_var["ind"])
            )
            ante_ind, con_ind = self.compute_indexes(smp_to_index, ante_pt, con_pt)
            y = self.compute_ys(sample_index_var, df_gex)
            z = sample_index_var["z"].to_list()
            # Initial curve fit
            coefficients, fitted_x, errors = self.fit_polynomial(z, y)

            # E-STEP
            for _ in range(num_iter):
                # RMSE Calculation
                new_rmse = self.rmse(fitted_x, y)
                rmse_err.append(new_rmse)
                z_iter.append(z)
                smp_ord_iter.append(sample_index_var["samples"].to_list())
                coefs_iter.append(coefficients)
                # Calculate Z_dash Parameters ----- Mean and SD of the positions
                zs_mn, zs_sdv = self.parameter_estimation_onlygex(
                    z_dash, coefficients, errors
                )
                sample_list = list(df_gex.columns)

                while len(sample_list) > 0:
                    # Select Sample
                    sample = sample_list.pop(0)

                    # extract antecedents and decendents
                    smp_index = sample_index_var[sample_index_var["samples"] == sample][
                        "ind"
                    ].values[0]
                    lkl = self.compute_likelihood(y, smp_index, zs_mn, zs_sdv)
                    lower_lim, upper_lim = self.extract_index_limits(
                        smp_index, ante_ind, con_ind, z
                    )
                    st, en = self.z_dash_range(z_dash, z, lower_lim, upper_lim)
                    new_z = self.compute_new_z_onlygex(st, en, z_dash, lkl)
                    if new_z > -1:
                        # Assign New Z value
                        sample_index_var.loc[
                            sample_index_var["samples"] == sample, "z"
                        ] = new_z
                        # Re-construct sample index values with changed z
                        sample_index_var = sample_index_var.sort_values(
                            by=["z"]
                        ).reset_index(drop=True)
                        sample_index_var["ind"] = list(range(0, len(sample_index_var)))
                        # Reconstruct Indexes
                        smp_to_index = dict(
                            zip(sample_index_var["samples"], sample_index_var["ind"])
                        )
                        ante_ind, con_ind = self.compute_indexes(
                            smp_to_index, ante_pt, con_pt
                        )
                        # New Z
                        z = sample_index_var["z"].to_list()
                        # Re-order Ys
                        y = self.compute_ys(sample_index_var, df_gex)
                # End of While (Iteration)
                # M-STEP
                y = self.compute_ys(sample_index_var, df_gex)
                z = sample_index_var["z"].to_list()
                coefficients, fitted_x, errors = self.fit_polynomial(z, y)

            # Getting only the best result wrt RMSE
            min_index_rmse = rmse_err.index(min(rmse_err))
            print("Initialization:{}, minimum_index={}".format(j, min_index_rmse))
            # Record values wrt minimum RMSE
            rmse_err_min.append(rmse_err[min_index_rmse])
            rmse_err_list.append(rmse_err)

            # Extracting Z values and sample orderings wrt minimum clinical score index
            z_list.append(z_iter[min_index_rmse])
            smp_ord_list.append(smp_ord_iter[min_index_rmse])
            coefs_list.append(coefs_iter[min_index_rmse])

        return rmse_err_min, rmse_err_list, z_list, smp_ord_list, coefs_list

    def expression_curves(self, ind, z_vec, df_fit, df_res):
        """
        Plot Expression Curves
        Input:
            - z-vector
            - DataFrame for fitted values
            - DataFrame for expression values
        """

        fig, axs = plt.subplots(1, len(ind), figsize=(30, 4))
        axs = axs.ravel()
        for i in range(len(ind)):
            axs[i].plot(z_vec, df_fit[i], color="red")
            axs[i].scatter(z_vec, df_res[i], marker="o")
            axs[i].grid(True)
            axs[i].set_title("gene_" + str(ind[i]))

    def sample_placement(self, z_vec):
        """
        Plots the sample along the Latent Z vector
        Input:
            - Z vector
        """

        plt.plot(np.linspace(0, len(z_vec) - 1, len(z_vec)), z_vec)
        plt.scatter(np.linspace(0, len(z_vec) - 1, len(z_vec)), z_vec, color="red")
        plt.xlabel("Position")
        plt.ylabel("Z Value")
        plt.title("Sample Placement Along Latent Space")
