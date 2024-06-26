import hail as hl
import pandas as pd
import random


def non_nan(x, defailt=0.0):
    return hl.if_else(hl.is_finite(x), x, defailt)


def log_sum_exp(log_values):
    max_log = hl.max(log_values)
    return max_log + hl.log(hl.sum([hl.exp(log_value - max_log) for log_value in log_values]))

 
def log_likelihood(genotype, freqs, pop):
    return (
        hl.case()
        .when(genotype.is_hom_ref(), non_nan(hl.log(freqs[pop].p_AA)))
        .when(genotype.is_het(), non_nan(hl.log(freqs[pop].p_AB)))
        .when(genotype.is_hom_var(), non_nan(hl.log(freqs[pop].p_BB)))
        .default(0.0)
    )


class AncestryPredictor:    
    def __init__(self) -> None:
        self.is_fitted = False

    def gt_freq_estimates(self, mt):
        count_AA = hl.agg.sum(mt.GT.is_hom_ref())
        count_AB = hl.agg.sum(mt.GT.is_het())
        total_count = hl.agg.count_where(hl.is_defined(mt.GT))
        p_ML = (2 * count_AA + count_AB) / (2 * total_count)
        
        return mt.aggregate_entries(
            hl.agg.group_by(
                mt.rsid,
                hl.agg.group_by(
                    mt[self.group_name],
                    hl.struct(
                        p_ML = p_ML,
                        p_AA = p_ML ** 2,
                        p_AB = 2 * p_ML * (1 - p_ML),
                        p_BB = (1 - p_ML) ** 2,
                    )
                )
            )
        )
    
    def freqs_estimates(self, mt):

        def create_freqs_dataframe(snp_freqs):
            rows = [{'rsid': rsid, 'freqs': hl.struct(**freqs)} 
                    for rsid, freqs in snp_freqs.items()]
            return pd.DataFrame(rows)
        
        snp_freqs = self.gt_freq_estimates(mt)
        freqs_ht = hl.Table.from_pandas(create_freqs_dataframe(snp_freqs), key='rsid')
        return freqs_ht

    def annotate_mt(self, mt):
        return mt.annotate_rows(
            freqs=self.freqs_ht[mt.rsid].freqs
        )
    
    def log_likelihoods(self, mt):
        return mt.annotate_cols(
            log_likelihoods=hl.struct(**{
                pop: hl.agg.sum(log_likelihood(mt.GT, mt.freqs, pop)) + hl.log(self.proportions[pop])
                for pop in self.proportions
            })
        )

    def calculate_posteriors(self, mt):
        log_likelihoods = [mt.log_likelihoods[pop] for pop in self.proportions]

        mt = mt.annotate_cols(log_likelihood_sum=log_sum_exp(log_likelihoods))
        return mt.annotate_cols(
            posteriors=hl.dict({
                pop: mt.log_likelihoods[pop] - mt.log_likelihood_sum
                for pop in self.proportions
            })
        )
        
    def fit(self, mt, group_name, snps):
        self.group_name = group_name
        self.snps = snps

        self.total_subjects = mt.count_cols()
        ethnicity_counts = mt.aggregate_cols(hl.agg.counter(mt.pop))
        proportions = {k: v / self.total_subjects for k, v in ethnicity_counts.items()}
        self.proportions = proportions

        mt = mt.filter_rows(hl.literal(self.snps).contains(mt.rsid))
        self.freqs_ht = self.freqs_estimates(mt)
        self.is_fitted = True
        return self
    
    def predict(self, mt):
        if not self.is_fitted:
            raise ValueError('Predictor is not fitted yet!')
        
        mt_likelihood = self.log_likelihoods(self.annotate_mt(mt))
        mt_probs = self.calculate_posteriors(mt_likelihood)

        return mt_probs.annotate_cols(predicted_ancestry=hl.bind(
            lambda x: hl.sorted(x.items(), key=lambda item: item[1], reverse=True)[0][0],
            mt_probs.posteriors
        ))