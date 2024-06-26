import hail as hl
import random
from sklearn.model_selection import train_test_split


class SNPSelector:
    def __init__(self, mt, predictor, group_name):
        self.mt = mt
        self.predictor = predictor
        self.group_name = group_name

    def bootstrap_sample(self, mt, n_samples):
        sample_ids = mt.cols().key_by().select('s').collect()

        def sample(mt):
            selected_sample_ids = random.choices(sample_ids, k=mt.count_cols())
            selected_sample_set = set(selected_sample_ids)
            excluded_sample_ids = [sid for sid in sample_ids if sid not in selected_sample_set]
            
            batch = mt.filter_cols(hl.literal(selected_sample_ids).contains(mt.col_key))
            excluded = mt.filter_cols(hl.literal(excluded_sample_ids).contains(mt.col_key))
            return batch, excluded

        bootstrap_samples = [sample(mt) for _ in range(n_samples)]
        return bootstrap_samples

    def calculate_error_rate(self, mt, predictor):
        mt = predictor.predict(mt)
        error_rate = mt.aggregate_cols(hl.agg.mean(mt.predicted_ancestry != mt.pop))
        return error_rate

    def calculate_632_plus_error_rate(self, mt, n_bootstrap_samples, snps):
        original_error_rate = self.calculate_error_rate(mt, self.predictor().fit(mt, self.group_name, snps))
        bootstrap_samples = self.bootstrap_sample(mt, n_bootstrap_samples)
        bootstrap_error_rates = [self.calculate_error_rate(excluded_bs, self.predictor().fit(bs, self.group_name, snps)) 
                                for bs, excluded_bs in bootstrap_samples]

        loo_error_rate = sum(bootstrap_error_rates) / n_bootstrap_samples
        bootstrap_632_plus_error_rate = 0.368 * original_error_rate + 0.632 * loo_error_rate
        return bootstrap_632_plus_error_rate

    def split_data(self, num_splits=50):
        snps = list(self.mt.aggregate_rows(hl.agg.collect_as_set(self.mt.rsid)))
        random.shuffle(snps)
        batch_size = len(snps) // num_splits
        return [snps[i:i + batch_size] for i in range(0, len(snps), batch_size)]

    def greedy_select_snps(self, mt, snps, top_n=40, n_bootstrap_samples=10):
        selected_snps = []
        remaining_snps = list(snps)

        for _ in range(top_n):
            best_snp = None
            best_error_rate = float('inf')
            print(f'snps={selected_snps}')

            for snp in remaining_snps:
                current_snps = selected_snps + [snp]
                error_rate = self.calculate_632_plus_error_rate(mt, n_bootstrap_samples=n_bootstrap_samples,
                                                                snps=current_snps)
                if error_rate < best_error_rate:
                    best_error_rate = error_rate
                    best_snp = snp

            selected_snps.append(best_snp)
            remaining_snps.remove(best_snp)

        return selected_snps

    def select_snps(self, n_bootstrap_samples=10, num_splits=50, top_n=40):
        snp_splits = self.split_data(num_splits)
        avg_error_rate = 0
        result_snps = []

        for i, snps in enumerate(snp_splits):
            col_ids = self.mt.cols().key_by().select('s').collect()
            train_ids, test_ids = train_test_split(col_ids, test_size=0.2, random_state=42)

            mt_train = self.mt.filter_cols(hl.literal(train_ids).contains(self.mt.col_key))
            mt_test = self.mt.filter_cols(hl.literal(test_ids).contains(self.mt.col_key))

            selected_snps = self.greedy_select_snps(mt_train, snps, top_n,
                                                    n_bootstrap_samples=n_bootstrap_samples)
            error_rate = self.calculate_error_rate(mt_test, 
                                                   self.predictor().fit(mt_test,
                                                                        self.group_name, 
                                                                        snps))
            print(f'True error rate on set {i}: {error_rate}')
            avg_error_rate += error_rate
            result_snps.extend(selected_snps)

        avg_error_rate /= num_splits
        return result_snps, avg_error_rate
