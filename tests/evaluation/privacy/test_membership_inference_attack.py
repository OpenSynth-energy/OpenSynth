import torch

from src.opensynth.evaluation.privacy import membership_inference_attack as mia


def mia_sample_fixture() -> mia.MembershipInferenceAttackSamples:
    index = 10
    synthetic_sample = torch.tensor([i for i in range(index)])
    train_sample = torch.tensor([i + index for i in range(index)])
    holdout_sample = torch.tensor([i + 2 * index for i in range(index)])
    outlier_seen_sample = torch.tensor([i + 3 * index for i in range(index)])
    outlier_unseen_diff_sample = torch.tensor(
        [i + 4 * index for i in range(index)]
    )
    outlier_unseen_same_sample = torch.tensor(
        [i + 5 * index for i in range(index)]
    )

    return mia.MembershipInferenceAttackSamples(
        synthetic_samples=synthetic_sample,
        train_samples=train_sample,
        holdout_samples=holdout_sample,
        outlier_seen_samples=outlier_seen_sample,
        outlier_unseen_diff_samples=outlier_unseen_diff_sample,
        outlier_unseen_same_samples=outlier_unseen_same_sample,
    )


class TestCreateMIAAttackDataset:

    mia_samples = mia_sample_fixture()
    df_train = mia._create_mia_train_dataset(mia_sample_fixture())
    df_attack = mia._create_mia_attack_dataset(mia_sample_fixture())

    def test_train_positive_labels(self):
        got_list = self.df_train.query("target==1")["tensors"].tolist()
        got_list.sort()
        # Positive samples for MIA training set are synthetic samples
        assert got_list == self.mia_samples.synthetic_samples.tolist()

    def test_train_negative_labels(self):
        got_list = self.df_train.query("target==0")["tensors"].tolist()
        got_list.sort()
        # Negative samples for MIA training set are holdout samples
        assert got_list == self.mia_samples.holdout_samples.tolist()

    def test_seen_outliers(self):
        seen_outliers = self.df_attack.query("type=='seen'")
        tensors = seen_outliers["tensors"].tolist()
        tensors.sort()
        assert tensors == self.mia_samples.outlier_seen_samples.tolist()
        # Seen outliers have labels == 1
        labels = seen_outliers["target"].tolist()
        assert labels == [1] * len(labels)

    def test_unseen_same_outliers(self):
        unseen_same_outliers = self.df_attack.query("type=='unseen_same'")
        tensors = unseen_same_outliers["tensors"].tolist()
        tensors.sort()
        assert tensors == self.mia_samples.outlier_unseen_same_samples.tolist()
        # Unseen same outliers have labels == 0
        labels = unseen_same_outliers["target"].tolist()
        assert labels == [0] * len(labels)

    def test_unseen_diff_outliers(self):
        unseen_diff_outliers = self.df_attack.query("type=='unseen_diff'")
        tensors = unseen_diff_outliers["tensors"].tolist()
        tensors.sort()
        assert tensors == self.mia_samples.outlier_unseen_diff_samples.tolist()
        # Unseen different outliers have labels == 0
        labels = unseen_diff_outliers["target"].tolist()
        assert labels == [0] * len(labels)
