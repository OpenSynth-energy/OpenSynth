import numpy as np
import torch

from opensynth.models.faraday.gaussian_mixture import (
    GaussianMixtureModel,
    gmm_utils,
)


class TestGMM:

    def test_gmm_sampling(self):
        num_components = 2
        num_features = 3
        model = GaussianMixtureModel(num_components, num_features)
        num_samples = 1000

        # set component probs to test sampling
        model.weights = torch.tensor([0.1, 0.9])
        model.means = torch.tensor([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]])
        model.covariances = torch.tensor(
            [
                [
                    [1e-2, 0, 0],
                    [0, 1e-2, 0],
                    [0, 0, 1e-2],
                ],
                [
                    [1e-2, 0, 0],
                    [0, 1e-2, 0],
                    [0, 0, 1e-2],
                ],
            ]
        )
        samples = model.sample(num_samples)

        # test that the number of samples in both clusters is as expected
        assert (
            np.round(
                sum(samples.numpy().round(1).mean(axis=1) < 100) / num_samples,
                1,
            )
            == 0.1
        )
        assert (
            np.round(
                sum(samples.numpy().round(1).mean(axis=1) > 100) / num_samples,
                1,
            )
            == 0.9
        )

    def test_gmm_e_step(self):
        num_components = 2
        num_features = 3

        data = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1],
                [1.2, 1.2, 1.2],
                [1.1, 1.1, 1.1],
                [200, 200, 200],
                [210, 210, 210],
                [220, 220, 220],
                [230, 230, 230],
            ]
        )
        model = GaussianMixtureModel(num_components, num_features)

        gmm_init_params = {
            "means": torch.tensor(
                [[1.0, 1.0, 1.0], [200, 200, 200]], dtype=torch.float32
            ),
            "precision_cholesky": torch.tensor(
                [
                    [
                        [1e-2, 0, 0],
                        [0, 1e-2, 0],
                        [0, 0, 1e-2],
                    ],
                    [
                        [1e-2, 0, 0],
                        [0, 1e-2, 0],
                        [0, 0, 1e-2],
                    ],
                ]
            ),
            "weights": torch.tensor([0.1, 0.9]),
        }

        model.initialise(gmm_init_params)

        # run e-step
        _, resp = model.e_step(data)

        # test that the responsibilities are as expected
        # first 4 samples should be in cluster 1, last 4 in cluster 2
        expected = [0, 0, 0, 0, 1, 1, 1, 1]

        # test that the responsibilities are as expected
        # first 4 samples should be in cluster 1, last 4 in cluster 2
        assert (np.argmax(np.exp(resp).numpy().round(1), 1) == expected).all

    def test_m_step_statistics_accumulate_across_batches(self):
        # Accumulating sufficient statistics across mini-batches and
        # combining them once should give the same result as running the
        # M-step on the full dataset in a single batch. This guards
        # against the M-step silently overwriting mixture parameters on
        # each batch instead of accumulating across the whole epoch.
        torch.manual_seed(0)
        num_components = 3
        num_features = 4
        num_samples = 50
        reg_covar = 1e-6

        X = torch.rand(num_samples, num_features) * 10
        raw_resp = torch.rand(num_samples, num_components)
        responsibilities = raw_resp / raw_resp.sum(dim=1, keepdim=True)

        full_weights, full_means, full_covariances = (
            gmm_utils.torch_estimate_gaussian_parameters(
                X, responsibilities=responsibilities, reg_covar=reg_covar
            )
        )

        batch_boundaries = [0, 7, 22, 41, num_samples]
        nk_total = torch.zeros(num_components)
        sk_total = torch.zeros(num_components, num_features)
        sk2_total = torch.zeros(num_components, num_features, num_features)
        for start, end in zip(batch_boundaries[:-1], batch_boundaries[1:]):
            nk, sk, sk2 = gmm_utils.torch_compute_sufficient_statistics(
                X[start:end], responsibilities[start:end]
            )
            nk_total += nk
            sk_total += sk
            sk2_total += sk2

        acc_weights, acc_means, acc_covariances = (
            gmm_utils.torch_estimate_gaussian_parameters_from_statistics(
                nk_total, sk_total, sk2_total, reg_covar=reg_covar
            )
        )

        assert torch.allclose(full_weights, acc_weights, atol=1e-5)
        assert torch.allclose(full_means, acc_means, atol=1e-5)
        assert torch.allclose(full_covariances, acc_covariances, atol=1e-4)
