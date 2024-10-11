import numpy as np
import torch

from opensynth.models.faraday.gaussian_mixture import GaussianMixtureModel


class TestGMM:

    def test_gmm_sampling(self):
        num_components = 2
        num_features = 3
        model = GaussianMixtureModel(num_components, num_features)
        num_samples = 1000

        # set component probs to test sampling
        model.component_probs = torch.tensor([0.1, 0.9])

        model.means = torch.tensor([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]])
        samples = model.sample(num_datapoints=num_samples)

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

    def test_gmm_forward(self):
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

        model.means = torch.tensor(
            [[1.0, 1.0, 1.0], [200, 200, 200]], dtype=torch.float32
        )

        # run a forward pass
        resp, _ = model.forward(data)

        # test that the responsibilities are as expected
        # first 4 samples should be in cluster 1, last 4 in cluster 2
        expected = [0, 0, 0, 0, 1, 1, 1, 1]

        # test that the responsibilities are as expected
        # first 4 samples should be in cluster 1, last 4 in cluster 2
        assert (np.argmax(np.exp(resp).numpy().round(1), 1) == expected).all