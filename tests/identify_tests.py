import os
import unittest

from click.testing import CliRunner

from riid.cli.main import identify
from riid.data.synthetic import get_dummy_seeds
from riid.data.synthetic.seed import SeedMixer
from riid.data.synthetic.static import StaticSynthesizer
from riid.models.neural_nets import MLPClassifier


class TestCliIdentifyCommand(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fg_seeds_ss, self.bg_seeds_ss = get_dummy_seeds().split_fg_and_bg()
        self.mixed_bg_seed_ss = SeedMixer(self.bg_seeds_ss, mixture_size=3).generate(10)

        self.static_synth = StaticSynthesizer(
            samples_per_seed=100,
            snr_function="log10",
            return_fg=False,
            return_gross=True
        )
        _, _, self.train_ss = self.static_synth.generate(
            self.fg_seeds_ss,
            self.mixed_bg_seed_ss,
            verbose=False
        )
        self.train_ss.normalize()

        self.model = MLPClassifier()
        self.history = self.model.fit(self.train_ss, epochs=10, patience=5).history
        self.model.predict(self.train_ss)

        # Generate some test data
        self.static_synth.samples_per_seed = 5
        _, _, self.test_ss = self.static_synth.generate(
            self.fg_seeds_ss,
            self.mixed_bg_seed_ss,
            verbose=False
        )
        self.test_ss.normalize()

        self.runner = CliRunner()

    def test_model_path_h5_data_path_h5_results_dir_varies(self):
        with self.runner.isolated_filesystem():
            self.model.save('model.h5')
            self.test_ss.to_hdf('data.h5')

            # User-provided results_dir does not exist
            result = self.runner.invoke(identify, ["model.h5", "data.h5",
                                                   "--results_dir", "not_test"])
            self.assertEqual(result.exit_code, 0)

            # User-provided results_dir does exist
            os.mkdir("./test")
            result = self.runner.invoke(identify, ["model.h5", "data.h5", "--results_dir", "test"])
            self.assertEqual(result.exit_code, 0)

            # User does not provide results_dir and default does not exist
            result = self.runner.invoke(identify, ["model.h5", "data.h5"])
            self.assertEqual(result.exit_code, 0)

            # User does not provide results_dir and default exists identify_results implicitly made
            result = self.runner.invoke(identify, ["model.h5", "data.h5"])
            self.assertEqual(result.exit_code, 0)

    def test_model_path_h5_data_path_pcf_results_dir_varies(self):
        with self.runner.isolated_filesystem():
            self.model.save('model.h5')
            self.test_ss.to_pcf('data.pcf')

            # User-provided results_dir does not exist
            result = self.runner.invoke(identify, ["model.h5", "data.pcf",
                                                   "--results_dir", "not_test"])
            self.assertEqual(result.exit_code, 0)

            # User-provided results_dir does exist
            os.mkdir("./test")
            result = self.runner.invoke(identify, ["model.h5", "data.pcf", "--results_dir", "test"])
            self.assertEqual(result.exit_code, 0)

            # User does not provide results_dir and default does not exist
            result = self.runner.invoke(identify, ["model.h5", "data.pcf"])
            self.assertEqual(result.exit_code, 0)

            # User does not provide results_dir and default exists identify_results implicitly made
            result = self.runner.invoke(identify, ["model.h5", "data.pcf"])
            self.assertEqual(result.exit_code, 0)

    def test_model_path_pcf_data_path_h5_results_dir_varies(self):
        with self.runner.isolated_filesystem():
            self.model.save('model.pcf')
            self.test_ss.to_hdf('data.h5')

            # User-provided results_dir does not exist
            result = self.runner.invoke(identify, ["model.pcf", "data.h5",
                                                   "--results_dir", "not_test"])
            self.assertEqual(result.exit_code, 0)

            # User-provided results_dir does exist
            os.mkdir("./test")
            result = self.runner.invoke(identify, ["model.pcf", "data.h5", "--results_dir", "test"])
            self.assertEqual(result.exit_code, 0)

            # User does not provide results_dir and default does not exist
            result = self.runner.invoke(identify, ["model.pcf", "data.h5"])
            self.assertEqual(result.exit_code, 0)

            # User does not provide results_dir and default exists identify_results implicitly made
            result = self.runner.invoke(identify, ["model.pcf", "data.h5"])
            self.assertEqual(result.exit_code, 0)

    def test_model_path_pcf_data_path_pcf_results_dir_varies(self):
        with self.runner.isolated_filesystem():
            self.model.save('model.pcf')
            self.test_ss.to_pcf('data.pcf')

            # User-provided results_dir does not exist
            result = self.runner.invoke(identify, ["model.pcf", "data.pcf",
                                                   "--results_dir", "not_test"])
            self.assertEqual(result.exit_code, 0)

            # User-provided results_dir does exist
            os.mkdir("./test")
            result = self.runner.invoke(identify, ["model.pcf", "data.pcf",
                                                   "--results_dir", "test"])
            self.assertEqual(result.exit_code, 0)

            # User does not provide results_dir and default does not exist
            result = self.runner.invoke(identify, ["model.pcf", "data.pcf"])
            self.assertEqual(result.exit_code, 0)

            # User does not provide results_dir and default exists identify_results implicitly made
            result = self.runner.invoke(identify, ["model.pcf", "data.pcf"])
            self.assertEqual(result.exit_code, 0)


if __name__ == '__main__':
    unittest.main()
