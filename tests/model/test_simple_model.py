import glob
import os
import unittest

from core.models import SimpleModel


class TestSimpleModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_shape = (64, 64, 3)
        cls.action_space = 5

        cls.model = SimpleModel(cls.input_shape, cls.action_space)
        cls.save_msg = cls.model.save_msg
        cls.back_up_count = cls.model.back_up_count

    @classmethod
    def tearDownClass(cls) -> None:
        for file in glob.glob(os.path.join("temp", "*.h5")):
            os.remove(file)

        if os.path.exists("temp"):
            os.removedirs("temp")

    def test_init(self):
        self.assertEqual(self.input_shape, self.model.input_shape, "Wrong input dimensions")
        self.assertEqual(self.action_space, self.model.action_space, "Wrong action space")

    def test_saving_loading(self):
        full_path_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        self.model.save_model(full_path_directory)
        self.model.load_model(full_path_directory)

        for episode in range(15):
            self.model.episodes = episode * 100
            self.model.frames = episode * 4048
            self.model.save_checkpoint(full_path_directory)

        saved_models = glob.glob(os.path.join(full_path_directory, "*weights.h5"))
        self.assertEqual(self.back_up_count, len(saved_models), "The back_up count is not set correctly")
        self.model.load_checkpoint(full_path_directory)