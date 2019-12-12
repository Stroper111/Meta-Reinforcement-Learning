import glob
import os
import unittest
import sys
import traceback


class Suppressor(object):
    """
        Supresses stdout calls
    """
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type_, value, trace):
        sys.stdout = self.stdout
        if type_ is not None:
            traceback.format_tb(trace)

    def write(self, x):
        """ This is the redirect.  """
        pass


with Suppressor():
    from core.models import BaseModel


class TestSimpleModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_shape = (64, 64, 3)
        cls.action_space = 5

        with Suppressor():
            cls.model = BaseModel(cls.input_shape, cls.action_space)

        cls.save_msg = cls.model.save_msg
        cls.back_up_count = cls.model.back_up_count

        cls.full_path_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

    @classmethod
    def tearDownClass(cls) -> None:
        for file in glob.glob(os.path.join(cls.full_path_directory, "*.h5")):
            os.remove(file)

        if os.path.exists(cls.full_path_directory):
            os.removedirs(cls.full_path_directory)

    def test_init(self):
        self.assertEqual(self.input_shape, self.model.input_shape, "Wrong input dimensions")
        self.assertEqual(self.action_space, self.model.action_space, "Wrong action space")

    def test_saving_loading(self):
        self.model.save_model(self.full_path_directory)
        self.model.load_model(self.full_path_directory)

        for episode in range(15):
            self.model.episodes = episode * 100
            self.model.frames = episode * 4048
            self.model.save_checkpoint(self.full_path_directory)

        saved_models = glob.glob(os.path.join(self.full_path_directory, "*weights.h5"))
        self.assertEqual(self.back_up_count, len(saved_models), "The back_up count is not set correctly")
        self.model.load_checkpoint(self.full_path_directory)
