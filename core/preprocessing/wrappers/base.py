

class BaseWrapper:
    def __init__(self, setup):
        self.setup = setup

    def process(self, img):
        raise NotImplementedError
