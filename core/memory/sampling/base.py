

from .abstract import AbstractSampling


class BaseSampling(AbstractSampling):
    """
        Base implementation of Sampling. Creates different batches of
        predefined batch sizes as a generator function or normal
        batches.

        batch_size: int
            The size of one batch
    """
    pass