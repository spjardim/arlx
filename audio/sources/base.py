from abc import ABC, abstractmethod
import numpy as np
class AudioSource(ABC):

    @abstractmethod
    def start(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_latest_sample(self) -> np.ndarray:
        raise NotImplementedError
    
