from abc import abstractmethod, ABC
from typing import Dict, Union, List
import torch
import pytorch_lightning as pl
from torch import Tensor
from .dataclasses import AgentInput, Scene, Trajectory, SensorConfig

class AbstractFeatureBuilder:
    """Abstract class of feature builder for agent training."""

    def __init__(self):
        pass

    @abstractmethod
    def get_unique_name(self) -> str:
        """
        :return: Unique name of created feature.
        """
        pass

    @abstractmethod
    def compute_features(self, agent_input: AgentInput) -> Dict[str, Tensor]:
        """
        Computes features from the AgentInput object, i.e., without access to ground-truth.
        Outputs a dictionary where each item has a unique identifier and maps to a single feature tensor.
        One FeatureBuilder can return a dict with multiple FeatureTensors.
        """
        pass


class AbstractTargetBuilder:
    def __init__(self):
        pass

    @abstractmethod
    def get_unique_name(self) -> str:
        """
        :return: Unique name of created target.
        """
        pass

    @abstractmethod
    def compute_targets(self, scene: Scene) -> Dict[str, Tensor]:
        """
        Computes targets from the Scene object, i.e., with access to ground-truth.
        Outputs a dictionary where each item has a unique identifier and maps to a single target tensor.
        One TargetBuilder can return a dict with multiple TargetTensors.
        """
        pass


class AbstractAgent(torch.nn.Module, ABC):
    """Interface for an agent in NAVSIM."""

    def __init__(
        self,
        requires_scene: bool = False,
    ):
        super().__init__()
        self.requires_scene = requires_scene

    @abstractmethod
    def name(self) -> str:
        """
        :return: string describing name of this agent.
        """
        pass

    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: Dataclass defining the sensor configuration for lidar and cameras.
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize agent
        :param initialization: Initialization class.
        """
        pass

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the agent.
        :param features: Dictionary of features.
        :return: Dictionary of predictions.
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: List of target builders.
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: List of feature builders.
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        # extract trajectory
        return Trajectory(poses)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss used for backpropagation based on the features, targets and model predictions.
        """
        raise NotImplementedError("No loss. Agent does not support training.")

    def get_optimizers(
        self,
    ) -> Union[torch.optim.Optimizer, Dict[str, Union[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]]]:
        """
        Returns the optimizers that are used by thy pytorch-lightning trainer.
        Has to be either a single optimizer or a dict of optimizer and lr scheduler.
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """
        Returns a list of pytorch-lightning callbacks that are used during training.
        See navsim.planning.training.callbacks for examples.
        """
        return []

from enum import IntEnum


class StateSE2Index(IntEnum):
    """Intenum for SE(2) arrays."""

    _X = 0
    _Y = 1
    _HEADING = 2

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)


class BoundingBoxIndex(IntEnum):
    """Intenum of bounding boxes in logs."""

    _X = 0
    _Y = 1
    _Z = 2
    _LENGTH = 3
    _WIDTH = 4
    _HEIGHT = 5
    _HEADING = 6

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        return cls._Z

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def HEIGHT(cls):
        return cls._HEIGHT

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)

    @classmethod
    @property
    def DIMENSION(cls):
        # assumes LENGTH, WIDTH, HEIGHT have subsequent indices
        return slice(cls._LENGTH, cls._HEIGHT + 1)


class LidarIndex(IntEnum):
    """Intenum for lidar point cloud arrays."""

    _X = 0
    _Y = 1
    _Z = 2
    _INTENSITY = 3
    _RING = 4
    _ID = 5

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def Z(cls):
        return cls._Z

    @classmethod
    @property
    def INTENSITY(cls):
        return cls._INTENSITY

    @classmethod
    @property
    def RING(cls):
        return cls._RING

    @classmethod
    @property
    def ID(cls):
        return cls._ID

    @classmethod
    @property
    def POINT2D(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def POSITION(cls):
        # assumes X, Y, Z have subsequent indices
        return slice(cls._X, cls._Z + 1)
