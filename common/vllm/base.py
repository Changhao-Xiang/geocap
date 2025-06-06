# -*- coding: utf-8 -*-
# @Date    : 2025-02-27 14:53:27
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from abc import ABC, abstractmethod


class GenerateModelBase(ABC):
    @abstractmethod
    def __init__(self, model: str, **kwargs):
        """
        Initialize the model with the given model name and any other arguments
        Args:
            model: str, from eval_args.eval_model, usually in the format of {model_name}-{model_size}
            **kwargs: any other arguments to load the model
        """
        self.kwargs = kwargs
        kwargs["temperature"] = kwargs.get("temperature", 0.0)
        kwargs["top_p"] = kwargs.get("top_p", 1.0)
        kwargs["top_k"] = kwargs.get("top_k", 0)
        kwargs["do_sample"] = kwargs.get("do_sample", False)

    @abstractmethod
    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        """
        Generate the responses of the model on the given image paths and prompts
        Args:
            image_paths: list[str], the paths of the input images in a batch
            prompts: list[str], the user prompts for the model in a batch
        Returns:
            list[str]: the raw responses of the model (the input prompts should not be included)
        """
        pass
