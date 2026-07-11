
import importlib.util
import logging
import os
import sys

from omegaconf import OmegaConf

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_interaction_class(cls_name):
    """Dynamically import and return the interaction class."""
    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    interaction_cls = getattr(module, class_name)
    return interaction_cls


def initialize_interactions_from_config(interaction_config_file):
    """Initialize interactions from configuration file.

    Args:
        interaction_config_file: Path to the interaction configuration file.

    Returns:
        dict: A dictionary mapping interaction names to BaseInteraction instances.
    """
    interaction_config = OmegaConf.load(interaction_config_file)
    interaction_map = {}

    for interaction_item in interaction_config.interaction:
        cls_name = interaction_item.class_name
        interaction_cls = get_interaction_class(cls_name)

        config = OmegaConf.to_container(interaction_item.config, resolve=True)

        name = interaction_item.get("name", None)
        if name is None:
            class_simple_name = cls_name.split(".")[-1]
            if class_simple_name.endswith("Interaction"):
                name = class_simple_name[:-11].lower()
            else:
                name = class_simple_name.lower()

        if name in interaction_map:
            raise ValueError(f"Duplicate interaction name '{name}' found. Each interaction must have a unique name.")

        config["name"] = name

        interaction = interaction_cls(config=config)
        interaction_map[name] = interaction

        logger.info(f"Initialized interaction '{name}' with class '{cls_name}'")

    return interaction_map
