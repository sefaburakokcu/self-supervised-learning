import yaml
import argparse

from pathlib import Path
from typing import Dict, Any


EXPERIMENT_CATEGORIES = ['baseline_experiments', 'ssl_pretraining_experiments',
                         'linear_probing_experiments', 'full_finetune_experiments',
                         'transfer_learning_experiments']


class ConfigManager:
    """Manage experiment configurations"""

    def __init__(self, config_path: str = 'configs/experiments.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _convert_param_types(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string parameters to proper types"""

        converted = {}

        for key, value in params.items():
            if isinstance(value, list):
                converted[key] = [self._convert_value(v) for v in value]
            elif isinstance(value, dict):
                converted[key] = self._convert_param_types(value)
            else:
                converted[key] = self._convert_value(value)

        return converted

    def _convert_value(self, value: Any) -> Any:
        """Convert single value to proper type"""

        if isinstance(value, str):
            try:
                if 'e' in value.lower() or '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value

        return value

    def get_experiment(self, exp_name: str) -> Dict[str, Any]:
        """Get experiment configuration by name"""

        for category in EXPERIMENT_CATEGORIES:
            if category in self.config:
                if exp_name in self.config[category]:
                    exp_config = self.config[category][exp_name].copy()

                    if 'optimizer_params' in exp_config:
                        exp_config['optimizer_params'] = self._convert_param_types(
                            exp_config['optimizer_params']
                        )

                    if 'scheduler_params' in exp_config:
                        exp_config['scheduler_params'] = self._convert_param_types(
                            exp_config['scheduler_params']
                        )

                    numeric_fields = [
                        'epochs', 'batch_size', 'finetune_epochs', 'finetune_batch_size',
                        'projection_dim', 'hidden_dim', 'temperature', 'momentum',
                        'decoder_dim', 'decoder_depth', 'mask_ratio'
                    ]

                    for field in numeric_fields:
                        if field in exp_config:
                            exp_config[field] = self._convert_value(exp_config[field])

                    defaults = self.config.get('defaults', {}).copy()
                    return {**defaults, **exp_config}

        raise ValueError(f"Experiment '{exp_name}' not found")

    def get_group(self, group_name: str) -> list:
        """Get list of experiments in a group"""
        if 'experiment_groups' not in self.config:
            raise ValueError("No experiment groups defined")

        if group_name not in self.config['experiment_groups']:
            raise ValueError(f"Group '{group_name}' not found")

        return self.config['experiment_groups'][group_name]

    def list_experiments(self) -> Dict[str, list]:
        """List all available experiments"""
        experiments = {}

        for category in EXPERIMENT_CATEGORIES:
            if category in self.config:
                experiments[category] = list(self.config[category].keys())

        return experiments

    def list_groups(self) -> list:
        """List all available groups"""
        if 'experiment_groups' in self.config:
            return list(self.config['experiment_groups'].keys())
        return []


def create_argparse_from_config(config: Dict[str, Any]) -> argparse.Namespace:
    """Convert config dict to argparse Namespace"""

    args_dict = {}

    for key, value in config.items():
        if not isinstance(value, dict):
            args_dict[key] = value

    return argparse.Namespace(**args_dict)
