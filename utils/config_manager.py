import yaml

from pathlib import Path
from typing import Dict, Any, List


class ConfigManager:
        """Load and manage experiment configurations"""

        def __init__(self, defaults_path: str = 'configs/defaults.yaml'):
            self.defaults_path = Path(defaults_path)
            self.defaults = self._load_yaml(self.defaults_path)

        @staticmethod
        def _load_yaml(path: Path) -> Dict[str, Any]:
            """Load YAML file"""
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}

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

        @staticmethod
        def _convert_value(value: Any) -> Any:
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

        def get_experiment(self, exp_path: str) -> Dict[str, Any]:
            """
            Load single experiment configuration

            Args:
                exp_path: Path like "baselines/baseline_resnet18"

            Returns:
                Experiment configuration dictionary
            """
            config_path = Path('configs') / f"{exp_path}.yaml"

            if not config_path.exists():
                available = self._list_available_configs()
                raise FileNotFoundError(
                    f"Experiment '{exp_path}' not found\n"
                    f"Available experiments:\n{available}"
                )

            exp_config = self._load_yaml(config_path)

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

            config = {**self.defaults, **exp_config}

            return config

        def _list_available_configs(self) -> str:
            """List all available experiment configs"""
            output = []
            configs_dir = Path('configs')

            for category_dir in sorted(configs_dir.iterdir()):
                if category_dir.is_dir() and category_dir.name != '__pycache__':
                    output.append(f"\n{category_dir.name}:")

                    yaml_files = sorted(category_dir.glob('*.yaml'))
                    for yaml_file in yaml_files:
                        exp_name = yaml_file.stem
                        rel_path = f"{category_dir.name}/{exp_name}"
                        output.append(f"  - {rel_path}")

            return "\n".join(output)

        def print_available_experiments(self):
            """Print all available experiments"""
            print("\nAvailable Experiments:")
            print(self._list_available_configs())
            print()

        def print_experiment_info(self, exp_path: str):
            """Print detailed experiment information"""
            config = self.get_experiment(exp_path)

            print(f"\n{'=' * 80}")
            print(f"Experiment: {exp_path}")
            print(f"{'=' * 80}")

            print("\nConfiguration:")
            for key in ['arch', 'mode', 'ssl_method']:
                if key in config:
                    print(f"  {key}: {config[key]}")

            print("\nTraining Parameters:")
            for key in ['epochs', 'batch_size', 'finetune_epochs', 'finetune_batch_size']:
                if key in config:
                    print(f"  {key}: {config[key]}")

            print(f"\nOptimizer: {config.get('optimizer', 'N/A')}")
            if 'optimizer_params' in config:
                for key, val in config['optimizer_params'].items():
                    print(f"  {key}: {val}")

            print(f"\nScheduler: {config.get('scheduler', 'N/A')}")
            if 'scheduler_params' in config:
                for key, val in config['scheduler_params'].items():
                    print(f"  {key}: {val}")

            ssl_keys = ['projection_dim', 'hidden_dim', 'temperature', 'momentum',
                        'decoder_dim', 'mask_ratio']
            ssl_params = {k: v for k, v in config.items() if k in ssl_keys}
            if ssl_params:
                print("\nModel Specific Parameters:")
                for key, val in ssl_params.items():
                    print(f"  {key}: {val}")

            if config.get('linear_probe'):
                print("\nLinear Probing: ENABLED (encoder frozen)")

            if config.get('use_imagenet'):
                print("\nImageNet Pretraining: ENABLED")

            print(f"\n{'=' * 80}\n")