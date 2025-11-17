import yaml

from pathlib import Path
from typing import Dict, Any, List


class ConfigManager:
    """Manage experiment configurations from files"""

    EXPERIMENT_DIRS = {
        'baselines': 'configs/baselines',
        'ssl_pretraining': 'configs/ssl_pretraining',
        'linear_probing': 'configs/linear_probing',
        'full_finetune': 'configs/full_finetune',
        'transfer_learning': 'configs/transfer_learning',
    }

    def __init__(self, defaults_path: str = 'configs/defaults.yaml',
                 groups_path: str = 'configs/experiment_groups.yaml'):
        self.defaults_path = Path(defaults_path)
        self.groups_path = Path(groups_path)

        self.defaults = self._load_yaml(self.defaults_path)
        self.groups_config = self._load_yaml(self.groups_path)

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

    def _find_experiment_file(self, exp_name: str) -> Path:
        """Find experiment file by name"""

        # Direct path if provided (e.g., "baselines/baseline_resnet18")
        if '/' in exp_name:
            exp_path = Path('configs') / f"{exp_name}.yaml"
            if exp_path.exists():
                return exp_path

        # Search in all experiment directories
        for dir_path in self.EXPERIMENT_DIRS.values():
            exp_path = Path(dir_path) / f"{exp_name}.yaml"
            if exp_path.exists():
                return exp_path

        raise FileNotFoundError(f"Experiment '{exp_name}' not found")

    def get_experiment(self, exp_name: str) -> Dict[str, Any]:
        """Get experiment configuration"""

        exp_path = self._find_experiment_file(exp_name)
        exp_config = self._load_yaml(exp_path)

        # Convert parameter types
        if 'optimizer_params' in exp_config:
            exp_config['optimizer_params'] = self._convert_param_types(
                exp_config['optimizer_params']
            )

        if 'scheduler_params' in exp_config:
            exp_config['scheduler_params'] = self._convert_param_types(
                exp_config['scheduler_params']
            )

        # Convert numeric fields
        numeric_fields = [
            'epochs', 'batch_size', 'finetune_epochs', 'finetune_batch_size',
            'projection_dim', 'hidden_dim', 'temperature', 'momentum',
            'decoder_dim', 'decoder_depth', 'mask_ratio'
        ]

        for field in numeric_fields:
            if field in exp_config:
                exp_config[field] = self._convert_value(exp_config[field])

        # Merge with defaults
        defaults = self.defaults.copy()
        return {**defaults, **exp_config}

    def get_group(self, group_name: str) -> List[str]:
        """Get list of experiments in a group"""

        if 'experiment_groups' not in self.groups_config:
            raise ValueError("No experiment groups defined")

        if group_name not in self.groups_config['experiment_groups']:
            raise ValueError(f"Group '{group_name}' not found")

        return self.groups_config['experiment_groups'][group_name]

    def list_experiments(self) -> Dict[str, List[str]]:
        """List all available experiments"""

        experiments = {}

        for category, dir_path in self.EXPERIMENT_DIRS.items():
            category_path = Path(dir_path)
            if category_path.exists():
                yaml_files = list(category_path.glob('*.yaml'))
                experiments[category] = [
                    f.stem for f in yaml_files
                ]

        return experiments

    def list_groups(self) -> List[str]:
        """List all available groups"""
        if 'experiment_groups' in self.groups_config:
            return list(self.groups_config['experiment_groups'].keys())
        return []

    def print_experiment_info(self, exp_name: str):
        """Print detailed experiment information"""

        config = self.get_experiment(exp_name)

        print(f"\n{'=' * 80}")
        print(f"Experiment: {exp_name}")
        print(f"{'=' * 80}")

        # Basic info
        print(f"\n📊 Basic Configuration:")
        for key in ['arch', 'mode', 'ssl_method']:
            if key in config:
                print(f"  {key}: {config[key]}")

        # Training params
        print(f"\nTraining Configuration:")
        for key in ['epochs', 'batch_size', 'finetune_epochs']:
            if key in config:
                print(f"  {key}: {config[key]}")

        # Optimizer
        print(f"\nOptimizer: {config.get('optimizer', 'N/A')}")
        if 'optimizer_params' in config:
            for key, val in config['optimizer_params'].items():
                print(f"  {key}: {val}")

        # Scheduler
        print(f"\nScheduler: {config.get('scheduler', 'N/A')}")
        if 'scheduler_params' in config:
            for key, val in config['scheduler_params'].items():
                print(f"  {key}: {val}")

        # SSL specific
        ssl_keys = ['projection_dim', 'temperature', 'momentum', 'decoder_dim', 'mask_ratio']
        ssl_params = {k: v for k, v in config.items() if k in ssl_keys}
        if ssl_params:
            print(f"\nSSL Specific Parameters:")
            for key, val in ssl_params.items():
                print(f"  {key}: {val}")

        print(f"\n{'=' * 80}\n")