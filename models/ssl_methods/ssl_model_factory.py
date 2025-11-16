from models.ssl_methods.simclr import SimCLR
from models.ssl_methods.moco import MoCoV3
from models.ssl_methods.byol import BYOL
from models.ssl_methods.mae import MAE


class SSLModelFactory:
    """Factory for creating SSL models"""

    @staticmethod
    def create(ssl_method: str, encoder, **kwargs):
        """Create SSL model from method name"""

        if ssl_method.lower() == 'simclr':
            return SimCLR(encoder, **kwargs)

        elif ssl_method.lower() == 'moco':
            return MoCoV3(encoder, **kwargs)

        elif ssl_method.lower() == 'byol':
            return BYOL(encoder, **kwargs)

        elif ssl_method.lower() == 'mae':
            return MAE(encoder, **kwargs)

        else:
            raise ValueError(f"Unknown SSL method: {ssl_method}")