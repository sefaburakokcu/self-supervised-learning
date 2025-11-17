from models.ssl_methods.simclr import SimCLR


class SSLModelFactory:
    """Factory for creating SSL models"""

    @staticmethod
    def create(ssl_method: str, encoder, **kwargs):
        """Create SSL model from method name"""

        if ssl_method.lower() == 'simclr':
            return SimCLR(encoder, **kwargs)

        else:
            raise ValueError(f"Unknown SSL method: {ssl_method}")