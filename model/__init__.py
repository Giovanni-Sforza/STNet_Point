from .PPC import PointPoolingClassifier as PPC
from .PPC_tonly4 import PointPoolingClassifier as PPC_Tonly4
from .PPC_ST import PointPoolingClassifier as PPC_ST
from .STDT import STDT
from .STNet_point2 import STNet_Point_Classifier_Advanced as STNet_Point

def create_model(config):
    """ Create model with given config, including coord_dim """
    model_name = config.get('model_name', 'STNet_Point')
    if model_name == 'PPC':
        model = PPC(
            input_feature_dim=config.get('input_feature_dim', 2), # Use a clearer name
            output_dim=config.get('output_dim', 512),
            num_class=config.get('num_class', 4),
            feature_out=config.get('feature_out', False),
            transformer_layers = config.get("transformer_layers",3),
            transformer_heads = config.get("transformer_heads",4)
        )
    elif model_name == 'PPC_ST':
        model = PPC_ST(
            input_feature_dim=config.get('input_feature_dim', 2), # Use a clearer name
            output_dim=config.get('output_dim', 512),
            num_class=config.get('num_class', 4),
            feature_out=config.get('feature_out', False)
        )
    elif model_name == 'PPC_Tonly4':
        model = PPC_Tonly4(
            input_feature_dim=config.get('input_feature_dim', 3), # Use a clearer name
            num_events = config.get("num_events",32),
            output_dim=config.get('output_dim', 512),
            num_classes=config.get('num_classes', 5),
            feature_out=config.get('feature_out', False),
            transformer_layers = config.get("transformer_layers",3),
            transformer_heads = config.get("transformer_heads",4)
        )
    elif model_name == 'STDT':
        model = STDT(
            input_feature_dim=config.get('input_feature_dim', 3), # Use a clearer name
            num_events = config.get("num_events",32),
            output_dim=config.get('output_dim', 512),
            num_classes=config.get('num_classes', 5),
            feature_out=config.get('feature_out', False)
        )
    elif model_name == 'STNet_Point':
        model = STNet_Point(
            input_feature_dim=config.get('input_feature_dim', 2), # Use a clearer name
            output_dim=config.get('output_dim', 512),
            num_class=config.get('num_class', 4),
            feature_out=config.get('feature_out', False),
            pretrain = config.get("pretrain",False)
        )
    else :
        print("this kind of model have not been defined")
    return model