from mlp import *

models = {
        'pos'    : MLP_Positional_Encoding,
        'onion'  : MLP_Positional_Onion_Encoding,
        'morlet' : MLP_Positional_Morlet_Encoding,
        'feat'   : MLP_Feature_Sinusoidal_Encoding,
}

lerps = {
        'linear' : lambda x: x,
        'floor'  : lambda x: torch.floor(16 * x)/16.0,
        'sin'    : lambda x: torch.sin(16.0 * np.pi * x)/(4.0 * np.pi) + x,
        'sincos' : lambda x: torch.sin(16.0 * np.pi * x) * torch.cos(32.0 * np.pi * x)/(4.0 * np.pi) + x,
}
