from mlp import *

models = {
        'pos'    : MLP_Positional_Encoding,
        'onion'  : MLP_Positional_Onion_Encoding,
        'morlet' : MLP_Positional_Morlet_Encoding,
        'feat'   : MLP_Feature_Sinusoidal_Encoding,
}

lerps = {
        'linear' : lambda x: x,
        'sin'    : lambda x: torch.sin(32.0 * x * np.pi / 2.0),
        'floor'  : lambda x: torch.floor(32 * x)/32.0,
        'smooth' : lambda x: (32.0 * x - torch.sin(32.0 * 2.0 * x * np.pi)/(2.0 * np.pi)) / 32.0,
        'cubic'  : lambda x: 25 * x ** 3/3.0 - 25 * x ** 2 + 31 * x/6.0,
}
