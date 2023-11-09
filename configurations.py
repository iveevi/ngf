from mlp import *

models = {
        'relu'   : MLP_Positional_Encoding,
        'elu'    : MLP_Positional_Elu_Encoding,
        'siren'  : MLP_Positional_Siren_Encoding,
        'gauss'  : MLP_Positional_Gaussian_Encoding,
        'sinc'   : MLP_Positional_Sinc_Encoding,
        'onion'  : MLP_Positional_Onion_Encoding,
        'morlet' : MLP_Positional_Morlet_Encoding,
}

lerps = {
        'linear' : lambda x: x,
        'floor'  : lambda x: torch.floor(16 * x)/16.0,
        'sin'    : lambda x: torch.sin(16.0 * np.pi * x)/(4.0 * np.pi) + x,
        'sincos' : lambda x: torch.sin(16.0 * np.pi * x) * torch.cos(32.0 * np.pi * x)/(4.0 * np.pi) + x,
}
