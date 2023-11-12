from mlp import *

models = {
        'relu'   : MLP_Positional_Encoding,
        'elu'    : MLP_Positional_Elu_Encoding,
        'siren'  : MLP_Positional_Siren_Encoding,
        'gauss'  : MLP_Positional_Gaussian_Encoding,
        'sinc'   : MLP_Positional_Sinc_Encoding,
        'morlet' : MLP_Positional_Morlet_Encoding,
        'rexin'  : MLP_Positional_Rexin_Encoding,
        'reenc'  : MLP_Positional_Reencoding,
        'onion'  : MLP_Positional_Onion_Encoding,
}

lerps = {
        'linear' : lambda x: x,
}
