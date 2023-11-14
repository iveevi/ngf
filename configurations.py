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

def kernel_interpolation(f=lambda x: x):
    def ftn(X, U, V):
        lp00 = X[:, 0, :].unsqueeze(1) * f(1.0 - U.unsqueeze(-1)) * f(1.0 - V.unsqueeze(-1))
        lp01 = X[:, 1, :].unsqueeze(1) * f(U.unsqueeze(-1)) * f(1.0 - V.unsqueeze(-1))
        lp10 = X[:, 3, :].unsqueeze(1) * f(1.0 - U.unsqueeze(-1)) * f(V.unsqueeze(-1))
        lp11 = X[:, 2, :].unsqueeze(1) * f(U.unsqueeze(-1)) * f(V.unsqueeze(-1))
        return lp00 + lp01 + lp10 + lp11
    return ftn

def morlet(x, f, g):
    q = (1 - 4 * (x - 0.5) ** 2)
    return x + q * torch.sin((f ** 2 + 1) * (x - g))

    # g = (g + 1)/2.0
    # return x + torch.sin(4 * np.pi * x) * torch.exp(-16 * (f * f) * (x - g) ** 2)

# TODO: sin as well...

def morlet_interpolation(X, U, V):
    X00 = X[:, 0, :].unsqueeze(1)
    X01 = X[:, 1, :].unsqueeze(1)
    X10 = X[:, 3, :].unsqueeze(1)
    X11 = X[:, 2, :].unsqueeze(1)

    X00f, X00g = X00[:, :, 0].unsqueeze(-1), X00[:, :, 1].unsqueeze(-1)
    X01f, X01g = X01[:, :, 0].unsqueeze(-1), X01[:, :, 1].unsqueeze(-1)
    X10f, X10g = X10[:, :, 0].unsqueeze(-1), X10[:, :, 1].unsqueeze(-1)
    X11f, X11g = X11[:, :, 0].unsqueeze(-1), X11[:, :, 1].unsqueeze(-1)

    X00 = X00[:, :, 2:]
    X01 = X01[:, :, 2:]
    X10 = X10[:, :, 2:]
    X11 = X11[:, :, 2:]

    Uplus  = U.unsqueeze(-1)
    Uminus = 1.0 - U.unsqueeze(-1)
    Vplus  = V.unsqueeze(-1)
    Vminus = 1.0 - V.unsqueeze(-1)

    UV_plusplus   = Uplus * Vplus
    UV_plusminus  = Uplus * Vminus
    UV_minusplus  = Uminus * Vplus
    UV_minusminus = Uminus * Vminus

    lp00 = X00 * morlet(UV_minusminus, X00f, X00g)
    lp01 = X01 * morlet(UV_plusminus,  X01f, X01g)
    lp10 = X10 * morlet(UV_minusplus,  X10f, X10g)
    lp11 = X11 * morlet(UV_plusplus,   X11f, X11g)

    lp = lp00 + lp01 + lp10 + lp11
    lp = torch.cat([
        torch.ones_like(lp[:, :, :2]), # TODO: try U and V
        lp
    ], dim=-1)

    return lp

lerps = {
        'linear' : kernel_interpolation(lambda x: x),
        'floor'  : kernel_interpolation(lambda x: torch.floor(32 * x)/32),
        'morlet' : morlet_interpolation,
}
