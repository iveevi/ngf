from mlp import *

models = {
        'relu'   : MLP_Positional_ReLU_Encoding,
        'lrelu'  : MLP_Positional_LeakyReLU_Encoding,
        'elu'    : MLP_Positional_Elu_Encoding,
        'lelu'   : MLP_Positional_LeakyElu_Encoding,
        'siren'  : MLP_Positional_Siren_Encoding,
        'gauss'  : MLP_Positional_Gaussian_Encoding,
        'sinc'   : MLP_Positional_Sinc_Encoding,
        'morlet' : MLP_Positional_Morlet_Encoding,
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

def wavy_interpolation(X, U, V):
    def wavy(x, f, g):
        q = (1 - 4 * (x - 0.5) ** 2)
        return x + q * torch.sin((f ** 2 + 1) * (x - g))

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

    UV_times = Uplus * Uminus * Vminus * Vplus
    UV_times_wavy1 = torch.cos(16 * np.pi * (UV_times - 0.5)) * torch.exp(-8 * (UV_times - 0.5) ** 2)
    UV_times_wavy2 = torch.cos(8 * np.pi * (UV_times - 0.5)) * torch.exp(-4 * (UV_times - 0.5) ** 2)

    UV_plusplus   = Uplus * Vplus
    UV_plusminus  = Uplus * Vminus
    UV_minusplus  = Uminus * Vplus
    UV_minusminus = Uminus * Vminus

    lp00 = X00 * wavy(UV_minusminus, X00f, X00g)
    lp01 = X01 * wavy(UV_plusminus,  X01f, X01g)
    lp10 = X10 * wavy(UV_minusplus,  X10f, X10g)
    lp11 = X11 * wavy(UV_plusplus,   X11f, X11g)

    lp = lp00 + lp01 + lp10 + lp11
    lp = torch.cat([
        UV_times_wavy1,
        UV_times_wavy2,
        lp
    ], dim=-1)

    return lp

def quadratic_interpolation(X, U, V):
    def quadratic(x, a):
        return x + 4 * a * x * (1 - x)

    X00 = X[:, 0, :].unsqueeze(1)
    X01 = X[:, 1, :].unsqueeze(1)
    X10 = X[:, 3, :].unsqueeze(1)
    X11 = X[:, 2, :].unsqueeze(1)

    X00a = X00[:, :, 0].unsqueeze(-1)
    X01a = X01[:, :, 0].unsqueeze(-1)
    X10a = X10[:, :, 0].unsqueeze(-1)
    X11a = X11[:, :, 0].unsqueeze(-1)

    X00 = X00[:, :, 1:]
    X01 = X01[:, :, 1:]
    X10 = X10[:, :, 1:]
    X11 = X11[:, :, 1:]

    Uplus  = U.unsqueeze(-1)
    Uminus = 1.0 - U.unsqueeze(-1)
    Vplus  = V.unsqueeze(-1)
    Vminus = 1.0 - V.unsqueeze(-1)

    UV_plusplus   = Uplus * Vplus
    UV_plusminus  = Uplus * Vminus
    UV_minusplus  = Uminus * Vplus
    UV_minusminus = Uminus * Vminus

    UV_times = Uplus * Uminus * Vminus * Vplus
    UV_times_wavy = torch.cos(16 * np.pi * (UV_times - 0.5)) * torch.exp(-8 * (UV_times - 0.5) ** 2)

    lp00 = X00 * quadratic(UV_minusminus, X00a)
    lp01 = X01 * quadratic(UV_plusminus,  X01a)
    lp10 = X10 * quadratic(UV_minusplus,  X10a)
    lp11 = X11 * quadratic(UV_plusplus,   X11a)

    lp = lp00 + lp01 + lp10 + lp11
    lp = torch.cat([
        UV_times_wavy, lp
    ], dim=-1)

    return lp

def morlet_central(X, U, V):
    def morlet(x, k):
        return torch.cos(k * np.pi * x) * torch.exp(-k * x ** 2)

    X00 = X[:, 0, :].unsqueeze(1)
    X01 = X[:, 1, :].unsqueeze(1)
    X10 = X[:, 3, :].unsqueeze(1)
    X11 = X[:, 2, :].unsqueeze(1)

    Uplus  = U.unsqueeze(-1)
    Uminus = 1.0 - U.unsqueeze(-1)
    Vplus  = V.unsqueeze(-1)
    Vminus = 1.0 - V.unsqueeze(-1)

    UV_plusplus   = Uplus * Vplus
    UV_plusminus  = Uplus * Vminus
    UV_minusplus  = Uminus * Vplus
    UV_minusminus = Uminus * Vminus

    lp00 = X00 * UV_minusminus
    lp01 = X01 * UV_plusminus
    lp10 = X10 * UV_minusplus
    lp11 = X11 * UV_plusplus

    lp = (lp00 + lp01 + lp10 + lp11)[:, :, 5:]

    UV_times = 16 * Uplus * Uminus * Vminus * Vplus

    concats = [ lp ]
    for i in range(5):
        concats.append(morlet(1 - UV_times, 2 ** i))

    return torch.cat(concats, dim=-1)

def sin_central(X, U, V):
    def sin(x, k):
        return torch.sin(k * np.pi * x)

    X00 = X[:, 0, :].unsqueeze(1)
    X01 = X[:, 1, :].unsqueeze(1)
    X10 = X[:, 3, :].unsqueeze(1)
    X11 = X[:, 2, :].unsqueeze(1)

    Uplus  = U.unsqueeze(-1)
    Uminus = 1.0 - U.unsqueeze(-1)
    Vplus  = V.unsqueeze(-1)
    Vminus = 1.0 - V.unsqueeze(-1)

    UV_plusplus   = Uplus * Vplus
    UV_plusminus  = Uplus * Vminus
    UV_minusplus  = Uminus * Vplus
    UV_minusminus = Uminus * Vminus

    lp00 = X00 * UV_minusminus
    lp01 = X01 * UV_plusminus
    lp10 = X10 * UV_minusplus
    lp11 = X11 * UV_plusplus

    lp = (lp00 + lp01 + lp10 + lp11)[:, :, 5:]

    UV_times = 16 * Uplus * Uminus * Vminus * Vplus

    concats = [ lp ]
    for i in range(5):
        concats.append(sin(1 - UV_times, 2 ** i))

    return torch.cat(concats, dim=-1)

lerps = {
        'linear' : kernel_interpolation(lambda x: x),
        'floor'  : kernel_interpolation(lambda x: torch.floor(32 * x)/32),
        'wavy'   : wavy_interpolation,
        'quad'   : quadratic_interpolation,
        'morlet' : morlet_central,
        'sin'    : sin_central
}
