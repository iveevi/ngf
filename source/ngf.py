import torch
import torch.nn.functional as functional

from mlp import MLP

class NGF:
    def __init__(self, points: torch.Tensor, complexes: torch.Tensor, features: torch.Tensor, config: dict, mlp=None):
        self.points = points
        self.complexes = complexes
        self.features = features

        self.encoding_levels = config['encoding_levels']
        self.jittering = config['jittering']
        self.encoder = config['encoder']
        self.normals = config['normals']

        features = config['features']
        ffin = features

        if self.encoder == 'sincos':
            ffin = features + 3 * (2 * self.encoding_levels + 1)
            self.encoder = self.sincos
        elif self.encoder == 'skewed':
            ffin = features + 3 * (self.encoding_levels + 1)
            self.encoder = self.skewed
        else:
            raise ValueError('Unknown encoder: %s' % self.encoder)

        self.mlp = MLP(ffin, functional.leaky_relu).cuda()
        if mlp is not None:
            self.mlp.load_state_dict(mlp.state_dict())

        self.sampler = self.sample_uniform
        if self.jittering:
            self.sampler = self.sample_jittered

        self.config = config

    # Extraction methods
    def eval(self, U, V):
        corner_points = self.points[self.complexes, :]
        corner_features = self.features[self.complexes, :]
        feature_size = self.features.shape[1]

        U_plus, U_minus = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        V_plus, V_minus = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

        lp00 = corner_points[:, 0, :].unsqueeze(1) * U_minus * V_minus
        lp01 = corner_points[:, 1, :].unsqueeze(1) * U_plus * V_minus
        lp10 = corner_points[:, 3, :].unsqueeze(1) * U_minus * V_plus
        lp11 = corner_points[:, 2, :].unsqueeze(1) * U_plus * V_plus

        lerped_points = (lp00 + lp01 + lp10 + lp11).reshape(-1, 3)

        lf00 = corner_features[:, 0, :].unsqueeze(1) * U_minus * V_minus
        lf01 = corner_features[:, 1, :].unsqueeze(1) * U_plus * V_minus
        lf10 = corner_features[:, 3, :].unsqueeze(1) * U_minus * V_plus
        lf11 = corner_features[:, 2, :].unsqueeze(1) * U_plus * V_plus

        lerped_features = (lf00 + lf01 + lf10 + lf11).reshape(-1, feature_size)

        X = self.encoder(points=lerped_points, features=lerped_features)

        return self.mlp(lerped_points, X)
    
    def base(self, rate):
        U, V = self.sample_uniform(rate)

        corner_points = self.points[self.complexes, :]

        U_plus, U_minus = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        V_plus, V_minus = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

        lp00 = corner_points[:, 0, :].unsqueeze(1) * U_minus * V_minus
        lp01 = corner_points[:, 1, :].unsqueeze(1) * U_plus * V_minus
        lp10 = corner_points[:, 3, :].unsqueeze(1) * U_minus * V_plus
        lp11 = corner_points[:, 2, :].unsqueeze(1) * U_plus * V_plus

        lerped_points = (lp00 + lp01 + lp10 + lp11).reshape(-1, 3)

        return lerped_points

    # Encoding functions
    def sincos(self, **kwargs):
        bases    = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]
        for i in range(self.encoding_levels):
            X += [ torch.sin(2 ** i * bases), torch.cos(2 ** i * bases) ]

        return torch.cat(X, dim=-1)

    def skewed(self, **kwargs):
        bases    = kwargs['points']
        features = kwargs['features']

        X = [ features, bases ]

        for i in range(self.encoding_levels):
            k = 2 ** (i/3.0)
            X += [ torch.sin(k * bases + 2/(i + 1)) ]

        return torch.cat(X, dim=-1)

    # Functionals
    def eval_normals(self, U, V, delta):
        Up, Um = U + delta, U - delta
        Vp, Vm = V + delta, V - delta

        v_Up, v_Um = self.eval(Up, V), self.eval(Um, V)
        v_Vp, v_Vm = self.eval(U, Vp), self.eval(U, Vm)

        v_dU = (v_Up - v_Um)/(2 * delta)
        v_dV = (v_Vp - v_Vm)/(2 * delta)

        N = torch.cross(v_dU, v_dV)
        length = torch.linalg.vector_norm(N, axis=1)
        length = torch.where(length == 0, torch.ones_like(length), length)

        return N/length.unsqueeze(-1)

    # Sampling functions
    def sample_uniform(self, rate: int):
        U = torch.linspace(0.0, 1.0, steps=rate).cuda()
        V = torch.linspace(0.0, 1.0, steps=rate).cuda()
        U, V = torch.meshgrid(U, V, indexing='ij')

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        return U, V

    def sample_jittered(self, rate: int):
        delta = 0.45/(rate - 1)

        U = torch.linspace(0.0, 1.0, steps=rate).cuda()
        V = torch.linspace(0.0, 1.0, steps=rate).cuda()
        U, V = torch.meshgrid(U, V, indexing='ij')

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        U_plus, U_minus = U, (1.0 - U)
        V_plus, V_minus = V, (1.0 - V)

        UV_interior = (U_minus * U_plus * V_minus * V_plus) > 0

        rand_U = (2 * torch.rand(U.shape, device='cuda') - 1) * delta
        rand_V = (2 * torch.rand(V.shape, device='cuda') - 1) * delta

        rand_U *= UV_interior
        rand_V *= UV_interior

        return U + rand_U, V + rand_V

    def save(self, filename):
        model = {
            'model'     : self.mlp,
            'features'  : self.features,
            'complexes' : self.complexes,
            'points'    : self.points,
            'config'    : self.config
        }

        torch.save(model, filename)

def load_ngf(data):
    return NGF(data['points'], data['complexes'], data['features'], data['config'], mlp=data['model'])
