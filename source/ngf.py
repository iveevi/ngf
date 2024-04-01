import torch

from mlp import MLP

class NGF:
    def __init__(self, points: torch.Tensor, complexes: torch.Tensor, features: torch.Tensor, config: dict, mlp=None):
        self.points = points
        self.features = features
        self.complexes = complexes

        self.encoding_levels = config['encoding_levels']
        self.jittering = config['jittering']
        self.normals = config['normals']

        features = config['features']

        self.ffin = features + 3 * (2 * self.encoding_levels + 1)

        self.mlp = MLP(self.ffin).cuda()
        if mlp is not None:
            self.mlp.load_state_dict(mlp.state_dict())

        self.sampler = self.sample_uniform
        if self.jittering:
            self.sampler = self.sample_jittered

        self.config = config

        torch.set_float32_matmul_precision('high')

        # Caches
        self.uv_cache = {}
        self.uv_mask = {}

    # List of parameters
    def parameters(self):
        return list(self.mlp.parameters()) + [ self.points, self.features ]

    # Positional encoding
    def posenc(self, I):
        X = [ I ]
        for i in range(self.encoding_levels):
            k = 2 ** i
            X += [ torch.sin(k * I), torch.cos(k * I) ]

        return torch.cat(X, dim=-1)

    # Extraction methods
    def eval(self, U, V):
        corner_points = self.points[self.complexes, :]
        corner_features = self.features[self.complexes, :]
        feature_size = self.features.shape[1]

        Up, Um = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        Vp, Vm = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

        lp00 = corner_points[:, 0, :].unsqueeze(1) * Um * Vm
        lp01 = corner_points[:, 1, :].unsqueeze(1) * Up * Vm
        lp10 = corner_points[:, 3, :].unsqueeze(1) * Um * Vp
        lp11 = corner_points[:, 2, :].unsqueeze(1) * Up * Vp

        lp = (lp00 + lp01 + lp10 + lp11).reshape(-1, 3)

        lf00 = corner_features[:, 0, :].unsqueeze(1) * Um * Vm
        lf01 = corner_features[:, 1, :].unsqueeze(1) * Up * Vm
        lf10 = corner_features[:, 3, :].unsqueeze(1) * Um * Vp
        lf11 = corner_features[:, 2, :].unsqueeze(1) * Up * Vp

        lf = (lf00 + lf01 + lf10 + lf11).reshape(-1, feature_size)

        I = torch.cat((lf, self.posenc(lp)), dim=-1)
        return lp + self.mlp(I)

    def base(self, rate):
        U, V = self.sample_uniform(rate)

        corner_points = self.points[self.complexes, :]

        Up, Um = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        Vp, m = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)

        lp00 = corner_points[:, 0, :].unsqueeze(1) * Um * m
        lp01 = corner_points[:, 1, :].unsqueeze(1) * Up * m
        lp10 = corner_points[:, 3, :].unsqueeze(1) * Um * Vp
        lp11 = corner_points[:, 2, :].unsqueeze(1) * Up * Vp

        return (lp00 + lp01 + lp10 + lp11).reshape(-1, 3)

    # Functionals
    def eval_normals(self, U, V, delta):
        Up, Um = U + delta, U - delta
        Vp, Vm = V + delta, V - delta

        v_Up, v_Um = self.eval(Up, V), self.eval(Um, V)
        v_Vp, v_Vm = self.eval(U, Vp), self.eval(U, Vm)

        v_dU = (v_Up - v_Um)/(2 * delta)
        v_dV = (v_Vp - v_Vm)/(2 * delta)

        N = torch.cross(v_dU.float(), v_dV.float())

        length = torch.linalg.vector_norm(N, axis=1)
        length = torch.where(length == 0, torch.ones_like(length), length)

        return N/length.unsqueeze(-1)

    # Sampling functions
    def sample_uniform(self, rate: int):
        if rate in self.uv_cache:
            return self.uv_cache[rate]

        U = torch.linspace(0.0, 1.0, steps=rate).cuda()
        V = torch.linspace(0.0, 1.0, steps=rate).cuda()
        U, V = torch.meshgrid(U, V, indexing='ij')

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        self.uv_cache[rate] = (U, V)

        return U, V

    def sample_jittered(self, rate: int):
        U, V = self.sample_uniform(rate)

        if rate in self.uv_mask:
            UV_interior = self.uv_mask[rate]
        else:
            U_plus, U_minus = U, (1.0 - U)
            V_plus, V_minus = V, (1.0 - V)
            UV_interior = (U_minus * U_plus * V_minus * V_plus) > 0
            self.uv_mask[rate] = UV_interior

        delta = 0.45/(rate - 1)

        # TODO: use a normal distribution with decreasing jittering...
        rand_UV = (2 * torch.rand((2, U.shape[0], U.shape[1]), device='cuda') - 1)
        rand_UV *= delta * UV_interior

        return U + rand_UV[0], V + rand_UV[1]

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
