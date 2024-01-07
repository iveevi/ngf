import torch
import torch.nn.functional as functional

from mlp import MLP

class NGF:
    def __init__(self, points: torch.Tensor, complexes: torch.Tensor, features: torch.Tensor, config: dict, mlp=None):
        self.points    = points
        self.features  = features
        self.complexes = complexes

        # assert self.points.dtype == torch.bfloat16
        # assert self.features.dtype == torch.bfloat16
        # assert self.complexes.dtype == torch.int32

        self.encoding_levels = config['encoding_levels']
        self.jittering = config['jittering']
        self.encoder = config['encoder']
        self.normals = config['normals']

        features = config['features']

        self.ffin = features
        if self.encoder == 'sincos':
            self.ffin = features + 3 * (2 * self.encoding_levels + 1)
            # self.ffin = features + 3 * (self.encoding_levels + 1)
            self.encoder = self.sincos
        # elif self.encoder == 'extint':
        #     self.ffin = features + 3 + 4 * (2 * self.encoding_levels)
        #     self.encoder = self.extint
        else:
            raise ValueError('Unknown encoder: %s' % self.encoder)

        self.mlp = MLP(self.ffin, functional.leaky_relu).cuda()
        if mlp is not None:
            self.mlp.load_state_dict(mlp.state_dict())

        self.sampler = self.sample_uniform
        if self.jittering:
            self.sampler = self.sample_jittered

        self.config = config

        torch.set_float32_matmul_precision('high')
        # self.fast_eval = torch.compile(self.eval, fullgraph=True)
        # self.fast_eval = torch.compile(self.eval, mode='reduce-overhead')

        # Caches
        self.uv_cache = {}
        self.uv_mask = {}

    # List of parameters
    def parameters(self):
        return list(self.mlp.parameters()) + [ self.points, self.features ]

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

        # U_plus, U_minus = U.unsqueeze(-1), (1.0 - U).unsqueeze(-1)
        # V_plus, V_minus = V.unsqueeze(-1), (1.0 - V).unsqueeze(-1)
        #
        # UV = (16 * U_minus * V_minus * U_plus * V_plus).square().reshape(-1, 1)

        Umid = (U - 0.5).abs().reshape(-1, 1)
        Vmid = (V - 0.5).abs().reshape(-1, 1)

        # UV = (Umid ** 2 + Vmid ** 2).sqrt().reshape(-1, 1)

        X = self.encoder(points=lerped_points, features=lerped_features, u=Umid, v=Vmid)

        return self.mlp(lerped_points.float(), X.float())

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
            k = 2 ** i
            X += [ torch.sin(k * bases), torch.cos(k * bases) ]
        
        # for i in range(self.encoding_levels):
        #     discrete = (bases * 2 ** i).int().sum(dim=-1)
        #     frequency = (4 + discrete.remainder(5)).unsqueeze(-1)
        #     normed = (bases * 2 ** i).frac()
        #     wavelet = torch.exp(-frequency * normed ** 2) * torch.cos(frequency * normed)
        #     X += [ wavelet ]

        return torch.cat(X, dim=-1)

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

        U = torch.linspace(0.0, 1.0, steps=rate, dtype=torch.float16).cuda()
        V = torch.linspace(0.0, 1.0, steps=rate, dtype=torch.float16).cuda()
        # U = torch.linspace(0.0, 1.0, steps=rate).cuda()
        # V = torch.linspace(0.0, 1.0, steps=rate).cuda()
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
        rand_UV = (2 * torch.rand((2, U.shape[0], U.shape[1]), device='cuda', dtype=torch.float16) - 1)
        # rand_UV = (2 * torch.rand((2, U.shape[0], U.shape[1]), device='cuda') - 1)
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
