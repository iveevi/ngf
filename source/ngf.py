import torch

from util import lerp

class NGF:
    def __init__(self, points, complexes, features, mlp):
        self.points = points
        self.complexes = complexes
        self.features = features
        self.mlp = mlp

        # Compile the kernel
        self.sample_kernel = torch.compile(self.sample_kernel_py, fullgraph=True)

    def sample(self, rate):
        return self.sample_kernel(rate)

    def sample_kernel_py(self, rate):
        # TODO: custom lerper for features?
        U = torch.linspace(0.0, 1.0, steps=rate).cuda()
        V = torch.linspace(0.0, 1.0, steps=rate).cuda()
        U, V = torch.meshgrid(U, V, indexing='ij')

        corner_points = self.points[self.complexes, :]
        corner_features = self.features[self.complexes, :]

        U, V = U.reshape(-1), V.reshape(-1)
        U = U.repeat((self.complexes.shape[0], 1))
        V = V.repeat((self.complexes.shape[0], 1))

        feature_size = self.features.shape[1]
        lerped_points = lerp(corner_points, U, V).reshape(-1, 3)
        lerped_features = lerp(corner_features, U, V).reshape(-1, feature_size)

        return lerped_points, lerped_features

    def save(self, filename):
        model = {
            'model'     : self.mlp,
            'features'  : self.features,
            'complexes' : self.complexes,
            'points'    : self.points
        }

        torch.save(model, filename)

def load_ngf(data):
    NGF(
        points    = data['points'],
        complexes = data['complexes'],
        features  = data['features'],
        mlp       = data['model']
    )
