import torch
import torch.nn as nn
import ngfutil


class NGFTextureFetchFunction(torch.autograd.Function):
    @staticmethod
    def forward(map, u, v, complexes, resx, resy, rate):
        return ngfutil.ngf_texture_fetch_forward(map, u, v, complexes, resx, resy, rate)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        map, u, v, complexes, resx, resy, rate = inputs
        ctx.mark_non_differentiable(u)
        ctx.mark_non_differentiable(v)
        ctx.save_for_backward(u, v)
        ctx.rest = [complexes, resx, resy, rate]

    @staticmethod
    def backward(ctx, d_color):
        u, v = ctx.saved_tensors
        # print('dCOLOR', d_color)
        d_map = ngfutil.ngf_texture_fetch_backward(d_color, u, v, *ctx.rest)
        # print('dMAP', d_map)
        return d_map, None, None, None, None, None, None


class NGFTextureFetch(nn.Module):
    def __init__(self, complexes, resx, resy, rate):
        super().__init__()
        self.complexes = complexes
        self.resx = resx
        self.resy = resy
        self.rate = rate

    def forward(self, map, u, v):
        return NGFTextureFetchFunction.apply(map, u, v, self.complexes, self.resx, self.resy, self.rate)
