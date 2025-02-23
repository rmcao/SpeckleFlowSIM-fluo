import numpy as np
import jax.numpy as jnp
import jax
from flax import linen as nn
from flax.struct import dataclass

import calcil as cc
from calcil.physics.wave_optics import genPupil, zernikePolynomial
from nstm import utils
from nstm import spacetime


class SpeckleFlowFluo(cc.forward.Model):
    spacetime_param: spacetime.SpaceTimeParameters
    optical_param: utils.SystemParameters
    annealed_epoch: float = 1

    def setup(self):
        assert (self.optical_param.padding_yx[0] == 0 and self.optical_param.padding_yx[1] == 0)

        self.padding_yx = ((self.optical_param.dim_yx[0]//2, self.optical_param.dim_yx[0] - self.optical_param.dim_yx[0]//2),
                           (self.optical_param.dim_yx[1]//2, self.optical_param.dim_yx[1] - self.optical_param.dim_yx[1]//2))
        self.spacetime = spacetime.SpaceTimeMLP(self.optical_param,
                                                self.spacetime_param,
                                                num_output_channels=1)

        self.I_speckle = self.param('I_speckle', nn.initializers.zeros, self.optical_param.dim_yx, jnp.float32)
        self.OTF = self.param('OTF', nn.initializers.ones,
                              (self.optical_param.dim_yx[0]*2, self.optical_param.dim_yx[1]*2), jnp.complex64)

    def __call__(self, input_dict):
        fluo_density = self.spacetime(t=input_dict['t'],
                                      coord_offset=np.zeros((1, 2)),
                                      alpha=input_dict['epoch']/self.annealed_epoch)[..., 0]
        self.sow('intermediates', 'fluo', fluo_density, init_fn=lambda: 0, reduce_fn=lambda x, y: y)

        imgs = jnp.fft.ifft2(self.OTF * jnp.fft.fft2(
            jnp.pad(self.I_speckle * fluo_density, ((0, 0), self.padding_yx[0], self.padding_yx[1]))))[:,
               self.padding_yx[0][0]:self.optical_param.dim_yx[0] + self.padding_yx[0][0],
               self.padding_yx[1][0]:self.optical_param.dim_yx[1] + self.padding_yx[1][0]].real

        return imgs


def gen_loss_l2(margin):
    assert margin >= 0, "the spatial margin needs to be non-negative."
    if margin == 0:
        def loss_l2(forward_output, variables, input_dict, intermediates):
            l2 = ((input_dict['img'] - forward_output) ** 2).mean()
            return l2

    else:
        def loss_l2(forward_output, variables, input_dict, intermediates):
            l2 = ((input_dict['img'] - forward_output)[:, margin:-margin, margin:-margin] ** 2).mean()
            return l2

    return loss_l2


def gen_loss_nonneg_reg():
    def loss_nonneg(forward_output, variables, input_dict, intermediates):
        reg = -jnp.minimum(intermediates['fluo'], 0.0).mean()
        return reg

    return loss_nonneg
