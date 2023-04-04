import abc

import torch
import torch.nn as nn


class MetaMask(nn.Module):
    """
    Mask applied to gradients for GradientLoss based objective.
    It has to put to zero the values we do not want to consider.

    WE HAVE TO BE CAREFUL BECAUSE MASK IS SETUP ONCE AND THEN NEVER CHANGED AGAIN.
    WE INDEEED MAKE THE HYPOTHESIS THAT WE ARE DOING AN ATTACK AND THAT THE `gradient_data` IS FIXED
    """

    def __init__(self):
        super().__init__()
        self.fixed_mask = None

    def forward(self, gradient_rec, gradient_data):
        if self.fixed_mask is None:
            self._generate_mask(gradient_data)
        for rec, data, mask in zip(gradient_rec, gradient_data, self.fixed_mask):
            rec.data = rec * mask
            data.data = data * mask
        return gradient_rec, gradient_data

    @abc.abstractmethod
    def _generate_mask(self, gradient_data):
        """generate a mask based on real gradient_data"""
        pass

    def process_architecture(self, model):
        """process architecture in order to have a OrderedDict with layers' name and params"""
        pass

    def get_sparsity(self):
        """return the sparsity of the mask"""
        return 100 - (sum([torch.sum(mask) for mask in self.fixed_mask]) / sum([torch.numel(mask) for mask in self.fixed_mask])) * 100


class mask_fixed_zero(MetaMask):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _generate_mask(self, gradients):
        self.fixed_mask = [torch.zeros_like(grad) for grad in gradients]


class mask_fixed_identity(MetaMask):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _generate_mask(self, gradients):
        self.fixed_mask = [torch.ones_like(grad) for grad in gradients]


class mask_fixed_bernoulli(MetaMask):
    def __init__(self, p, *args, **kwargs):
        super().__init__()
        self.p = p

    def _generate_mask(self, gradients):
        self.fixed_mask = [(torch.rand_like(grad) < self.p) for grad in gradients]


class mask_fixed_layer(MetaMask):
    def __init__(self, layer=None, nb_layer=1, *args, **kwargs):
        super().__init__()
        self.fixed_layer = layer
        self.nb_layer = nb_layer

    def _generate_mask(self, gradients):
        if self.fixed_layer is None:
            n = len(gradients)
            perm = torch.randperm(n)
            self.fixed_layer = perm[: self.nb_layer]
        self.fixed_mask = [torch.ones_like(grad) if i in self.fixed_layer else torch.zeros_like(grad) for i, grad in enumerate(gradients)]


class mask_adaptative_clip(MetaMask):
    """
    clip to zero the values of grad that are lower to a certain %-treshhold of gradient's norm.
    """

    def __init__(self, clip_value, norm_fn=lambda x: max(torch.max(y) for y in x), *args, **kwargs):
        super().__init__()
        self.clip_value = clip_value
        self.norm_fn = norm_fn
        self.need_init = True

    def _generate_mask(self, gradients):
        norm = self.norm_fn(gradients)
        self.fixed_mask = [(torch.abs(grad) >= self.clip_value * norm) for grad in gradients]
        self.need_init = False


class mask_adaptative_layer_clip(MetaMask):
    """
    clip to zero the values of grad that are lower to a certain %-treshhold of gradient's norm.
    """

    def __init__(self, clip_value, norm_fn=lambda x: max(torch.max(y) for y in x), *args, **kwargs):
        super().__init__()
        self.clip_value = clip_value
        self.norm_fn = norm_fn
        self.need_init = True

    def _generate_mask(self, gradients):
        self.fixed_mask = [(torch.abs(grad) >= self.clip_value * self.norm_fn(grad)) for grad in gradients]
        self.need_init = False


class mask_quantile_clip(MetaMask):
    """
    clip to zero the values of grad to keep only a certain %. Only small magnitude gradients are clipped.
    """

    def __init__(self, q, norm_fn=lambda x: max(torch.max(y) for y in x), *args, **kwargs):
        super().__init__()
        self.q = q
        self.norm_fn = norm_fn
        self.need_init = True

    def _generate_mask(self, gradients):
        concat_gradients = torch.cat([grad.flatten().abs() for grad in gradients])
        treshold = torch.quantile(concat_gradients, self.q)
        self.fixed_mask = [(torch.abs(grad) >= treshold) for grad in gradients]
        self.need_init = False


class mask_quantile_layer_clip(MetaMask):
    """
    clip to zero the values of grad to keep only a certain %. Only small magnitude gradients are clipped.
    """

    def __init__(self, q, norm_fn=lambda x: max(torch.max(y) for y in x), *args, **kwargs):
        super().__init__()
        self.q = q
        self.norm_fn = norm_fn
        self.need_init = True

    def _generate_mask(self, gradients):
        self.fixed_mask = []
        for grad in gradients:
            treshold_grad = torch.quantile(grad.flatten().abs(), self.q)
            self.fixed_mask.append((torch.abs(grad) >= treshold_grad))
            self.need_init = False


class mask_percentage_in_channels(MetaMask):
    def __init__(self, p, *args, **kwargs):
        super().__init__()
        self.p = p

    def _generate_mask(self, gradient_data):
        self.fixed_mask = []
        for grad in gradient_data:
            d = len(grad.shape)
            self.fixed_mask += dimension_lookup_in_channels[d](self.p, [grad])


class mask_percentage_out_channels(MetaMask):
    def __init__(self, p, *args, **kwargs):
        super().__init__()
        self.p = p

    def _generate_mask(self, gradient_data):
        self.fixed_mask = []
        for grad in gradient_data:
            d = len(grad.shape)
            self.fixed_mask += dimension_lookup_out_channels[d](self.p, [grad])


def MaskParameters1d_percentage(p, gradient_data):
    """1d such as bias with (out_channels)"""
    for grad in gradient_data:
        assert len(grad.shape) == 1
    fixed_mask = []
    for grad in gradient_data:
        out_channels = grad.shape[0]
        perm = torch.randperm(out_channels)
        mask = torch.zeros_like(grad)
        mask[perm[: int(p * out_channels)]] = 1.0
        fixed_mask.append(mask)
    return fixed_mask


def MaskParameters2d_percentage_out_channels(p, gradient_data):
    """2d such as Linear Layer with (out_channels, in_channels)"""
    fixed_mask = []
    for grad in gradient_data:
        assert len(grad.shape) == 2
        out_channels = grad.shape[0]
        perm = torch.randperm(out_channels)
        mask = torch.zeros_like(grad)
        mask[perm[: int(p * out_channels)]] = 1.0
        fixed_mask.append(mask)
    return fixed_mask


def MaskParameters2d_percentage_in_channels(p, gradient_data):
    """2d such as Linear Layer with (out_channels, in_channels)"""
    fixed_mask = []
    for grad in gradient_data:
        assert len(grad.shape) == 2
        out_channels = grad.shape[1]
        perm = torch.randperm(out_channels)
        mask = torch.zeros_like(grad)
        mask[:, perm[: int(p * out_channels)]] = 1.0
        fixed_mask.append(mask)
    return fixed_mask


# def MaskParameters2d_percentage_random_channels(p, gradient_data):
#     """2d such as Linear Layer with (out_channels, in_channels)"""
#     fixed_mask = []
#     for grad in gradient_data:
#         assert len(grad.shape) == 2
#         out_channels = grad.shape[0]
#         in_channels = grad.shape[1]
#         total_channels = out_channels * in_channels
#         perm = torch.randperm(out_channels)
#         mask = torch.zeros_like(grad)
#         mask[perm[:int(p * out_channels)]] = 1.
#         fixed_mask.append(mask)
#     return fixed_mask


def MaskParameters3d_percentage_out_channels(p, gradient_data):
    """3d such as Conv1d with (out_channels, in_channels, L)"""
    fixed_mask = []
    for grad in gradient_data:
        assert len(grad.shape) == 3
        out_channels = grad.shape[0]
        perm = torch.randperm(out_channels)
        mask = torch.zeros_like(grad)
        mask[perm[: int(p * out_channels)]] = 1.0
        fixed_mask.append(mask)
    return fixed_mask


def MaskParameters3d_percentage_in_channels(p, gradient_data):
    """3d such as Linear Layer with (out_channels, in_channels, L)"""
    fixed_mask = []
    for grad in gradient_data:
        assert len(grad.shape) == 3
        out_channels = grad.shape[1]
        perm = torch.randperm(out_channels)
        mask = torch.zeros_like(grad)
        mask[:, perm[: int(p * out_channels)]] = 1.0
        fixed_mask.append(mask)
    return fixed_mask


def MaskParameters4d_percentage_out_channels(p, gradient_data):
    """3d such as Conv1d with (out_channels, in_channels, L)"""
    fixed_mask = []
    for grad in gradient_data:
        assert len(grad.shape) == 4
        out_channels = grad.shape[0]
        perm = torch.randperm(out_channels)
        mask = torch.zeros_like(grad)
        mask[perm[: int(p * out_channels)]] = 1.0
        fixed_mask.append(mask)
    return fixed_mask


def MaskParameters4d_percentage_in_channels(p, gradient_data):
    """4d such as Conv2d with (out_channels, in_channels, H, W)"""
    fixed_mask = []
    for grad in gradient_data:
        assert len(grad.shape) == 4
        out_channels = grad.shape[1]
        perm = torch.randperm(out_channels)
        mask = torch.zeros_like(grad)
        mask[:, perm[: int(p * out_channels)]] = 1.0
        fixed_mask.append(mask)
    return fixed_mask


mask_lookup = {
    "mask_fixed_zero": mask_fixed_zero,
    "mask_fixed_identity": mask_fixed_identity,
    "mask_fixed_bernoulli": mask_fixed_bernoulli,
    "mask_fixed_layer": mask_fixed_layer,
    "mask_quantile_clip": mask_quantile_clip,
    "mask_quantile_layer_clip": mask_quantile_layer_clip,
    "mask_adaptative_clip": mask_adaptative_clip,
    "mask_adaptative_layer_clip": mask_adaptative_layer_clip,
    "mask_percentage_in_channels": mask_percentage_in_channels,
    "mask_percentage_out_channels": mask_percentage_out_channels,
}

dimension_lookup_in_channels = {
    1: MaskParameters1d_percentage,
    2: MaskParameters2d_percentage_in_channels,
    3: MaskParameters3d_percentage_in_channels,
    4: MaskParameters4d_percentage_in_channels,
}

dimension_lookup_out_channels = {
    1: MaskParameters1d_percentage,
    2: MaskParameters2d_percentage_out_channels,
    3: MaskParameters3d_percentage_out_channels,
    4: MaskParameters4d_percentage_out_channels,
}

# dimension_lookup_random_channels = {
#     1: MaskParameters1d_percentage,
#     2: MaskParameters2d_percentage_random_channels,
#     3: MaskParameters3d_percentage_random_channels,
#     4: MaskParameters4d_percentage_random_channels,
# }
