import torch
try:
    from pytorch_searchsorted.searchsorted import searchsorted
except ImportError:
    raise ImportError(
     '\nThe CUDA interp1d function depends on the '
     'pytorch-searchsorted module.\n'
     'You must:\n'
     ' * clone this repo through '
     '`git clone git@github.com:aliutkus/pytorch_searchsorted.git`.\n'
     ' * Go in the corresponding directory and launch `make`.\n'
     ' * Have this pytorch_searchsorted directory in your Python path\n')

import scipy.interpolate as sp_interp


class Interp1d(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y, xnew, out=None, cuda=True):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.

        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        cuda : bool
            if True, will attempt to use CUDA if available.
            If False, a non-optimized loop over D will take place, calling the
            scipy function for each of the D problemns.

        Dependencies
        ------------
        If CUDA is selected there is a dependency with the module
        (`pytorch-searchsorted`)[https://github.com/aliutkus/pytorch-searchsorted]

        """
        # making the vectors at least 2D
        is_flat = {}
        v = {}
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].view(1, -1)
            reshaped_xnew = True

        # identifying the device to use
        device = 'cpu' if not cuda or not torch.cuda.is_available() else 'cuda'

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            assert out.numel() == shape_ynew[0]*shape_ynew[1], (
                    'The output provided is of incorrect shape.')
            ynew = out.reshape(shape_ynew)
        else:
            ynew = torch.zeros(*shape_ynew, device=device)

        # starting with handling the cpu version through scipy
        if device == 'cpu':
            # helper function to get the unique row in case of 1-D inputs
            def get_row(name, d):
                if v[name].shape[0] == 1:
                    return v[name][0].cpu().numpy()
                else:
                    return v[name][d].cpu().numpy()

            # for each dimension, apply interpolation
            for d in range(D):
                interp_func = sp_interp.interp1d(get_row('x', d),
                                                 get_row('y', d),
                                                 bounds_error=False,
                                                 fill_value='extrapolate')
                ynew[d] = torch.tensor(interp_func(get_row('xnew', d)),
                                       device='cpu')
            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)
            return ynew

        # the CPU case is handled if needs be. now we switch to the GPU case.
        # moving everything to GPU in case it was not there already (not
        # handlingthe case things do not fit entirely, user will do it if
        # required.)
        for name in v:
            v[name] = v[name].to('cuda')

        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (v['x'][:, 1:]-v['x'][:, :-1])
                )
        num_slopes = v['slopes'].shape[1]

        # calling searchsorted on the x values.
        ind = ynew
        searchsorted(v['x'], v['xnew'], ind)
        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        ind = torch.clamp(ind, 0, num_slopes - 1).long()

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # now build the linear interpolation
        ynew = sel('y') + sel('slopes')*(v['xnew'] - sel('x'))
        if reshaped_xnew:
            ynew = ynew.view(original_xnew_shape)

        return ynew
