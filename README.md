# torchinterp1d
## CUDA 1-D interpolation for Pytorch

## Presentation

This repository implements an `interp1d` function, enabling
linear 1D interpolation on the GPU for Pytorch.

```
def interp1d(x, y, xnew, out=None, cuda=True):
```

This function returns interpolated values of a set of 1-D functions at the desired query points `xnew`.

It works similarly to Matlab™ or scipy functions with
the `linear` interpolation mode on, except that it parallelises over any number of desired interpolation problems and exploits CUDA on the GPU

### Parameters for `interp1d`

* `x` : a (N, ) or (D, N) Pytorch Tensor:
Either 1-D or 2-D. It contains the coordinates of the observed samples.

* `y` : (N,) or (D, N) Pytorch Tensor.
Either 1-D or 2-D. It contains the actual values that correspond to the coordinates given by `x`.
The length of `y` along its last dimension must be the same as that of `x`

* `xnew` : (P,) or (D, P) Pytorch Tensor.
Either 1-D or 2-D. If it is not 1-D, its length along the first dimension must be the same as that of whichever `x` and `y` is 2-D. x-coordinates for which we want the interpolated output.

* `out` : (D, P) Pytorch Tensor`
        Tensor for the output. If None: allocated automatically.
        
### Results

a Pytorch tensor of shape (D, P), containing the interpolated values.

## Installation

The CUDA `interp1d` function depends on the [torchsearchsorted](https://github.com/aliutkus/torchsearchsorted) repository.

If not installed, you must:
1. Clone that repo through `git clone git@github.com:aliutkus/torchsearchsorted.git`
2. Go in the corresponding `install` subdirectory of that repo and launch `python setup.py install`.


Then, go in the `install` subfolder and type `python setup.py install`

## Usage

Try out `python test.py`
```
Solving 100000 interpolation problems: each with 100 observations and 30 desired values
CPU: 8060.260ms, GPU: 70.735ms, error: 0.000000%.
```
