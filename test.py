import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from interpolate import interp1d


if __name__ == "__main__":
    # problem dimensions
    D = 100000
    Dnew = 1
    N = 100
    P = 30

    # draw the data
    x = torch.rand(D, N, device='cuda') * 10000
    x = x.sort(dim=1)[0]

    y = torch.linspace(0, 1000, D*N).to('cuda').view(D, -1)
    y -= y[:, 0, None]

    xnew = torch.rand(Dnew, P, device='cuda')*10000

    print('Solving %d interpolation problems: '
          'each with %d observations and %d desired values' % (D, N, P))

    # launching the cuda version
    t0 = time.time()
    yq_gpu = interp1d(x, y, xnew, cuda=True)
    t1 = time.time()

    # putting everything back to CPU before calling the CPU version
    x = x.to('cpu')
    y = y.to('cpu')
    xnew = xnew.to('cpu')

    # calling the cpu version
    t0_cpu = time.time()
    yq_cpu = interp1d(x, y, xnew, cuda=False)
    t1_cpu = time.time()

    # compute the difference between both
    error = torch.norm(yq_cpu - yq_gpu.to('cpu'))/torch.norm(yq_cpu)*100.
    print('CPU: %0.3fms, GPU: %0.3fms, error: %f%%.' % ((t1_cpu-t0_cpu)*1000,
                                                        (t1-t0)*1000,
                                                        error))

    # plot the result for the first 10 dimensions at most
    d_plot = min(D, 10)
    x = x[:d_plot].cpu().numpy()
    y = y[:d_plot].cpu().numpy()
    xnew = xnew[:d_plot].cpu().numpy()
    yq_cpu = yq_cpu[:d_plot].cpu().numpy()
    yq_gpu = yq_gpu[:d_plot].cpu().numpy()

    plt.plot(x.T, y.T, '-',
             xnew.T, yq_gpu.T, 'o',
             xnew.T, yq_cpu.T, 'x')
    not_close = np.nonzero(np.invert(np.isclose(yq_gpu, yq_cpu)))
    if not_close[0].size:
        plt.scatter(xnew[not_close].T, yq_cpu[not_close].T,
                    edgecolors='r', s=100, facecolors='none')
    plt.grid(True)
    plt.show()
