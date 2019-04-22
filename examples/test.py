import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from torchinterp1d import Interp1d


if __name__ == "__main__":
    # defining the number of tests
    ntests = 2

    # problem dimensions
    D = 1000
    Dnew = 1
    N = 100
    P = 30

    yq_gpu = None
    yq_cpu = None
    for ntest in range(ntests):
        # draw the data
        x = torch.rand(D, N) * 10000
        x = x.sort(dim=1)[0]

        y = torch.linspace(0, 1000, D*N).view(D, -1)
        y -= y[:, 0, None]

        xnew = torch.rand(Dnew, P)*10000

        print('Solving %d interpolation problems: '
              'each with %d observations and %d desired values' % (D, N, P))

        # calling the cpu version
        t0_cpu = time.time()
        yq_cpu = Interp1d()(x, y, xnew, yq_cpu)
        t1_cpu = time.time()

        display_str = 'CPU: %0.3fms, ' % ((t1_cpu-t0_cpu)*1000)

        if torch.cuda.is_available():
            x = x.to('cuda')
            y = y.to('cuda')
            xnew = xnew.to('cuda')

            # launching the cuda version
            t0 = time.time()
            yq_gpu = Interp1d()(x, y, xnew, yq_gpu)
            t1 = time.time()

            # compute the difference between both
            error = torch.norm(
                yq_cpu - yq_gpu.to('cpu'))/torch.norm(yq_cpu)*100.

            display_str += 'GPU: %0.3fms, error: %f%%.' % (
                (t1-t0)*1000, error)
        print(display_str)

    if torch.cuda.is_available():
        # for the last test, plot the result for the first 10 dimensions max
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
