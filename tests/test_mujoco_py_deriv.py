import os.path as op
from multiprocessing import cpu_count, Pool
import time

import numpy as np
from kwvars import kwvars, expand_variations
import mujoco_py as mj
from keyword2cmdline import command, func_kwonlydefaults

from mujoco_py_deriv import MjDerivative, checkderiv


def relpath(f, thisfile=__file__):
    return (f
            if op.isabs(f)
            else op.join(op.dirname(thisfile) or ".", f))

accuracy = [
    "G2*F2 - I ",
    "G2 - G2'  ",
    "G1 - G1'  ",
    "F2 - F2'  ",
    "G1 + G2*F1",
    "G0 + G2*F0",
    "F1 + F2*G1",
    "F0 + F2*G0"
]

@command
def main(path="flat_pusher_sample.xml",
         nepoch=20, nstep=500, niter=30, nwarmup=3, eps=1e-6,
         nthread=cpu_count()):
    print("nthread : %d" % nthread)
    print("niter   : %d" % niter)
    print("nwarmup : %d" % nwarmup)
    print("nepoch  : %d" % nepoch)
    print("nstep   : %d" % nstep)
    print("eps     : %g" % eps)
    model = mj.load_model_from_path(relpath(path))
    sim = mj.MjSim(model, nsubsteps=nstep)
    dmain = sim.data

    fs, xs = ["qacc"], ["qfrc_applied", "qvel", "qpos"]
    deriv_fwd = MjDerivative(model, dmain, fs, xs, isforward=1,
                               nwarmup=nwarmup, eps=eps, nthread=nthread,
                               niter=niter)
    fs, xs = ["qfrc_inverse"], ["qacc", "qvel", "qpos"]
    deriv_inv = MjDerivative(model, dmain, fs, xs, isforward=0,
                               nwarmup=nwarmup, eps=eps, nthread=nthread,
                               niter=niter)
    # allocate statistics
    nefc = 0
    deriv = np.zeros((6, model.nv, model.nv), dtype=np.float64)
    cputm = np.zeros((nepoch,2), dtype=np.float64)
    error = np.zeros((nepoch,8), dtype=np.float64)
    for epoch in range(nepoch):

        # main solution
        sim.step(with_udd=False)

        # count number of active constraints
        nefc += dmain.nefc;

        # start timer
        starttm = time.time()

        # test forward and inverse
        for isforward in range(2):
            if isforward:
                npderiv = deriv[np.newaxis, 3:, :, :]
                deriv_fwd.compute(npderiv)
            else:
                npderiv = deriv[np.newaxis, :3, :, :]
                deriv_inv.compute(npderiv)
            # record duration in ms
            cputm[epoch, isforward] = 1000 * (time.time() - starttm)

        nv = model.nv
        error[epoch, :] = checkderiv(model, deriv)

    # compute statistics
    mcputm = cputm.sum(axis=0)
    merror = error.sum(axis=0)

    # print sizes, timing, accuracy
    print("sizes   : nv %d, nefc %d\n" % ( model.nv, nefc/nepoch))
    print("inverse : %.2f ms" % (mcputm[0]/nepoch))
    print("forward : %.2f ms\n" % (mcputm[1]/nepoch))
    print("accuracy: log10(residual L1 relnorm)")
    print("------------------------------------")
    for ie in range(merror.shape[0]):
        print("  %s : %.2g" % ( accuracy[ie], merror[ie]/nepoch))
    print()
    return 0


if __name__ == '__main__':
    main()
