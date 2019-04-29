# cython: language_level=3
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from mujoco_py.pxd.mujoco cimport mjModel, mjData, mju_printMat, mjtNum, mju_malloc, mju_free
cimport cmujoco_jac_struct as cmj
from libc.stdint cimport uintptr_t
from libc.stdio cimport printf

cimport numpy as np
import numpy as np
from tempfile import TemporaryDirectory
import mujoco_py.cymj as cymj

def checkderiv(model,
               np.ndarray[double, ndim=3, mode="c"] deriv,
               np.ndarray[double, ndim=1, mode="c"] error = None):
    assert isinstance(model, cymj.PyMjModel)
    assert 6 == deriv.shape[0]
    assert model.nv == deriv.shape[1]
    assert model.nv == deriv.shape[2]
    if error is None:
        error = np.empty((8,), dtype=np.float64)
    else:
        assert error.shape[0] == 8

    cdef uintptr_t model_uintptr = model.uintptr
    cdef mjModel* cm = <mjModel*>model_uintptr
    cdef cmj.mjtNum* cerror = <cmj.mjtNum*>error.data
    cdef cmj.mjtNum* cderiv = <cmj.mjtNum*>deriv.data

    cmj.checkderiv(cm, cderiv, cerror)
    return error

cdef cmj.MJDGetter_enum* _enum_array(
    attrs,
    mapping=dict(ctrl=cmj.MJDGetter_ctrl,
                 qpos=cmj.MJDGetter_qpos,
                 qvel=cmj.MJDGetter_qvel,
                 qacc=cmj.MJDGetter_qacc,
                 qfrc_inverse=cmj.MJDGetter_qfrc_inverse,
                 qfrc_applied=cmj.MJDGetter_qfrc_applied)):
    cdef cmj.MJDGetter_enum* c_attrs = <cmj.MJDGetter_enum*>PyMem_Malloc(
        len(attrs)*sizeof(cmj.MJDGetter_enum));
    cdef size_t i = 0
    for i in range(len(attrs)):
        c_attrs[i] = mapping[attrs[i]]
    return c_attrs

cdef cmj.mjDGetters* mjDGetters_new(attrs):
    cdef cmj.MJDGetter_enum* c_attrs = _enum_array(attrs)
    cdef cmj.mjDGetters* ptr = cmj.mjDGetters_new(len(attrs), c_attrs)
    PyMem_Free(c_attrs)
    return ptr

cdef class PyMjGetters:
    cdef cmj.mjDGetters* ptr
    def __cinit__(self, *attrs):
        cdef cmj.MJDGetter_enum* c_attrs = _enum_array(attrs)
        self.ptr = cmj.mjDGetters_new(len(attrs), c_attrs)
        mju_free(c_attrs)

    @property
    def ptr(self):
        return <uintptr_t>self.ptr

    @property
    def len(self):
        return self.ptr.len

    def __destroy__(self):
        cmj.mjDGetters_free(self.ptr)


cdef class PyMjDeriv:
    cdef cmj.mjDeriv* ptr
    cdef mjModel* model
    cdef mjData* data
    cdef cmj.mjDGetters* fs
    cdef cmj.mjDGetters* xs
    cdef cmj.perturb_t* perturbers
    cdef cmj.mjtStage* stages
    cdef cmj.mj_warmup_t* warmer
    def __cinit__(self, model, data, fs, xs,
                  int isforward = 1,
                  int nwarmup = 3, double eps = 1e-6, int nthread = 1):
        assert isinstance(model, cymj.PyMjModel)
        assert isinstance(data, cymj.PyMjData)
        cdef uintptr_t model_uintptr = model.uintptr
        self.model = <mjModel*>model_uintptr
        cdef uintptr_t data_uintptr = data.uintptr
        self.data = <mjData*>data_uintptr
        self.fs = mjDGetters_new(fs)
        cdef cmj.MJDGetter_enum* x_attrs = _enum_array(xs)
        self.xs = cmj.mjDGetters_new(len(xs), x_attrs)
        self.perturbers = cmj.mjPerturb_new(self.xs.len, x_attrs, isforward)
        self.stages = cmj.mjStages_new(self.xs.len, x_attrs)
        PyMem_Free(x_attrs)
        if isforward:
            self.warmer = &cmj.mj_warmup_fwd
        else:
            self.warmer = &cmj.mj_warmup_inv
        self.ptr = cmj.mjDeriv_new(self.model, self.data, <mjtNum*>0,
                                   self.fs,
                                   self.xs,
                                   self.perturbers, self.stages, self.warmer,
                                   nwarmup, eps, nthread)
        #self.ptr = cmj.mjDeriv_new_default(self.model, self.data, <mjtNum*>0,
        #                                   isforward, nwarmup, eps, nthread)

    def deriv_shape(self):
        return (self.fs.len, self.xs.len, self.model.nv, self.model.nv)

    def compute(self, np.ndarray[np.float64_t, ndim=4, mode='c'] npderiv = None):
        cdef cmj.mjDeriv* mjd = self.ptr
        if npderiv is None:
            npderiv = np.empty((self.fs.len, self.xs.len,
                                self.model.nv, self.model.nv),
                               dtype=np.float64)
        else:
            assert npderiv.shape[0] == self.fs.len
            assert npderiv.shape[1] == self.xs.len
            assert npderiv.shape[2] == self.model.nv
            assert npderiv.shape[3] == self.model.nv
        mjd.deriv = <mjtNum*>npderiv.data


        mjd.compute_mp(mjd)

        return npderiv

    @property
    def ptr(self):
        return <uintptr_t>self.ptr

    @property
    def perturbers(self):
        return <uintptr_t>self.perturbers

    @property
    def stages(self):
        return <uintptr_t>self.stages

    @property
    def warmer(self):
        return <uintptr_t>self.warmer

    def xslen(self):
        return self.xs.len

    def fslen(self):
        return self.fs.len

    def __destroy__(self):
        cmj.mjDGetters_free(self.fs)
        cmj.mjDGetters_free(self.xs)
        cmj.mjPerturb_free(self.perturbers)
        cmj.mjStages_free(self.stages)
        self.ptr.free(self.ptr)

class MjDerivative:
    def __init__(self, model, data, *args, **kwargs):
        """
        Parameters:

            model: `mujoco_py.PyMjModel`
            data : `mujoco_py.PyMjData`
            fs   : list of str   : `mujoco_py.PyMjData` attributes whose
                                     derivative is needed
            xs   : list of str   : `mujoco_py.PyMjData` attributes w.r.t.
                                 derivative is needed

        Optional:

            isforward : {0, 1}  : Forward derivative or inverse?
            nwarmup   : int=3   : Number of iterations for warm start
            eps       : float=1e-6: Step size for numerical differentiation
            nthread   : int=1   : Open mp threads
            niter     : int=30  : Max iterations for the optimizer
        """
        self.pymodel = model
        self.niter = kwargs.pop("niter", 30)
        self.ext = PyMjDeriv(model, data, *args, **kwargs)

    def compute(self, npderiv=None):
        """
        Optional Parameters:

            npderiv: 4D numpy array of shape : len(fs) * len(xs) * model.nv * model.nv

        Returns:

            npderiv: 4D numpy array of shape : len(fs) * len(xs) * model.nv * model.nv
        """
        # save solver options
        save_iterations = self.pymodel.opt.iterations;
        save_tolerance = self.pymodel.opt.tolerance;

        # set solver options for finite differences
        self.pymodel.opt.iterations = self.niter;
        self.pymodel.opt.tolerance = 0;

        # Main numerical differentiation
        deriv = self.ext.compute(npderiv)

        # set solver options for main simulation
        self.pymodel.opt.iterations = save_iterations;
        self.pymodel.opt.tolerance = save_tolerance;
        return deriv
