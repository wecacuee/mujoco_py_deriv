# cython: language_level=3

cdef extern from "mujoco.h" nogil:
    ctypedef double mjtNum

from mujoco_py.pxd.mujoco cimport mjModel, mjData, mjtStage


# cdef extern from "mujoco_derivative.c" nogil:
#     mjtNum* alloc_deriv(const mjModel* m);
#     void compute_derivative(const mjModel* m, const mjData* dmain, mjData* d, mjtNum*  deriv, int isforward, int nwarmup, double eps, int id, int nthread)
#     void compute_derivative_mp(const mjModel* m, const mjData* dmain, mjtNum*  deriv, int isforward, int nwarmup, double eps, int nthread)
#     void checkderiv(const mjModel* m, mjtNum* deriv, mjtNum error[8])

cdef extern from "mujoco_deriv_struct.h" nogil:
    mjtNum* mjd_get_ctrl        (const mjData* d)
    mjtNum* mjd_get_qpos        (const mjData* d)
    mjtNum* mjd_get_qvel        (const mjData* d)
    mjtNum* mjd_get_qacc        (const mjData* d)
    mjtNum* mjd_get_qfrc_inverse(const mjData* d)
    mjtNum* mjd_get_qfrc_applied(const mjData* d)

    ctypedef mjtNum* mjDGetter(const mjData* d)
    ctypedef struct mjDGetters:
        mjDGetter** fs
        size_t len

    ctypedef void mj_warmup_t(const mjModel* m, mjData* d, int nwarmup)

    ctypedef struct mjDeriv:
        void (*compute_mp)(mjDeriv* self)
        void (*free)(mjDeriv* self)
        mjtNum* deriv;

    ctypedef void (*perturb_t)(mjDeriv* deriv, int xk, int threadid, int n)

    ctypedef enum MJDGetter_enum:
        MJDGetter_ctrl, MJDGetter_qpos, MJDGetter_qvel, MJDGetter_qacc,
        MJDGetter_qfrc_inverse, MJDGetter_qfrc_applied

    mjDGetters* mjDGetters_new(size_t len, MJDGetter_enum* attrs)
    mjDGetters* mjDGetters_free(mjDGetters* g)

    mjDeriv* mjDeriv_new(const mjModel* m, const mjData* dmain, mjtNum* deriv,
                         mjDGetters* fs, mjDGetters* xs, perturb_t* perturbers,
                         mjtStage* stages,
                         mj_warmup_t* warmer, int nwarmup, double eps, int nthread)
    void mjDeriv_free(mjDeriv* mjderiv)

    mjDeriv* mjDeriv_new_default(const mjModel* m, const mjData* dmain, mjtNum* deriv,
                                 int isforward, int nwarmup, double eps, int nthread)
    void mjDeriv_free_default(mjDeriv* mjderiv)
    void mj_warmup_fwd(const mjModel* m, mjData* d, int nwarmup)

    void mj_warmup_inv(const mjModel* m, mjData* d, int nwarmup)
    perturb_t* mjPerturb_new(size_t len, MJDGetter_enum* attrs, int isforward);
    void mjPerturb_free(perturb_t* fun);

    mjtStage* mjStages_new(size_t len, MJDGetter_enum* attrs)
    void mjStages_free(mjtStage* fun);
    void checkderiv(const mjModel* m, mjtNum* deriv, mjtNum error[8])
