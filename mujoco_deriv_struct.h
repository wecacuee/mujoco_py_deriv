#ifndef MUJOCO_DERIV_STRUCT_H
#define MUJOCO_DERIV_STRUCT_H

#include "mujoco.h"

typedef enum { MJDGetter_ctrl, MJDGetter_qpos, MJDGetter_qvel, MJDGetter_qacc,
                      MJDGetter_qfrc_inverse, MJDGetter_qfrc_applied
} MJDGetter_enum;

#define MJDATA_GETTER_DECL(attr)                     \
  mjtNum* mjd_get_ ## attr(const mjData* d);

MJDATA_GETTER_DECL(ctrl)
MJDATA_GETTER_DECL(qpos)
MJDATA_GETTER_DECL(qvel)
MJDATA_GETTER_DECL(qacc)
MJDATA_GETTER_DECL(qfrc_inverse)
MJDATA_GETTER_DECL(qfrc_applied)

typedef mjtNum* mjDGetter(const mjData* d);

typedef struct _mjDGetters {
  size_t len;
  void (*free)(struct _mjDGetters* g);
  mjDGetter** fs;
} mjDGetters;


typedef void mj_warmup_t(const mjModel* m, mjData* d, int nwarmup);


typedef void mj_moveSkip(const mjModel* m,
                         mjData* d,
                         mjtStage stage,
                         int n,
                         mjtNum* warmstart);


typedef struct _mjDeriv {
  // Need from user
  const mjModel* m;
  const mjData* dmain;
  mjtNum* deriv;
  mjDGetters* fs;
  mjDGetters* xs;
  mjtStage* stages;
  double eps;
  int nthread;

  // Need from user
  int nwarmup;

  // Array of pertubers same size as xs
  // TODO make xs part of perturbers
  // NOTE should be exactly same as &perturb_t
  void (**perturbers)(struct _mjDeriv* deriv, int xk, int threadid, int i);
  mj_warmup_t* warmer;

  // Methods
  void (*compute_mp)(struct _mjDeriv* self);
  void (*compute)(struct _mjDeriv* self, int threadid);
  void (*free)(struct _mjDeriv* self);

  // Intermediate state
  mjData** d;
  mjtNum* warmstart;
} mjDeriv;

// NOTE should be exactly same as *mjDeriv.perturbers
typedef void (*perturb_t)(struct _mjDeriv* deriv, int xk, int threadid, int i);

void mjDGetters_free(mjDGetters* g);

mjDGetters* mjDGetters_new(size_t len, MJDGetter_enum* attrs);

perturb_t* mjPerturb_new(size_t len, MJDGetter_enum* attrs, int isforward);
void mjPerturb_free(perturb_t* fun);

mjtStage* mjStages_new(size_t len, MJDGetter_enum* attrs);
void mjStages_free(mjtStage* fun);

mjDeriv* mjDeriv_new(const mjModel* m, const mjData* dmain, mjtNum* deriv,
                     mjDGetters* fs, mjDGetters* xs, perturb_t* perturbers,
                     mjtStage* stages,
                     mj_warmup_t* warmer, int nwarmup, double eps, int nthread);
void mjDeriv_compute_mp(mjDeriv* deriv);
void mjDeriv_free(mjDeriv* mjderiv);

mjDeriv* mjDeriv_new_default(const mjModel* m, const mjData* dmain, mjtNum* deriv,
                          int isforward, int nwarmup, double eps, int nthread);
void mjDeriv_free_default(mjDeriv* mjderiv);

// mj_warmup_t
void mj_warmup_fwd(const mjModel* m, mjData* d, int nwarmup) ;

// mj_warmup_t
void mj_warmup_inv(const mjModel* m, mjData* d, int nwarmup) ;

void checkderiv(const mjModel* m, mjtNum* deriv, mjtNum error[8]);
#endif // MUJOCO_DERIV_STRUCT_H
