// -*- evil-shift-width: 4 -*-
/*  Copyright © 2018, Roboti LLC

    This file is licensed under the MuJoCo Resource License (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.roboti.us/resourcelicense.txt
*/


#include "mujoco.h"
#include <stdio.h>
#include <errno.h>        /* errno, perror */
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

#include "mujoco_deriv_struct.h"


// global variables: user-defined, with defaults
#define MAXTHREAD 64   // maximum number of threads allowed
#define MAXSTATEN 8 // max allowed state members

// enable compilation with and without OpenMP support
#if defined(_OPENMP)
    #include <omp.h>
#else
    // omp timer replacement
    double omp_get_wtime(void)
    {
        struct timeval start;
        struct timezone tz;
        gettimeofday(&start, &tz);

        struct timeval end;
        gettimeofday(&end, &tz);
        return (double)start.tv_sec + 1e-6 * (double)start.tv_usec;
    }

    // omp functions used below
    void omp_set_dynamic(int x) {}
    void omp_set_num_threads(int x) {}
    int omp_get_num_procs(void) {return 1;}
#endif





mjtNum* alloc_deriv(const mjModel* m)
{
  // allocate derivatives
  return (mjtNum*) mju_malloc(6*sizeof(mjtNum)*m->nv*m->nv);
}

void mj_copyStateCtrlData(mjData* d, const mjModel* m, const mjData* dmain) {
  d->time = dmain->time;
  mju_copy(d->qpos, dmain->qpos, m->nq);
  mju_copy(d->qvel, dmain->qvel, m->nv);
  mju_copy(d->qacc, dmain->qacc, m->nv);
  mju_copy(d->qacc_warmstart, dmain->qacc_warmstart, m->nv);
  mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
  mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
  mju_copy(d->ctrl, dmain->ctrl, m->nu);
}

////////////////////////////////////////////////////////////////////////////////
// mjData Getters
////////////////////////////////////////////////////////////////////////////////

#define MJDATA_GET_PTR(attr) &mjd_get_ ## attr

#define MJDATA_GETTER(attr)                     \
  mjtNum* mjd_get_ ## attr(const mjData* d)       \
  {                                             \
    return d-> attr;                           \
  }

MJDATA_GETTER(ctrl)
MJDATA_GETTER(qpos)
MJDATA_GETTER(qvel)
MJDATA_GETTER(qacc)
MJDATA_GETTER(qfrc_inverse)
MJDATA_GETTER(qfrc_applied)


void mjDGetters_free(mjDGetters* g)
{
  mju_free(g->fs);
  mju_free(g);
}


mjDGetter* mjd_enum2getter(const MJDGetter_enum mjdg)
{
  switch (mjdg)
    {
    case MJDGetter_ctrl:
      return MJDATA_GET_PTR(ctrl);
    case MJDGetter_qpos:
      return MJDATA_GET_PTR(qpos);
    case MJDGetter_qvel:
      return MJDATA_GET_PTR(qvel);
    case MJDGetter_qacc:
      return MJDATA_GET_PTR(qacc);
    case MJDGetter_qfrc_inverse:
      return MJDATA_GET_PTR(qfrc_inverse);
    case MJDGetter_qfrc_applied:
      return MJDATA_GET_PTR(qfrc_applied);
    default:
      printf("Error: Bad enum %d", mjdg);
      exit(EXIT_FAILURE);
      break;
    }
}


char* mjd_getter2str(mjDGetter* mjdg)
{
  if (MJDATA_GET_PTR(ctrl) == mjdg)
    return "ctrl";
  else if (MJDATA_GET_PTR(qpos) == mjdg)
    return "qpos";
  else if (MJDATA_GET_PTR(qvel) == mjdg)
    return "qvel";
  else if (MJDATA_GET_PTR(qacc) == mjdg)
    return "qacc";
  else if (MJDATA_GET_PTR(qfrc_inverse) == mjdg)
    return "qfrc_inverse";
  else if (MJDATA_GET_PTR(qfrc_applied) == mjdg)
    return "qfrc_applied";
  else {
    printf("Error: Bad getter %p", (void*) mjdg);
    exit(EXIT_FAILURE);
  }
}

void mjDGetters_print(const mjDGetters* mjdg)
{
  for (size_t i = 0; i < mjdg->len; i++)
    printf("%s %s", i ? "," : "", mjd_getter2str(mjdg->fs[i]));
  printf("\n");
}

mjDGetters* mjDGetters_new(size_t len, MJDGetter_enum* attr)
{
  mjDGetters* g = (mjDGetters*) mju_malloc(sizeof(mjDGetters));
  g->len = len;
  g->fs = (mjDGetter**) mju_malloc(len * sizeof(mjDGetter*));
  for (size_t i = 0; i < len; i++)
    g->fs[i] = mjd_enum2getter(attr[i]);
  g->free = &mjDGetters_free;
  return g;
}


// mj_warmup_t
void mj_warmup_fwd(const mjModel* m, mjData* d, int nwarmup) {
  mj_forward(m, d);

  // extra solver iterations to improve warmstart (qacc) at center point
  for( int rep=1; rep<nwarmup; rep++ )
    mj_forwardSkip(m, d, mjSTAGE_VEL, 1);
}

// mj_warmup_t
void mj_warmup_inv(const mjModel* m, mjData* d, int nwarmup) {
  mj_inverse(m, d);
}

size_t mjDeriv_deriv_size(const mjModel* m, int xN, int fN)
{
  int nv = m->nv;
  return fN*xN*nv*nv;
}

// An implementation of mj_moveSkip
void mj_warmFowardSkip(const mjModel* m,
                       mjData* d,
                       mjtStage stage,
                       int n,
                       mjtNum* warmstart)
{
  mju_copy(d->qacc_warmstart, warmstart, m->nv);
  mj_forwardSkip(m, d, stage, n);
}

// An implementation of mj_moveSkip
void mj_invSkip(const mjModel* m,
                mjData* d,
                mjtStage stage,
                int n,
                mjtNum* warmstart)
{
  mj_inverseSkip(m, d, stage, n);
}

void perturb_fwd_inv(mjDeriv* deriv,
                     int xk,
                     int threadid,
                     int i,
                     mj_moveSkip move)
{
  mjtNum* warmstart = deriv->warmstart;
  mjtStage stage = deriv->stages[xk];

  mjtNum eps = deriv->eps;
  const mjModel* m = deriv->m;
  mjData* d = deriv->d[threadid];

  // save output for center point and warmstart (needed in forward only)
  mjtNum* target = deriv->xs->fs[xk](d);
  const mjtNum originali = deriv->xs->fs[xk](deriv->dmain)[i];

  // perturb selected target
  target[i] += eps;

  // move forward or backward by 
  (*move)(m, d, stage, 1, warmstart);

  // undo perturbation
  target[i] = originali;
}

// perturb_t interface
void perturb_fwd(mjDeriv* deriv,
                 int xk,
                 int threadid,
                 int i)
{
  perturb_fwd_inv(deriv, xk, threadid, i, &mj_warmFowardSkip);
}

// perturb_t interface
void perturb_inv(mjDeriv* deriv,
                 int xk,
                 int threadid,
                 int i)
{
  perturb_fwd_inv(deriv, xk, threadid, i, &mj_invSkip);
}

void perturb_fwd_inv_pos(mjDeriv* deriv,
                         int xk,
                         int threadid,
                         int i,
                         mj_moveSkip* move)
{
      mjtNum* warmstart = deriv->warmstart;
      mjtStage stage = deriv->stages[xk];

      mjtNum eps = deriv->eps;
      const mjModel* m = deriv->m;
      mjData* d = deriv->d[threadid];
      const mjData* dmain = deriv->dmain;

      // get joint id for this dof
      int jid = m->dof_jntid[i];

      // get quaternion address and dof position within quaternion (-1: not in quaternion)
      int quatadr = -1, dofpos = 0;
      if( m->jnt_type[jid]==mjJNT_BALL )
      {
          quatadr = m->jnt_qposadr[jid];
          dofpos = i - m->jnt_dofadr[jid];
      }
      else if( m->jnt_type[jid]==mjJNT_FREE && i>=m->jnt_dofadr[jid]+3 )
      {
          quatadr = m->jnt_qposadr[jid] + 3;
          dofpos = i - m->jnt_dofadr[jid] - 3;
      }

      // apply quaternion or simple perturbation
      if( quatadr>=0 )
      {
          mjtNum angvel[3] = {0,0,0};
          angvel[dofpos] = eps;
          mju_quatIntegrate(d->qpos+quatadr, angvel, 1);
      }
      else
          d->qpos[m->jnt_qposadr[jid] + i - m->jnt_dofadr[jid]] += eps;

      // evaluate dynamics, with center warmstart
      move(m, d, stage, 1, warmstart);

      // undo perturbation
      mju_copy(d->qpos, dmain->qpos, m->nq);
}

// perturb_t interface
void perturb_fwd_pos(mjDeriv* deriv,
                 int xk,
                 int threadid,
                 int i)
{
  perturb_fwd_inv_pos(deriv, xk, threadid, i, &mj_warmFowardSkip);
}

// perturb_t interface
void perturb_inv_pos(mjDeriv* deriv,
                     int xk,
                     int threadid,
                     int i)
{
  perturb_fwd_inv_pos(deriv, xk, threadid, i, &mj_invSkip);
}

perturb_t mjd_enum2perturber(MJDGetter_enum attr, int isforward)
{
  if (isforward) {
    if (MJDGetter_qpos == attr)
      return &perturb_fwd_pos;
    else
      return &perturb_fwd;
  } else {
    if (MJDGetter_qpos == attr)
      return &perturb_inv_pos;
    else
      return &perturb_inv;
  }
}

perturb_t* mjPerturb_new(size_t len, MJDGetter_enum* attrs, int isforward)
{
  perturb_t* perturbers = (perturb_t*) mju_malloc(len * sizeof(perturb_t));
  for (int i = 0; i < len; i++)
    perturbers[i] = mjd_enum2perturber(attrs[i], isforward);
  return perturbers;
}
void mjPerturb_free(perturb_t* perturbers)
{
  mju_free(perturbers);
}

mjtStage mjd_enum2stage(MJDGetter_enum attr)
{
  switch (attr)
    {
    case MJDGetter_qpos:
      return mjSTAGE_NONE;
    case MJDGetter_qvel:
      return mjSTAGE_POS;
    default:
      return mjSTAGE_VEL;
    }
}

mjtStage* mjStages_new(size_t len, MJDGetter_enum* attrs)
{
  mjtStage* stages = (mjtStage*) mju_malloc(len * sizeof(mjtStage));
  for (int i = 0; i < len; i++)
    stages[i] = mjd_enum2stage(attrs[i]);
  return stages;
}

void mjStages_print(mjtStage* stages, size_t len)
{
  for (size_t i = 0; i < len; i++)
    printf("%s%d", i ? ", " : "", stages[i]);
  printf("\n");
}

void mjStages_free(mjtStage* stages)
{
  mju_free(stages);
}


void mjDeriv_compute(mjDeriv* deriv, int threadid)
{
  mjData* d = deriv->d[threadid];
  const mjModel* m = deriv->m;
  const mjData* dmain = deriv->dmain;
  mjDGetters* fs = deriv->fs;
  mjDGetters* xs = deriv->xs;
  int nthread = deriv->nthread;
  mjtNum eps = deriv->eps;

  int nv = m->nv;

  // allocate stack space for result at center
  mjMARKSTACK
  mjtNum* center = mj_stackAlloc(d, nv);
  mjtNum* warmstart = mj_stackAlloc(d, nv);

  // prepare static schedule: range of derivative columns to be computed by this thread
  int chunk = (m->nv + nthread-1) / nthread;
  int istart = threadid * chunk;
  int iend = mjMIN(istart + chunk, m->nv);

  // copy state and control from dmain to thread-specific d
  mj_copyStateCtrlData(d, m, dmain);

  // run full computation at center point (usually faster than copying dmain)
  (*deriv->warmer)(m, d, deriv->nwarmup);

  for (int fk=0; fk < fs->len; fk++) {
    // select target vector and original vector for force or acceleration derivative
    mjtNum* output = fs->fs[fk](d); // d->qacc

    // save output for center point and warmstart (needed in forward only)
    mju_copy(center, output, nv);
    mju_copy(warmstart, d->qacc_warmstart, nv);
    for (int xk=0; xk < xs->len; xk++) {
      deriv->warmstart = warmstart;

      for( int i=istart; i<iend; i++ ) {
        (deriv->perturbers[xk])(deriv, xk, threadid, i);

        // compute column i of derivative 2
        for( int j=0; j<nv; j++ ) {
          size_t idx = (fk*xs->len + xs->len-xk-1)*nv*nv + j*nv + i;
          deriv->deriv[idx] = (output[j] - center[j])/eps;
        }
      }
    }
  }
  mjFREESTACK
}

void mjDeriv_print(mjDeriv* mjd)
{
  printf("mjd->m->nv: %d\n", mjd->m->nv);
  printf("mjd->dmain->nefc: %d\n", mjd->dmain->nefc);
  printf("mjd->deriv: %p\n", (void*) mjd->deriv);
  printf("mjd->fs: ");  mjDGetters_print(mjd->fs);
  printf("mjd->xs: ");  mjDGetters_print(mjd->xs);
  printf("mjd->stages:"); mjStages_print(mjd->stages, mjd->xs->len);
  printf("mjd->eps: %f\n", mjd->eps);
  printf("mjd->nthread: %d\n", mjd->nthread);
  printf("mjd->nwarmup: %d\n", mjd->nwarmup);
  printf("mjd->warmer: %s\n", mjd->warmer == mj_warmup_fwd ? "fwd" :
         (mjd->warmer == mj_warmup_inv ? "inv" : "error"));
}


void mjDeriv_compute_mp(mjDeriv* deriv)
{
  int nthread = deriv->nthread;
  const mjModel* m = deriv->m;

  // set up OpenMP (if not enabled, this does nothing)
  omp_set_dynamic(0);
  omp_set_num_threads(nthread);

  //mjData** d = deriv->d;
  mjData* d[MAXTHREAD];
  for( int n=0; n<nthread; n++ ) {
    mjData* dn = mj_makeData(m);
    d[n] = dn;
    if ( d[n] == NULL ) {
      perror("Unable to allocate memory\n");
      exit(EXIT_FAILURE);
    }
  }
  deriv->d = d;

  // run worker threads in parallel if OpenMP is enabled
  #pragma omp parallel for schedule(static)
  for( int n=0; n<nthread; n++ ) {
    (*deriv->compute)(deriv, n);
  }

  for( int n=0; n<nthread; n++ )
    mj_deleteData(d[n]);
}

void mjDeriv_free(mjDeriv* mjderiv)
{
  mju_free(mjderiv);
}

mjDeriv* mjDeriv_new(const mjModel* m, const mjData* dmain, mjtNum* deriv,
                     mjDGetters* fs, mjDGetters* xs, perturb_t* perturbers,
                     mjtStage* stages,
                     mj_warmup_t* warmer, int nwarmup, double eps, int nthread)
{
  mjDeriv* mjd = (mjDeriv*) mju_malloc(sizeof(mjDeriv));
  mjd->m = m;
  mjd->dmain = dmain;
  mjd->deriv = deriv;
  mjd->fs = fs;
  mjd->xs = fs;
  mjd->perturbers = perturbers;
  mjd->nwarmup  = nwarmup;
  mjd->warmer = warmer;
  mjd->xs = xs;
  mjd->perturbers = perturbers;
  mjd->stages = stages;

  mjd->nthread = nthread;
  // mjd->d = (mjData**) mju_malloc(nthread * sizeof(mjData*));
  mjd->eps = eps;

  mjd->compute = &mjDeriv_compute;
  mjd->compute_mp = &mjDeriv_compute_mp;
  mjd->free = &mjDeriv_free;
  return mjd;
}


void mjDeriv_free_default(mjDeriv* mjderiv)
{
  // mju_free(mjderiv->d);
  mju_free(mjderiv->stages);
  mju_free(mjderiv->perturbers);
  (*mjderiv->xs->free)(mjderiv->xs);
  (*mjderiv->fs->free)(mjderiv->fs);
  mju_free(mjderiv);
}


mjDeriv* mjDeriv_new_default(const mjModel* m, const mjData* dmain, mjtNum* deriv,
                          int isforward, int nwarmup, double eps, int nthread)
{
  size_t fN = 1;
  size_t xN = 3;
  mjDeriv* mjd = (mjDeriv*) mju_malloc(sizeof(mjDeriv));
  mjd->m = m;
  mjd->dmain = dmain;
  mjd->deriv = deriv;
  MJDGetter_enum fs_attr[1] = { MJDGetter_qacc };
  if (isforward) {
      fs_attr[0] = MJDGetter_qacc ;
  } else {
      fs_attr[0] = MJDGetter_qfrc_inverse;
  }
  mjDGetters* fs = mjDGetters_new(fN, fs_attr);
  mjd->fs = fs;

  MJDGetter_enum xs_attr[3] = { MJDGetter_qfrc_applied, MJDGetter_qvel, MJDGetter_qpos };
  if (isforward) {
      xs_attr[0] = MJDGetter_qfrc_applied;
  } else {
      xs_attr[0] = MJDGetter_qacc;
  }
  mjDGetters* xs = mjDGetters_new(xN, xs_attr);
  perturb_t* perturbers = (perturb_t*) mju_malloc(xN * sizeof(perturb_t));
  if (isforward) {
    perturbers[0] = &perturb_fwd;
    perturbers[1] = &perturb_fwd;
    perturbers[2] = &perturb_fwd_pos;
    mjd->nwarmup  = nwarmup;
    mjd->warmer   = &mj_warmup_fwd;
  } else {
    perturbers[0] = &perturb_inv;
    perturbers[1] = &perturb_inv;
    perturbers[2] = &perturb_inv_pos;
    mjd->nwarmup  = 0;
    mjd->warmer   = &mj_warmup_inv;
  }
  mjd->xs = xs;
  mjd->perturbers = perturbers;

  mjd->stages = (mjtStage*) mju_malloc(xN * sizeof(mjtStage));
  mjd->stages[0] = mjSTAGE_VEL;
  mjd->stages[1] = mjSTAGE_POS;
  mjd->stages[2] = mjSTAGE_NONE;

  mjd->nthread = nthread;
  // mjd->d = (mjData**) mju_malloc(nthread * sizeof(mjData*));
  mjd->eps = eps;

  mjd->compute = &mjDeriv_compute;
  mjd->compute_mp = &mjDeriv_compute_mp;
  mjd->free = &mjDeriv_free_default;
  return mjd;
}

double relnorm(mjtNum* residual, mjtNum* base, int n)
{
  mjtNum L1res = 0, L1base = 0;
  for( int i=0; i<n; i++ )
    {
      L1res += mju_abs(residual[i]);
      L1base += mju_abs(base[i]);
    }

  return (double) mju_log10(mju_max(mjMINVAL,L1res/mju_max(mjMINVAL,L1base)));
}

// names of residuals for accuracy check
const char* accuracy[8] = {
    "G2*F2 - I ",
    "G2 - G2'  ",
    "G1 - G1'  ",
    "F2 - F2'  ",
    "G1 + G2*F1",
    "G0 + G2*F0",
    "F1 + F2*G1",
    "F0 + F2*G0"
};


// check accuracy of derivatives using known mathematical identities
void checkderiv(const mjModel* m, mjtNum* deriv, mjtNum error[8])
{
    mjData* d = mj_makeData(m);
    int nv = m->nv;

    // allocate space
    mjMARKSTACK
    mjtNum* mat = mj_stackAlloc(d, nv*nv);

    // get pointers to derivative matrices
    mjtNum* G0 = deriv;                 // dinv/dpos
    mjtNum* G1 = deriv + nv*nv;         // dinv/dvel
    mjtNum* G2 = deriv + 2*nv*nv;       // dinv/dacc = dqfrc_inverse / dacc
    mjtNum* F0 = deriv + 3*nv*nv;       // dacc/dpos
    mjtNum* F1 = deriv + 4*nv*nv;       // dacc/dvel
    mjtNum* F2 = deriv + 5*nv*nv;       // dacc/dfrc = dacc / dfrc_applied

    // G2*F2 - I
    mju_mulMatMat(mat, G2, F2, nv, nv, nv);
    for( int i=0; i<nv; i++ )
        mat[i*(nv+1)] -= 1;
    error[0] = relnorm(mat, G2, nv*nv);

    // G2 - G2'
    mju_transpose(mat, G2, nv, nv);
    mju_sub(mat, mat, G2, nv*nv);
    error[1] = relnorm(mat, G2, nv*nv);

    // G1 - G1'
    mju_transpose(mat, G1, nv, nv);
    mju_sub(mat, mat, G1, nv*nv);
    error[2] = relnorm(mat, G1, nv*nv);

    // F2 - F2'
    mju_transpose(mat, F2, nv, nv);
    mju_sub(mat, mat, F2, nv*nv);
    error[3] = relnorm(mat, F2, nv*nv);

    // G1 + G2*F1
    mju_mulMatMat(mat, G2, F1, nv, nv, nv);
    mju_addTo(mat, G1, nv*nv);
    error[4] = relnorm(mat, G1, nv*nv);

    // G0 + G2*F0
    mju_mulMatMat(mat, G2, F0, nv, nv, nv);
    mju_addTo(mat, G0, nv*nv);
    error[5] = relnorm(mat, G0, nv*nv);

    // F1 + F2*G1
    mju_mulMatMat(mat, F2, G1, nv, nv, nv);
    mju_addTo(mat, F1, nv*nv);
    error[6] = relnorm(mat, F1, nv*nv);

    // F0 + F2*G0
    mju_mulMatMat(mat, F2, G0, nv, nv, nv);
    mju_addTo(mat, F0, nv*nv);
    error[7] = relnorm(mat, F0, nv*nv);

    mjFREESTACK
    mj_deleteData(d);
}

int main(int argc, char** argv)
{
    // gloval variables: internal
    const int MAXEPOCH = 100;   // maximum number of epochs
    mjtNum* deriv = 0;          // dynamics derivatives (6*nv*nv):
                                //  dinv/dpos, dinv/dvel, dinv/dacc, dacc/dpos, dacc/dvel, dacc/dfrc
    int nthread = 0;
    int niter = 30;             // fixed number of solver iterations for finite-differencing
    int nwarmup = 3;            // center point repetitions to improve warmstart
    int nepoch = 20;            // number of timing epochs
    int nstep = 500;            // number of simulation steps per epoch
    double eps = 1e-6;          // finite-difference epsilon
    // print help if not enough arguments
    if( argc<2 )
    {
        printf("\n Arguments: modelfile [nthread niter nwarmup nepoch nstep eps]\n\n");
        return 1;
    }

    // default nthread = number of logical cores (usually optimal)
    nthread = omp_get_num_procs();

    // get numeric command-line arguments
    if( argc>2 )
        sscanf(argv[2], "%d", &nthread);
    if( argc>3 )
        sscanf(argv[3], "%d", &niter);
    if( argc>4 )
        sscanf(argv[4], "%d", &nwarmup);
    if( argc>5 )
        sscanf(argv[5], "%d", &nepoch);
    if( argc>6 )
        sscanf(argv[6], "%d", &nstep);
    if( argc>7 )
        sscanf(argv[7], "%lf", &eps);

    // check number of threads
    if( nthread<1 || nthread>MAXTHREAD )
    {
        printf("nthread must be between 1 and %d\n", MAXTHREAD);
        return 1;
    }

    // check number of epochs
    if( nepoch<1 || nepoch>MAXEPOCH )
    {
        printf("nepoch must be between 1 and %d\n", MAXEPOCH);
        return 1;
    }

    // activate and load model
    mj_activate("mjkey.txt");
    mjModel* m = 0;
    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        m = mj_loadModel(argv[1], NULL);
    else
        m = mj_loadXML(argv[1], NULL, NULL, 0);
    if( !m )
    {
        printf("Could not load modelfile '%s'\n", argv[1]);
        return 1;
    }

    // print arguments
#if defined(_OPENMP)
    printf("\nnthread : %d (OpenMP)\n", nthread);
#else
    printf("\nnthread : %d (serial)\n", nthread);
#endif
    printf("niter   : %d\n", niter);
    printf("nwarmup : %d\n", nwarmup);
    printf("nepoch  : %d\n", nepoch);
    printf("nstep   : %d\n", nstep);
    printf("eps     : %g\n\n", eps);

    // make mjData: main, per-thread
    mjData* dmain = mj_makeData(m);

    int nv = m->nv;
    deriv = alloc_deriv(m);
    mjDeriv* mjderiv_inv = mjDeriv_new_default(m, dmain, deriv, 0, nwarmup, eps, nthread);
    mjDeriv* mjderiv_fwd = mjDeriv_new_default(m, dmain, deriv + 3*nv*nv, 1, nwarmup, eps, nthread);
    mjDeriv* mj_derivs[2] = {mjderiv_inv, mjderiv_fwd};

    // save solver options
    int save_iterations = m->opt.iterations;
    mjtNum save_tolerance = m->opt.tolerance;

    // allocate statistics
    int nefc = 0;
    double cputm[MAXEPOCH][2];
    mjtNum error[MAXEPOCH][8];

    // run epochs, collect statistics
    for( int epoch=0; epoch<nepoch; epoch++ )
    {
        // set solver options for main simulation
        m->opt.iterations = save_iterations;
        m->opt.tolerance = save_tolerance;

        // advance main simulation for nstep
        for( int i=0; i<nstep; i++ )
            mj_step(m, dmain);

        // count number of active constraints
        nefc += dmain->nefc;

        // set solver options for finite differences
        m->opt.iterations = niter;
        m->opt.tolerance = 0;

        // start timer
        double starttm = omp_get_wtime();

        // test forward and inverse
        for( int isforward=0; isforward<2; isforward++ ) {
            mjDeriv* mjderiv = mj_derivs[isforward];
            (*mjderiv->compute_mp)(mjderiv);
            // record duration in ms
            cputm[epoch][isforward] = 1000*(omp_get_wtime() - starttm);
        }

        // check derivatives
        checkderiv(m, deriv, error[epoch]);
    }

    // compute statistics
    double mcputm[2] = {0,0}, merror[8] = {0,0,0,0,0,0,0,0};
    for( int epoch=0; epoch<nepoch; epoch++ )
    {
        mcputm[0] += cputm[epoch][0];
        mcputm[1] += cputm[epoch][1];

        for( int ie=0; ie<8; ie++ )
            merror[ie] += error[epoch][ie];
    }

    // print sizes, timing, accuracy
    printf("sizes   : nv %d, nefc %d\n\n", m->nv, nefc/nepoch);
    printf("inverse : %.2f ms\n", mcputm[0]/nepoch);
    printf("forward : %.2f ms\n\n", mcputm[1]/nepoch);
    printf("accuracy: log10(residual L1 relnorm)\n");
    printf("------------------------------------\n");
    for( int ie=0; ie<8; ie++ )
        printf("  %s : %.2g\n", accuracy[ie], merror[ie]/nepoch);
    printf("\n");

    // shut down
    for (int isforward = 0; isforward < 2; isforward++) {
      mjDeriv* mjderiv = mj_derivs[isforward];
      (*mjderiv->free)(mjderiv);
    }
    mju_free(deriv);
    mj_deleteData(dmain);
    mj_deleteModel(m);
    mj_deactivate();
    return 0;
}

