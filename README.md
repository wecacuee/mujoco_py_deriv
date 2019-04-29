# Numerical derivatives of dynamics for mujoco

* Needs mujoco licence to run.
* Wraps [derivative.cpp](http://www.mujoco.org/book/programming.html#saDerivative) to call from Python.

## Installation

1. Install [mujoco_py](https://github.com/openai/mujoco-py/)
2. `pip install mujoco_py_deriv`

## Usage

Prepare mujoco model.

``` python
import mujoco_py as mj
from mujoco_py_deriv import MjDerivative, checkderiv

# Prepare mujoco model and data
model = mj.load_model_from_path("flat_pusher_sample.xml")
sim = mj.MjSim(model, nsubsteps=nstep)
dmain = sim.data

```

Compute numerical derivative

``` python
# To compute δf/δx
f = ["qacc"]
x = ["qfrc_applied", "qvel", "qpos"]
deriv_obj = MjDerivative(model, dmain, f, x)
deriv = deriv_obj.compute()
```


## Test case

``` shellsession
python test/test_mujoco_py_deriv.py
```

## Dynamics with derivatives

Take a look at [mujoco_py_deriv_dynamics.py](mujoco_py_deriv_dynamics.py)

