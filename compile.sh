gcc -g -fopenmp -fPIC -shared -I$HOME/.mujoco/mujoco200/include/ -L$HOME/.mujoco/mujoco200/bin/ mujoco_derivative.c -lmujoco200 -lglew -lGL -lgomp -lm -o mujoco_derivative.so
gcc -g -fopenmp -I$HOME/.mujoco/mujoco200/include/ -L$HOME/.mujoco/mujoco200/bin/ mujoco_derivative.c -lmujoco200 -lglew -lGL -lgomp -lm -o mujoco_derivative.exe
gcc -g -fopenmp -I$HOME/.mujoco/mujoco200/include/ -L$HOME/.mujoco/mujoco200/bin/ mujoco_deriv_struct.c -lmujoco200 -lglew -lGL -lgomp -lm -o mujoco_deriv_struct.exe
