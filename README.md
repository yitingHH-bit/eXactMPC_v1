# eXactMPC

https://github.com/xuexianlim/eXactMPC

# C++
This program implements kinematic MPC for the eXact electric excavator arm. Casadi and Ipopt (already included in Casadi) are required for it to run. The installation steps can be found here: https://github.com/casadi/casadi/wiki/InstallationLinux

To build this repository, run the following commands:
```
$ cd build
$ cmake ..
$ make
```

To run the program, run the following command:
```
$ ./eXactMPC
```

It prints the motor velocity commands for a single MPC time step.

# Python
This program runs a simulation of the excavator arm using the controller. It solves the optimisation problem each time step and calculates the excavator arm response before repeating the process. It therefore takes a while longer to finish outputting the visualisation plots.

The file to run is `MPCSimulation.py`. At least Python 3.10 is required to run the program due to the presence of match-case syntax.

In order to generate videos from the visualisation module, OpenCV should be installed via the following command:
```
$ sudo apt-get install python3-opencv
```

Using pip3 to install opencv-python might not work due to the absence of the necessary codec for encoding the video.

# MATLAB
This program contains several useful functions for the excavator arm model which the kinematic MPC is built upon. The code runs as it is. The file to run is `simulation.m`.


Ideas for RL updates:- +60/-60 log spawn is good! But maybe add random rog rotation around its Z origin (yaw)?- add win condition for faster training (env. reset (win) when log is above x meters).Final evaluation- run trained model in play mode (...scripts/reinforcement_learning/rl_games/play.py --task=...) to get smooth movement. Collect data from the play runs to get the final accuracy and performance metrics!![Uploading image.png…]()




















