## Building this for MSP430

* git clone ssh://git@github.com/EMCLab-Sinica/Tools.git Tools
* git clone ssh://git@github.com/EMCLab-Sinica/driverlib-msp430.git driverlib
* git clone ssh://git@github.com/EMCLab-Sinica/stateful-cnn.git intermittent-cnn
* Run `git submodule update --init --recursive` and `./transform.py mnist` in intermittent-cnn

And then import this project into CCStudio.
