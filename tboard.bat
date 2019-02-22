@echo off

@rem Change the working directory to the location of this file so that relative paths will work
cd /D "%~dp0"

tensorboard --logdir=runs --host 127.0.0.1