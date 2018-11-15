# !/usr/bin/bash

# -W arg : warning control; arg is action:message:category:module:lineno, also PYTHONWARNINGS=arg
# -m mod : run library module as a script (terminates option list)

# -j, job dir
# -i, image dir
JOB_DIR=''
IMG_DIR=''
python -W ignore -m trainer.train -j $JOB_DIR -i $IMG_DIR

