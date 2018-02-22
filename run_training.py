###################################################
#
#   Script to launch the training
#
##################################################

import os, sys
import ConfigParser


#config file to read from
config = ConfigParser.RawConfigParser()
config.readfp(open(r'./configuration.txt'))
#===========================================
#name of the experiment
name_experiment = config.get('experiment name', 'name')
nohup = config.getboolean('training settings', 'nohup')   #std output on log file?

#run_GPU = '' if sys.platform == 'win32' else ' KERAS_BACKEND=theano THEANO_FLAGS=device=gpu,floatX=float32 PATH=/usr/local/cuda/bin/:$PATH '
run_GPU = '' # use tensorflow

#create a folder for the results
result_dir = name_experiment
print "\n1. Create directory for the results (if not already existing)"
if os.path.exists(result_dir):
    print "Dir already existing"
elif sys.platform=='win32':
    os.system('mkdir ' + result_dir)
else:
    os.system('mkdir -p ' +result_dir)

print "copy the configuration file in the results folder"
if sys.platform=='win32':
    os.system('copy configuration.txt .\\' +name_experiment+'\\'+name_experiment+'_configuration.txt')
else:
    os.system('cp configuration.txt ./' +name_experiment+'/'+name_experiment+'_configuration.txt')

# pythonCmd = '/home/lgianca/anaconda2/envs/maya/bin/python'
pythonCmd = '/home/lgianca/anaconda2/bin/python'

# run the experiment
if nohup:
    print "\n2. Run the training on GPU with nohup"
    os.system(run_GPU +' nohup '+pythonCmd+' -u ./src/retinaNN_training.py > ' +'./'+name_experiment+'/'+name_experiment+'_training.nohup')
else:
    print "\n2. Run the training on GPU (no nohup)"
    os.system(run_GPU +' '+pythonCmd+' ./src/retinaNN_training.py')

#Prediction/testing is run with a different script
