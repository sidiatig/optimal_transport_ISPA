from os import system
import os


code_dir = os.path.split(os.getcwd())[0]

experiments = ['fmril']
data_methods = ['indi'] #  'no'
clf_methods = ['logis']
cv_starts = [0]
nrounds = [i for i in range(11,31)]


for experiment in experiments:
    for data_method in data_methods:
        for clf_method in clf_methods:
            for cv_start in cv_starts:
                cmd = "frioul_batch -c 3 " \
                      "'/hpc/crise/anaconda3/bin/python3.5 " \
                      "%s/frioul_ot_sinkhorn_knopp.py %s %s %s %d'" \
                      % (code_dir, experiment, data_method, clf_method, cv_start)

                # a = commands.getoutput(cmd)
                a = system(cmd)
                print(cmd)
