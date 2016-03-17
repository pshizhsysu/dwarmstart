#!/usr/bin/env python

import sys, subprocess, uuid, os, math, shutil

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print('usage: {0} machinefile svm_file [split_svm_file]'.format(sys.argv[0]))
    sys.exit(1)
machinefile_path, src_path = sys.argv[1:3] #machinefile_path==sys.argv[1],scr_path==sys_argv[2]
#print(sys.argv)
#sys.exit(1)

machines = set() #Here machine is an empty set.
#print(machines)
#sys.exit(1)
for line in open(machinefile_path):
    machine = line.strip()
    if machine in machines:
        print('Error: duplicated machine {0}'.format(machine))# check if the machines are duplicated
        sys.exit(1)
    machines.add(machine)
nr_machines = len(machines)

src_basename = os.path.basename(src_path)	# src_basename is "heart_scale" not "/home/jing/data/heart_scale"
if len(sys.argv) == 4:
    dst_path = sys.argv[3]
else:
    dst_path = '{0}.sub'.format(src_basename)

cmd = 'wc -l {0}'.format(src_path)
# wc -l is to count the  number of the lines, wc= word count

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
#p is a sub-process

nr_instances = int(p.stdout.read().strip().split()[0])
#print(nr_instances)
#sys.exit(1)

p.communicate()

while True:
    temp_dir = 'tmp_{0}'.format(uuid.uuid4())
    if not os.path.exists(temp_dir): break
os.mkdir(temp_dir)
#UUID (Universally Unique IDentifier)uuid4()random number
#print(temp_dir)
#sys.exit(1)

print('Spliting data...')
nr_digits = int(math.log10(nr_machines))+1
#print(nr_digits)
#sys.exit(1)

cmd = 'split -l {0} --numeric-suffixes -a {1} {2} {3}.'.format(
          int(math.ceil(float(nr_instances)/nr_machines)), nr_digits, src_path,
          os.path.join(temp_dir, src_basename))
p = subprocess.Popen(cmd, shell=True)
p.communicate()
#sys.exit(1);

for i, machine in enumerate(machines):
    temp_path = os.path.join(temp_dir, src_basename + '.' + 
                             str(i).zfill(nr_digits))
    #print(temp_path)
    if machine == '127.0.0.1' or machine == 'master':
        cmd = 'mv {0} {1}'.format(temp_path, os.path.join('/home/jing/dis_data/',dst_path))	# /home/jing/dis_data/ + /home/jing/dis_data/heart_scale.sub ?
    else:
        cmd = 'scp {0} {1}:{2}'.format(temp_path, machine,
                                       os.path.join('/home/jing/dis_data', dst_path))
               #print(os.getcwd())get current working directory

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    p.communicate()
    print('The subset of data has been copied to {0}'.format(machine))
shutil.rmtree(temp_dir)# remove the temp_dir
