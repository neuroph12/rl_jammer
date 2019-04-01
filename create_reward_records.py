import write_record as wr, glob,pickle,os, numpy as np
from constants import *
from subprocess import *
 # Loads all data from pkl directory
 # removes examples so classes are balanced
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--only_count', action='store', default=True)
parser.add_argument('--even_out', action='store', default=True)
parser.add_argument('--protocol', action='store', default="both")
args = parser.parse_args()
only_count = int(args.only_count)
even_out = int(args.even_out)
 # removes all data files and writes single pkl to the directory

all_pkl_files = glob.glob(os.path.join(PKL_DIRECTORY,"*.pkl"))

all_data = None
for pkl_file in all_pkl_files:
	if args.protocol != "both" and args.protocol not in pkl_file:
		continue
	data = pickle.load(open(pkl_file, 'rb'))
	if all_data is None:
		all_data = data
		continue
	for i in range(n_classes):
		try:
			all_data[i,0]
		except KeyError:
			all_data[i,0] = []
		try:
			for el in data[i,0]:
				all_data[i,0].append(el)
		except KeyError:
			continue
lens = [len(all_data[i,0]) for i in range(n_classes)]
print("Lengths of classes:")
print(lens)
if only_count:
	exit(0)
if even_out:
	print("Evening out classes so all of them have length %d"%np.min(lens))
	for i in range(n_classes):
		try:
			r_inds = np.random.choice(np.arange(len(all_data[i,0])), size=np.min([lens[0], lens[2], lens[3], lens[4], lens[5]]), replace=False)
			all_data[i,0] = [all_data[i,0][j] for j in r_inds]
		except:
			continue

with open("%s/all_data%s.pkl"%(PKL_DIRECTORY,args.protocol), 'wb') as f:
	pickle.dump(all_data, f)
wr.write_split(PKL_DIRECTORY,TFRECORDS_DIRECTORY,specific_files=["all_data%s.pkl"%args.protocol])
