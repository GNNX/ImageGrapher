import webcolors, datetime
import cv2
from copy import copy
from time import sleep
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys

args = sys.argv

# Defines parameters
n_regions = int(args[1]) #15000                - Number of cuts on minimum spanning tree
minimum_component_length = int(args[2]) #100   - Minimum size to be condered a region
def_len1 = int(args[3]) #5                     - Work applied on x axis to remove isolated edges
def_len2 = int(args[4]) #5                     - Work applied on y axis to remove isolated edges
resolut_len = int(args[5]) #2                  - Unresolution power
maximum_stdn_dev = int(args[6]) #30            - Maximum standard deviation of neighborhood so pixel is considered noise

def find_adj_nodes(x,y,a,b,len1=1,len2=1):
	minx,maxx,miny,maxy = x-len1,x+len1,y-len2,y+len2
	if x - len1 + 1 <= 0:
		minx = x
	if x + len1 - 1 >= a:
		maxx = x
	if y - len2 + 1 <= 0:
		miny = y
	if y + len2 - 1 >= b:
		maxy = y
	return [(i,j) for i in range(minx,maxx+1) for j in range(miny,maxy+1) if [i,j] != [x,y]]

# Show original pic
print("Original pic: (Press N to release)")
video = cv2.VideoCapture(0)
while True:
	if len(args) > 7:
		original_frame = cv2.imread("{}.jpg".format(args[7]))
	else:
		original_frame = video.read()[1]
	
	cv2.imshow("Capturing...",original_frame)
	key = cv2.waitKey(33)
	if key == ord('n') or key == ord('N'):
		break
video.release()
cv2.destroyAllWindows()
sleep(0.1)

# Go away, colors!
mono_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

# Show monocromatic pic
print("Black and white pic: (Press N to release)")
while True:
	cv2.imshow("Much simpler",mono_frame)
	key = cv2.waitKey(33)
	if key == ord('n') or key == ord('N'):
		break
cv2.destroyAllWindows()
sleep(0.1)

#Get the original size of pic (should be able to define)
a,b = len(mono_frame),len(mono_frame[0])
print("Size of image: ({},{})".format(a,b))

# Tries to homogenize the pic
if maximum_stdn_dev > 0:
	homogenize_count = 0
	for i in range(a):
		for j in range(b):
			values = []
			for duo in find_adj_nodes(i,j,a-1,b-1):
				values.append(mono_frame[duo[0],duo[1]])
			n = len(values)
			avg_v = sum(values)/n
			stnd_dev = (sum([(i-avg_v)**2])/(n-1))**(0.5)
			if stnd_dev < maximum_stdn_dev:
				homogenize_count += 1
				mono_frame[i][j] = avg_v
	print("Just lower resolution by {} pixels...".format(homogenize_count))

	# Show homogenized pic
	print("Unresoluted pic: (Press N to release)")
	while True:
		cv2.imshow("Even more simple!",mono_frame)
		key = cv2.waitKey(33)
		if key == ord('n') or key == ord('N'):
			break
	cv2.destroyAllWindows()
	sleep(0.1)

# Creates graph from mono_frame
G = nx.Graph()
for i in range(a):
	for j in range(b):
		G.add_node(i*b + j + 1)
for i in range(a):
	for j in range(b):
		p1 = mono_frame[i][j]
		n = i*b + j + 1
		for duo in find_adj_nodes(i,j,a-1,b-1):
			p2 = mono_frame[duo[0]][duo[1]]
			m = duo[0]*b + duo[1] + 1
			G.add_edge(n,m,weight=abs(p1-p2))
print("Graph created with {} nodes and {} edges".format(G.number_of_nodes(),G.number_of_edges()))

# From graph, gets minimum spanning tree
T = nx.minimum_spanning_tree(G)
print("Tree created with {} nodes and {} edges".format(T.number_of_nodes(),T.number_of_edges()))

# Cuts the n_regions-1 "heaviest" edges from minimum spanning tree
edges_in_order = sorted(T.edges(data=True),key=lambda x: x[2]['weight'],reverse=True)[0:n_regions-1]
for e in edges_in_order:
	T.remove_edge(*e[:2])
print("MST modificated to {} nodes and {} edges".format(T.number_of_nodes(),T.number_of_edges()))

# Format minimum spanning tree to frame
cutted_tree = copy(mono_frame)
# First the black region
for i in range(a):
	for j in range(b):
		cutted_tree[i][j] = 255
# Now the white cuts
for e in T.edges():
	y1 = e[0] % b
	x1 = int((e[0] - y1) / b)
	y2 = e[1] % b
	x2 = int((e[1] - y2) / b)
	cutted_tree[x2-1][y2-1] = 0
	cutted_tree[x1-1][y1-1] = 0

print("(Press N to release)")
# Show result form cuts
while True:
	cv2.imshow("New image!",cutted_tree)
	key =  cv2.waitKey(33)
	if key == ord('n') or key == ord('N'):
		break
cv2.destroyAllWindows()
sleep(0.1)

# Split script in 2 strategies
print("Press 'r' for look into lines and 'b' for regions:")
while True:
	choose_frame = cv2.imread("matrix.jpg")
	cv2.imshow("Choose!",choose_frame)
	key =  cv2.waitKey(33)
	if key == ord('r') or key == ord('R'):
		strategy = 1
		break
	elif key == ord('b') or key == ord('B'):
		strategy = 2
		break
cv2.destroyAllWindows()
sleep(0.1)

# Strategy 1: look for lines in image
if strategy == 1:

	# Calculates average density of cuts
	white_dots = sum([sum(p) for p in cutted_tree])/255
	avg_density = white_dots/(a*b)

	# Removes 'isolated' cuts
	cutted_count = 0
	extra_cutted_tree = copy(cutted_tree)
	for i in range(a):
		for j in range(b):
			if cutted_tree[i][j] == 255:
				neighborhood = find_adj_nodes(i,j,a-1,b-1,len1=def_len1,len2=def_len2)
				n_neighborhood = len(neighborhood)+1
				neighborhood_density = 1/n_neighborhood
				for neighbor in neighborhood:
					if cutted_tree[neighbor[0],neighbor[1]] == 255:
						neighborhood_density += 1/n_neighborhood
				if neighborhood_density < avg_density:
					cutted_count += 1
					extra_cutted_tree[i][j] = 0
	print("Cleaning process removed {} edges!".format(cutted_count))

	# Show better results
	while True:
		cv2.imshow("New image!",extra_cutted_tree)
		key =  cv2.waitKey(33)
		if key == ord('n') or key == ord('N'):
			break
	cv2.destroyAllWindows()
	sleep(0.1)

	# Creates graph from extra_cutted_tree
	F = nx.Graph()
	for i in range(a):
		for j in range(b):
			F.add_node(i*b + j + 1)
	for i in range(a):
		for j in range(b):
			p1 = extra_cutted_tree[i][j]
			if p1 == 255:
				n = i*b + j + 1
				for duo in find_adj_nodes(i,j,a-1,b-1):
					p2 = extra_cutted_tree[duo[0]][duo[1]]
					if p2 == 0:
						m = duo[0]*b + duo[1] + 1
						F.add_edge(n,m)
	print("Graph created with {} nodes and {} edges".format(F.number_of_nodes(),F.number_of_edges()))

	# Separates forest into lines
	print("Separating image into lines...")
	setteds = set()
	this_sets = []
	for i in range(a):
		for j in range(b):
			m = i*b + j + 1
			if m not in setteds:
				B = nx.node_connected_component(F,m)
				if len(B) > minimum_component_length:
					for k in B:
						setteds.add(k)
					this_sets.append(B)
	print("Yay! We found {} lines".format(len(this_sets)))

	# Shows each line
	if False:
		for region in this_sets:
			print("This is a {} length line...".format(len(region)))
			region_frame = copy(mono_frame)
			for i in range(a):
				for j in range(b):
					region_frame[i][j] = 255
			for n in region:
				y = n % b
				x = int((n-y)/b)
				region_frame[x-1][y-1] = mono_frame[x-1][y-1]
			while True:
				cv2.imshow("New image!",region_frame)
				key =  cv2.waitKey(33)
				if key == ord('n') or key == ord('N'):
					break
		cv2.destroyAllWindows()
		sleep(0.1)

	# Shows the full pic
	total_frame = copy(mono_frame)
	for i in range(a):
		for j in range(b):
			total_frame[i][j] = 255
	for region in this_sets:
		for n in region:
			y = n % b
			x = int((n-y)/b)
			total_frame[x-1][y-1] = mono_frame[x-1][y-1]
	while True:
		cv2.imshow("New image!",total_frame)
		key =  cv2.waitKey(33)
		if key == ord('n') or key == ord('N'):
			break
	cv2.destroyAllWindows()
	sleep(0.1)



	
# Strategy 2: look for regions in image
elif strategy == 2:

	# Separates forest into regions
	print("Separating image into regions...")
	setteds = set()
	this_sets = []
	for i in range(a):
		for j in range(b):
			m = i*b + j + 1
			if m not in setteds:
				B = nx.node_connected_component(T,m)
				if len(B) > minimum_component_length/2:
					for k in B:
						setteds.add(k)
					this_sets.append(B)
	print("Yay! We found {} regions".format(len(this_sets)))

	# Shows each region
	if False:
		for region in this_sets:
			print("This is a {} length region...".format(len(region)))
			region_frame = copy(mono_frame)
			for i in range(a):
				for j in range(b):
					region_frame[i][j] = 255
			for n in region:
				y = n % b
				x = int((n-y)/b)
				region_frame[x-1][y-1] = mono_frame[x-1][y-1]
			while True:
				cv2.imshow("New image!",region_frame)
				key =  cv2.waitKey(33)
				if key == ord('n') or key == ord('N'):
					break
		cv2.destroyAllWindows()
		sleep(0.1)

	# Shows the full pic
	total_frame = copy(mono_frame)
	for i in range(a):
		for j in range(b):
			total_frame[i][j] = 255
	for region in this_sets:
		for n in region:
			y = n % b
			x = int((n-y)/b)
			total_frame[x-1][y-1] = mono_frame[x-1][y-1]
	while True:
		cv2.imshow("New image!",total_frame)
		key =  cv2.waitKey(33)
		if key == ord('n') or key == ord('N'):
			break
	cv2.destroyAllWindows()
	sleep(0.1)
	

# Creating some squares
colors = [(255,0,0),(0,255,0),(0,0,255),(255, 0, 255),(255,255,0),(0,255,255),(255, 153, 0),(102, 255, 255),(204, 255, 51)]
ind = 0
for sett in this_sets:
	maxx,maxy,minx,miny = 0,0,10**10,10**10
	for m in sett:
		y = m % b
		x = int((m-j)/b)
		maxx = max(maxx,x)
		minx = min(minx,x)
		maxy = max(maxy,y)
		miny = min(miny,y)
	for x in range(minx,maxx+1):
		for y in [miny,maxy]:
			original_frame[x][y] = colors[ind]
	for y in range(miny,maxy+1):
		for x in [minx,maxx]:
			original_frame[x][y] = colors[ind]
	ind += 1
	if ind == len(colors):
		ind = 0

# Now the full pic with object limits...
print("And finally...")
while True:
	cv2.imshow("Let's take a final look!",original_frame)
	key =  cv2.waitKey(33)
	if key == ord('m') or key == ord('M'):
		break
cv2.destroyAllWindows()
sleep(0.1)