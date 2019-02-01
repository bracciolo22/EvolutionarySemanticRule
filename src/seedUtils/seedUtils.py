#----------------------------------------------------------------------------------------------------------------------
# IMPORTS
#----------------------------------------------------------------------------------------------------------------------

from sklearn.neighbors import NearestNeighbors
from copy import deepcopy
import random
import bezier
import numpy as np
import pandas as pd
import cv2
from deap import base, creator, tools

import matplotlib.pyplot as plt
import matplotlib.patches as patches

#----------------------------------------------------------------------------------------------------------------------
# GLOBALS - DATASET
#----------------------------------------------------------------------------------------------------------------------

dataset = None
dataset_muR = None
dataset_muG = None
dataset_muB = None
dataset_r = None
dataset_g = None
dataset_b = None
dataset_rgb = None
dataset_mask = None

basePath = "../"
baseInputFolder = "dataset/"
imgSubfolder = "img/"
maskSubfolder = "mask/"
datasetFilename = "datasetOverview.csv"

fullInputPath_dataset = basePath + baseInputFolder
fullInputPath_img = basePath + baseInputFolder + imgSubfolder
fullInputPath_mask = basePath + baseInputFolder + maskSubfolder

#----------------------------------------------------------------------------------------------------------------------
# GLOBALS - EA and CHROMOSOME
#----------------------------------------------------------------------------------------------------------------------

toolbox = None

nDecimalsChromosome = 2
nPointsPerChromosome = 8
minNPointsPerShape = 3

populationSize = 10

eliteRate = 0.1
minEliteSize = 1

crossoverProb = 0.8			 # probability of crossing over 2 parents
uniformCrossoverProb = 0.5		 # probability for uniform crossover
mutationProb = 0.8

subMutate_float_mu = 0.0
subMutate_float_sd = 0.2

tournamentSize = 1

bezierPerimeter_pointOffset = 0.001
nPointsPerPolyEdge = 50

overlapThreshold_minvalue = 0.1
overlapThreshold_maxvalue = 0.9
colourThreshold_minvalue = 0.1
colourThreshold_maxvalue = 0.9

chromosomeOrder = ['isShaped',
				   'isColoured', 
				   #'isBezier', 
				   'enabledPoints', 
				   'pointCoords', 
				   'colour', 
				   'shapeMatchingOverlap', 
				   'colourMatchingOverlap']
				   
chromosomeDef = [
	{'name' : 'isShaped', 'len' : 1, 'dtype' : bool},
	{'name' : 'isColoured', 'len' : 1, 'dtype' : bool},
	#{'name' : 'isBezier', 'len' : 1, 'dtype' : bool},
	{'name' : 'enabledPoints', 'len' : nPointsPerChromosome, 'dtype' : bool},
	{'name' : 'pointCoords', 'len' : 2 * nPointsPerChromosome, 'dtype' : float},
	{'name' : 'colour', 'len' : 3, 'dtype' : float},
	{'name' : 'shapeMatchingOverlap', 'len' : 1, 'dtype' : float},
	{'name' : 'colourMatchingOverlap', 'len' : 1, 'dtype' : float},	
	]
	
nChromosomesToShow = populationSize
exploitChromosomeRatio = 0.25
minNExploitChromosomes = 4

nEvolutionaryGenerationsPerLoop = 1
	
chromosomeLen = 0
for i in range(len(chromosomeDef)) :
	chromosomeDef[i]['startIdx'] = chromosomeLen
	chromosomeLen += chromosomeDef[i]['len']
	
nExploitChromosomes = np.round(float(populationSize) * exploitChromosomeRatio, 0)
if nExploitChromosomes < minNPointsPerShape :
	nExploitChromosomes = minNExploitChromosomes
nExploreChromosomes = nChromosomesToShow - nExploitChromosomes
print("hence, present ", nChromosomesToShow, "chromosomes, of which:\n", nExploitChromosomes, "exploited (mutation only)\n", nExploreChromosomes, "exploited (novelty search)")

eliteSize = populationSize * eliteRate
if eliteSize < 1 :
	eliteSize = 1
eliteSize = int(eliteSize)
print("elite size for novelty:", eliteSize)

#----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------

def getChromosomeBehaviour(ind, printDebug=False) :
	global chromosomeOrder, dataset, dataset_mask, dataset_muB, dataset_muG, dataset_muR
	
	chromosome_split, _ = splitChromosome(ind)
	
	if printDebug :
		print("chromosome split:")
		for x in chromosome_split :
			print(x, chromosome_split[x])
	
	shapeThreshold = np.round(chromosome_split[chromosomeOrder.index('shapeMatchingOverlap')][0], 2)
	isShaped = chromosome_split[chromosomeOrder.index('isShaped')][0]
	isColoured = chromosome_split[chromosomeOrder.index('isColoured')][0]
	chromosomeRGB = 255.0 * np.asarray(chromosome_split[chromosomeOrder.index('colour')])
	colourThreshold = np.round(chromosome_split[chromosomeOrder.index('colourMatchingOverlap')][0], 2)

	chromosomeBehaviour = [False for i in range(len(dataset))]
	
	chromosomeColourRatio = [None for i in range(len(dataset))]
	chromosomeOverlapRatio = [None for i in range(len(dataset))]
	chromosomeOverlapImg = [None for i in range(len(dataset))]
	
	for img_idx in range(len(dataset)) :
		overlapMatch = True
		if isShaped :
			img_bbox = getDatasetImgBBox(img_idx)
			chromosome_mask, chromosome_rgb = chromosome2cv2(ind, img_bbox, False)
			overlapScore, overlapImg = getImgOverlapScore(dataset_mask[img_idx], chromosome_mask, shapeThreshold, printDebug)
			
			chromosomeOverlapRatio[img_idx] = np.round(overlapScore, 2)
			chromosomeOverlapImg[img_idx] = deepcopy(overlapImg)
			
			overlapMatch = overlapScore > shapeThreshold
			if printDebug :
				print("\toverlapScore", np.round(overlapScore, 2), 
					  'shapeThreshold', np.round(shapeThreshold, 2), 'match', overlapMatch)

		colourMatch = True
		if isColoured :
			imgMuRGB = [dataset_muR[img_idx], dataset_muG[img_idx], dataset_muB[img_idx]]
			
			try :
				dist = np.round(np.linalg.norm(imgMuRGB - chromosomeRGB) / 
							np.linalg.norm(np.asarray([0,0,0]) - np.asarray([255, 255, 255])), 2)
			except Warning :
				print("getChromosomeBehaviour 35")
				dist = np.round(np.linalg.norm(imgMuRGB - chromosomeRGB) / 
							np.linalg.norm(np.asarray([0,0,0]) - np.asarray([255, 255, 255])), 2)
				
			# the match here is if the distance is less than a threshold!
			chromosomeColourRatio[img_idx] = dist
			colourMatch = dist < colourThreshold
			if printDebug :
				print("\timgMuRGB=", [np.round(x, 2) for x in imgMuRGB], 
					  "chromoRGB=", [np.round(x, 2) for x in chromosomeRGB], 
					  "dist=", dist, 
					  "threshold=", np.round(colourThreshold, 2), 
					  "match=", colourMatch)

		chromosomeBehaviour[img_idx] = overlapMatch & colourMatch

	chromosomeBehaviour = [int(x) for x in chromosomeBehaviour]

	if printDebug :
		print("final chromosomeOverlapRatio:", chromosomeOverlapRatio)
		print("final behaviour:", chromosomeBehaviour)
		
	return chromosomeBehaviour, chromosomeOverlapRatio, chromosomeOverlapImg, chromosomeColourRatio

def createPopulation() :
	#global population, populationBehaviour

	population = toolbox.population(populationSize)
	for i in range(len(population)) :
		repairChromosome(population[i])
		
	populationBehaviour = [getChromosomeBehaviour(x)[0] for x in population]
	
	return population, populationBehaviour
	
def splitChromosome(x, printDebug=False) :
	global chromosomeDef
	
	xSplit = {}
	xSplitName = {}
	for i in range(len(chromosomeDef)) :
		i0 = chromosomeDef[i]['startIdx']
		i1 = i0 + chromosomeDef[i]['len']
		xSplit[i] = x[i0:i1]
		xSplitName[chromosomeDef[i]['name']] = x[i0:i1]
	return xSplit, xSplitName
	
def getPointCoords(x, i) :
	global chromosomeOrder, chromosomeDef
	xSplit = splitChromosome(x)
	
	x0 = chromosomeDef[chromosomeOrder.index('pointCoords')]['startIdx']
	x1 = x0 + chromosomeDef[chromosomeOrder.index('pointCoords')]['len']
	
	pointCoords = x[x0 : x1]
	
	return (pointCoords[2*i], pointCoords[2*i + 1])
	
def repairChromosome_minUniqPointCoords(x, rounded, printDebug=False) :
	global chromosomeOrder, chromosomeDef
	
	if printDebug :
		print("repairChromosome_minUniqPointCoords")
		print(chromosomeToStr(x))
	
	loop = True
	
	chosen_idx = None
	mutatedPoint = None
	
	count = 0
	while loop :
		count += 1
		if count % 1000 == 0:
			print("yo!")
		_, xSplit = splitChromosome(x)
		
		enabledPoints = np.argwhere(np.asarray(xSplit['enabledPoints']) == 1).ravel()
		points = [getPointCoords(x, i) for i in enabledPoints]
		
		if printDebug :
			print("enabled points")
			print(enabledPoints)
			print("coordinates")
			print(points)

		if len(set(points)) == len(enabledPoints) :
			# no need to repair
			if printDebug :
				print("repaired!")
			loop = False
			continue
		
		#------------------------------------------------------------------------------------
		# repair chromosome
		#------------------------------------------------------------------------------------
		
		if printDebug :
			print("repair!")
		
		# 1. extract a point coordinate whith count > 1
		p_count = 0
		while p_count <= 1 :
			p_idx = np.random.randint(0, len(enabledPoints))
			p = points[p_idx]
			p_count = points.count(p)

		# 2. pick one of the points with that coordinate
		candidatePoints = np.argwhere(np.all(np.asarray(points) == p, axis=1)).ravel()
		chosen_idx = np.random.choice(candidatePoints, 1)[0]
		
		if printDebug :
			print("chosen_idx=", chosen_idx, "i.e. index", enabledPoints[chosen_idx])

		# 3. mutate the point
		mutatedPoint = toolbox.mutateSubGene_float(list(points[chosen_idx]))[0]
		for j in range(len(mutatedPoint)) :
			if mutatedPoint[j] < 0.0 :
				mutatedPoint[j] = 0.0
			elif mutatedPoint[j] > 1.0 :
				mutatedPoint[j] = 1.0
				
		# 4. round the point
		mutatedPoint = [np.round(v, rounded) for v in mutatedPoint]
		
		if printDebug :
			print("mutatedPoint bounded and rounded:", mutatedPoint)
		
		# 5. reassign
		x0 = chromosomeDef[chromosomeOrder.index('pointCoords')]['startIdx']
		
		if printDebug :
			print("convert coord", x[x0 + 2*enabledPoints[chosen_idx] : x0 + 2*enabledPoints[chosen_idx] + 2], "to", mutatedPoint)
		
		x[x0 + 2*enabledPoints[chosen_idx] : x0 + 2*enabledPoints[chosen_idx] + 2] = mutatedPoint
		
	return enabledPoints, points, chosen_idx, mutatedPoint
	
def repairChromosome_enabledPoints(x, printDebug=False) :
	global chromosomeDef, chromosomeOrder, minNPointsPerShape
	x0 = chromosomeDef[chromosomeOrder.index('enabledPoints')]['startIdx']
	l = chromosomeDef[chromosomeOrder.index('enabledPoints')]['len']
	x1 = x0 + l
	if printDebug :
		print(chromosomeToStr(x))
		print(x[x0 : x1])
	while(sum(x[x0 : x1]) < minNPointsPerShape) :
		idx = np.random.randint(0, l)
		x[x0 + idx] = 1
		if printDebug :
			print(idx, x[x0 : x1])

def repairChromosome_shapedOrColoured(x, printDebug=False) :
	global chromosomeDef, chromosomeOrder
	xShape = chromosomeDef[chromosomeOrder.index('isShaped')]['startIdx']
	xColour = chromosomeDef[chromosomeOrder.index('isColoured')]['startIdx']
	if printDebug :
		print(chromosomeToStr(x))
		print("indices:", xShape, xColour)
		print(x[xShape], x[xColour])
	while(x[xShape] + x[xColour] == 0) :
		idx = np.random.randint(0, 2)
		if idx == 0 :
			x[xShape] = 1
		else :
			x[xColour] = 1
		if printDebug :
			print(x[xShape], x[xColour])	
	
def repairChromosome_roundAlleleValues(x, nDecimals, printDebug=False) :
	global chromosomeDef
	
	for gene in chromosomeDef :
		if gene['dtype'] == float :
			if printDebug :
				print(gene)
			for i in range(gene['startIdx'], gene['startIdx'] + gene['len']) :
				if printDebug :
					print(x[i], end=' -> ')
				x[i] = np.round(x[i], nDecimals)
				if printDebug :
					print(x[i])
	
def repairChromosome_boundedFloatAlleleValues(x, printDebug=False) :
	global chromosomeDef
	
	for gene in chromosomeDef :
		if gene['dtype'] == float :
			x0 = gene['startIdx']
			x1 = gene['startIdx'] + gene['len']
			if printDebug :
				print(gene)
			x[x0 : x1] = list(np.clip(x[x0:x1], 0.0, 1.0))
	
def repairChromosome_noLineShape(x, rounding, printDebug=False) :
	global chromosomeOrder, chromosomeDef
	
	xSplit, _ = splitChromosome(x)
	
	x0 = chromosomeDef[chromosomeOrder.index('pointCoords')]['startIdx']
	x1 = x0 + chromosomeDef[chromosomeOrder.index('pointCoords')]['len']
	
	if printDebug :
		print(x0, x1)

	enabledPoints = np.argwhere(np.asarray(xSplit[chromosomeOrder.index('enabledPoints')]) == 1).ravel()
	points = xSplit[chromosomeOrder.index('pointCoords')]
	
	if printDebug :
		print(enabledPoints)
		print(points)
	
	points_x = np.asarray(points)[[2*i for i in enabledPoints]]
	points_y = np.asarray(points)[[2*i+1 for i in enabledPoints]]
	
	if printDebug :
		print(points_x)
		print(points_y)

	isHorizontal = len(set(points_x)) == 1
	isVertical = len(set(points_y)) == 1
	
	if isHorizontal :
		if printDebug :
			print("horizontal!")
			print(points_x)
			
		idx = np.random.randint(0, len(enabledPoints))
		targetIdx = x0 + 2 * enabledPoints[idx]
		v = points_x[idx]
		mv = v
		while mv == v :
			mv = toolbox.mutateSubGene_float([v])[0][0]
			mv = np.round(mv, rounding)
			if mv < 0.0 :
				mv = 0.0
			if mv > 1.09 :
				mv = 1.0
				
		if printDebug :
			print(mv, " in idx", targetIdx)
		x[targetIdx : targetIdx + 1] = [mv]
	#else :
	#	print("not horizontal!")
		
	if isVertical :
		if printDebug :
			print("vertical!")
			print(points_y)
		# just modify one value
		idx = np.random.randint(0, len(enabledPoints))
		targetIdx = x0 + 2 * enabledPoints[idx] + 1
		v = points_y[idx]
		mv = v
		while mv == v :
			mv = toolbox.mutateSubGene_float([v])[0][0]
			mv = np.round(mv, rounding)
			if mv < 0.0 :
				mv = 0.0
			if mv > 1.09 :
				mv = 1.0
				
		if printDebug :
			print(mv, " in idx", targetIdx)
		x[targetIdx : targetIdx + 1] = [mv]
	
	#else :
	#	print("not vertical!")
				
def repairChromosome_boundedThresholds(x, printDebug=False) :
	global overlapThreshold_minvalue, overlapThreshold_maxvalue, colourThreshold_minvalue, colourThreshold_maxvalue, chromosomeDef, chromosomeOrder
	
	xOverlap = chromosomeDef[chromosomeOrder.index('shapeMatchingOverlap')]['startIdx']
	xColour = chromosomeDef[chromosomeOrder.index('colourMatchingOverlap')]['startIdx']
	
	if printDebug :
		print("repairChromosome_boundedThresholds")
		print(chromosomeToStr(x))
		print("indices overlap, colour:", xOverlap, xColour)
		print("values:", x[xOverlap], x[xColour])
		print("overlap bounds:", overlapThreshold_minvalue, overlapThreshold_maxvalue)
		print("colour bounds:", colourThreshold_minvalue, colourThreshold_maxvalue)
	
	if x[xOverlap] < overlapThreshold_minvalue :
		x[xOverlap] = overlapThreshold_minvalue
	
	if x[xOverlap] > overlapThreshold_maxvalue :
		x[xOverlap] = overlapThreshold_maxvalue
		
	if x[xColour] < colourThreshold_minvalue :
		x[xColour] = colourThreshold_minvalue
	
	if x[xColour] > colourThreshold_maxvalue :
		x[xColour] = colourThreshold_maxvalue
		
	if printDebug :
		print("final values:", x[xOverlap], x[xColour])
	
def repairChromosome(x, rounded=nDecimalsChromosome, printDebug=False) :
	repairChromosome_shapedOrColoured(x, printDebug)
	repairChromosome_enabledPoints(x, printDebug)  
	repairChromosome_boundedFloatAlleleValues(x, printDebug)
	repairChromosome_roundAlleleValues(x, rounded, printDebug)	
	repairChromosome_minUniqPointCoords(x, rounded)
	repairChromosome_noLineShape(x, rounded)
	repairChromosome_boundedThresholds(x, printDebug)
	
def chromosomeToStr(x, rounding=nDecimalsChromosome) :
	global chromosomeOrder
	
	xSplit, _ = splitChromosome(x)
	
	chStr = ""
	for k in xSplit :
		chStr += " ".join([str(np.round(v, rounding)) for v in xSplit[k]])
		if k != list(xSplit.keys())[-1] :
			chStr += "|"
	return chStr
	
def extractEnabledPoints(splitChromosome) :
	bitmask = splitChromosome['enabledPoints']
	coords = splitChromosome['pointCoords']
	
	extract = []
	for i in range(len(bitmask)) :
		if bitmask[i] == 1 :
			extract.append(coords[2*i])
			extract.append(coords[2*i+1])
			
	return extract
	
def bezierify(ind, plot=False) :

	bez = np.asarray(ind).reshape(1, -1, 2)[0]
	
	bez_x = list(bez[:,0])
	bez_y = list(bez[:,1])
	
	bez_x += [bez_x[0]]
	bez_y += [bez_y[0]]
	
	target_points = np.asfortranarray([bez_x, bez_y])
	target_points_closed = np.asfortranarray([[bez_x[-1], bez_x[0]], [bez_y[-1], bez_y[0]]])
	
	target_curve = bezier.Curve.from_nodes(target_points)
	target_curve_closed = bezier.Curve.from_nodes(target_points_closed)
	
	target_poly = bezier.CurvedPolygon(target_curve, target_curve_closed)
	
	try :
		s_vals = np.linspace(0.0, 1.0, int(np.round(target_curve.length / bezierPerimeter_pointOffset, 0)))
	except Warning :
		print("bezierify 20") 
		s_vals = np.linspace(0.0, 1.0, int(np.round(target_curve.length / bezierPerimeter_pointOffset, 0)))
		
	s_points = target_curve.evaluate_multi(s_vals)
	
	try :
		s_vals2 = np.linspace(0.0, 1.0, int(np.round(target_curve_closed.length / bezierPerimeter_pointOffset, 0)))
	except Warning :
		print("bezierify 28")
		s_vals2 = np.linspace(0.0, 1.0, int(np.round(target_curve_closed.length / bezierPerimeter_pointOffset, 0)))
		
	s_points2 = target_curve_closed.evaluate_multi(s_vals2)
	bezierPerimeter = np.concatenate((s_points, s_points2), axis=1)
	
	# extract the bbox
	bez_bbox = {
		'x0' : np.min(bezierPerimeter[0]),
		'y0' : np.min(bezierPerimeter[1]),
		'width' : np.max(bezierPerimeter[0]) - np.min(bezierPerimeter[0]),
		'height' : np.max(bezierPerimeter[1]) - np.min(bezierPerimeter[1])
	}

	if plot :
		fig = plt.figure()
		ax = target_poly.plot(100)
		ax = target_curve.plot(100, ax=ax, alpha=0)
		ax = target_curve_closed.plot(100, ax=ax, alpha=0)
		ax.scatter(x=bezierPerimeter[0], y=bezierPerimeter[1], c='black', marker='*')
		ax.scatter(x=target_points[0], y=target_points[1], c='red', marker='o')
		ax.scatter(x=target_points_closed[0], y=target_points_closed[1], c='red', marker='o')
		rect = patches.Rectangle((bez_bbox['x0'], bez_bbox['y0']), bez_bbox['width'], bez_bbox['height'], linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		plt.title("original bezier")
		plt.show()
		plt.cla()
		plt.clf()
		plt.clos('all')
		
	additionalInfo = {
		'poly' : target_poly,
		'curve' : target_curve,
		'curve_closed' : target_curve_closed,
		'points' : target_points,
		'points_closed' : target_points_closed
	}
	
	return bezierPerimeter, bez_bbox, additionalInfo 
	
def polify(ind, plot=False) :
	global nPointsPerPolyEdge
	
	for _ in range(10) :
		print("I SHOULD NOT RUN POLIFY!")
	
	pol = np.asarray(ind).reshape(1, -1, 2)[0]
	
	pol_x = list(pol[:,0])
	pol_y = list(pol[:,1])
	
	pol_x += [pol_x[0]]
	pol_y += [pol_y[0]]
	
	target_points = np.asfortranarray([pol_x, pol_y])
	target_points_closed = np.asfortranarray([[pol_x[-1], pol_x[0]], [pol_y[-1], pol_y[0]]])
	
	polyPerimeter_x = []
	polyPerimeter_y = []

	for i in range(len(target_points[0])-1) :
		linePoints_x = list(np.linspace(target_points[0][i], target_points[0][i+1], nPointsPerPolyEdge)[:-1])
		linePoints_y = list(np.linspace(target_points[1][i], target_points[1][i+1], nPointsPerPolyEdge)[:-1])
		polyPerimeter_x += linePoints_x
		polyPerimeter_y += linePoints_y

	polyPerimeter_x.append(polyPerimeter_x[0])
	polyPerimeter_y.append(polyPerimeter_y[0])
	
	polyPerimeter = np.asarray([polyPerimeter_x, polyPerimeter_y])
	
	# extract the bbox
	poly_bbox = {
		'x0' : np.min(polyPerimeter[0]),
		'y0' : np.min(polyPerimeter[1]),
		'width' : np.max(polyPerimeter[0]) - np.min(polyPerimeter[0]),
		'height' : np.max(polyPerimeter[1]) - np.min(polyPerimeter[1])
	}
	
	additionalInfo = None
	
	if plot :
		fig, ax = plt.subplots()
		ax.scatter(x=target_points[0], y=target_points[1], c='red', marker='o')
		plt.scatter(x=target_points_closed[0], y=target_points_closed[1], c='red', marker='o')
		plt.scatter(x=polyPerimeter[0], y=polyPerimeter[1], c='black', marker='*')
		poly = patches.Polygon(target_points.transpose())
		ax.add_patch(poly)
		rect = patches.Rectangle((poly_bbox['x0'], poly_bbox['y0']), 
								 poly_bbox['width'], poly_bbox['height'], linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		
		plt.title("original poly")
		plt.show()
		
	return polyPerimeter, poly_bbox, additionalInfo	
	
def renderChromosome(x) :
    global chromosomeOrder
    
    chromosomeSplit, _ = splitChromosome(x)
    perimeter, bbox, colour, alpha = chromosome2image(x)
    
    fig, ax = plt.subplots()

    isShaped = chromosomeSplit[chromosomeOrder.index('isShaped')][0]
    if isShaped :
        poly = patches.Polygon(perimeter.transpose(), color=colour + alpha)
        ax.add_patch(poly)
        ax.scatter(perimeter[0], perimeter[1], c='black', marker='.')
        plt.xlim(bbox['x0'], bbox['x0']+bbox['width'])
        plt.ylim(bbox['y0'], bbox['y0']+bbox['height'])
    else :
        ble = np.asarray(255.0 * np.asarray(colour), dtype=np.uint8)
        ble = np.asarray([[ble]], dtype=np.uint8)
        plt.imshow(ble)
        
    plt.title("chromosome\n" + chromosomeToStr(x))
    plt.show()
	
def chromosome2image(x) :
	_, split_ch = splitChromosome(x)
	extract = extractEnabledPoints(split_ch)
	
	perimeter = None
	bbox = None
	
	perimeter, bbox, _ = bezierify(extract)
		
	isColoured = split_ch['isColoured']
	colour = split_ch['colour']
	
	return perimeter, bbox, colour, isColoured
	
def getDatasetImgBBox(img_idx) :
	global dataset
	
	img_bbox = {'x0' : dataset.iloc[img_idx]['bbox_x0'], 'y0' : dataset.iloc[img_idx]['bbox_y0'], 
			   'width' : dataset.iloc[img_idx]['bbox_width'], 'height' : dataset.iloc[img_idx]['bbox_height']}
	
	return img_bbox

def loadDataset(printProgress=False) :

	global dataset, dataset_muR, dataset_muG, dataset_muB
	global dataset_r, dataset_g, dataset_b, dataset_rgb, dataset_mask, datasetFilename
	global fullInputPath_dataset, fullInputPath_img, fullInputPath_mask
	
	dataset = pd.DataFrame()
	dataset_rgb = []
	dataset_mask = []
	dataset_r = []
	dataset_g = []
	dataset_b = []
	dataset_muR = []
	dataset_muG = []
	dataset_muB = []
	
	dataset = pd.read_csv(fullInputPath_dataset + datasetFilename)
	
	for img_idx in range(len(dataset)) :
		
		if printProgress :
			print(img_idx+1, '/', len(dataset), 10*' ', end='\r')
		
		f = dataset.iloc[img_idx]['filename']
		rgb_img = cv2.imread(fullInputPath_img + f)
		mask_img = cv2.imread(fullInputPath_mask + f)
		
		img_bbox = getDatasetImgBBox(img_idx)
		mask_img = mask_img[img_bbox['y0'] : img_bbox['y0'] + img_bbox['height'], img_bbox['x0'] : img_bbox['x0'] + img_bbox['width']]
			
		rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
		mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
		
		refImg = rgb_img[img_bbox['y0'] : img_bbox['y0'] + img_bbox['height'], img_bbox['x0'] : img_bbox['x0'] + img_bbox['width']]
		cutImg = cv2.bitwise_and(refImg, cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB))
		roiPixels = np.argwhere(mask_img > 0)
		tmp_r, tmp_g, tmp_b = cv2.split(cutImg)
		
		# now the average value?
		mu_r = np.mean(tmp_r[roiPixels[:,0], roiPixels[:,1]])
		mu_g = np.mean(tmp_g[roiPixels[:,0], roiPixels[:,1]])
		mu_b = np.mean(tmp_b[roiPixels[:,0], roiPixels[:,1]])
				
		dataset_rgb.append(rgb_img)
		dataset_mask.append(mask_img)
		dataset_r.append(tmp_r)
		dataset_g.append(tmp_g)
		dataset_b.append(tmp_b)
		dataset_muR.append(mu_r)
		dataset_muG.append(mu_g)
		dataset_muB.append(mu_b)
		
	if printProgress :
		print("dataset loaded")
		
def initDEAP() :
	global chromosomeDef, chromosomeOrder, toolbox
	
	toolbox = base.Toolbox()
	
	# single-objective function: novelty!
	creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
	creator.create("Individual", list, fitness=creator.FitnessMax)
	
	# sub-chromosome description
	toolbox.register("allele_bit", random.randint, 0, 1)
	toolbox.register("allele_float", random.random)

	chromosomeDEAP = []
	for i in range(len(chromosomeOrder)) :
		subGeneType = chromosomeDef[i]['dtype']
		subGeneLen = chromosomeDef[i]['len']
		toolboxAttr = None
		if subGeneType == bool :
			toolboxAttr = toolbox.allele_bit
		elif subGeneType == float :
			toolboxAttr = toolbox.allele_float
		else :
			print("UNHANDLED SUB-CHROMOSOME TYPE", subGeneType)
		chromosomeDEAP += subGeneLen * [toolboxAttr]

	# chromosome
	toolbox.register("individual", tools.initCycle, creator.Individual, tuple(chromosomeDEAP), n=1)

	# population
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	# subGene crossover
	toolbox.register("crossoverSubGene", tools.cxUniform, indpb=uniformCrossoverProb)

	# crossover
	toolbox.register("crossover", crossover, crossoverProb=uniformCrossoverProb)

	# subGene mutation
	toolbox.register("mutateSubGene_bit", tools.mutFlipBit, indpb=mutationProb)
	toolbox.register("mutateSubGene_float", tools.mutGaussian, mu=subMutate_float_mu, sigma=subMutate_float_sd, indpb=mutationProb)

	# chromosome mutation
	toolbox.register("mutate", hierarchicalMutation, mutateProb=mutationProb)

	# selection (roulette cannot be used for minimisation or for fitness score = 0)
	toolbox.register("select", tools.selTournament, tournsize=tournamentSize)

	# fitness function
	toolbox.register("fitnessFunction_noveltySearch", fitnessFunction_noveltySearch)
	
	return toolbox
	
def getImgOverlapScore(img1, img2, th, plot=False) :
	
	try :
		overlap = (img1.astype(float) + img2.astype(float)) / (np.max(img1.astype(float)) + np.max(img2.astype(float)))
	except Warning :
		print("getImgOverlapScore 4")
		overlap = (img1.astype(float) + img2.astype(float)) / (np.max(img1.astype(float)) + np.max(img2.astype(float)))
	
	importantArea = overlap.ravel()
	importantArea = importantArea[np.where(importantArea > 0.0)]
   
	try :
		ratioOverlap = float(len(importantArea[np.where(importantArea == np.max(overlap))])) / len(importantArea)
	except Warning :
		print("getImgOVerlapScore 13")
		ratioOverlap = float(len(importantArea[np.where(importantArea == np.max(overlap))])) / len(importantArea)
	
	result = (overlap * 255.0).astype(np.int)
	
	#if plot :
	#	plt.imshow(result)
	#	plt.title("image overlap (score={0}, thresh={1})".format(np.round(ratioOverlap, 2), np.round(th, 2)))
	#	plt.show()
		
	return ratioOverlap, result
	
def cv2fy(chromosome_perimeter, width, height, isColoured, colour, plot=False) :
	chromosome_perimeter = chromosome_perimeter.round(0)
	chromosome_perimeter = chromosome_perimeter.astype(np.uint)
	chromosome_perimeter = chromosome_perimeter.transpose()
	chromosome_image = np.zeros((height, width, 3), np.uint8)
	
	if isColoured :
		chromosome_image = cv2.polylines(chromosome_image, np.int32([chromosome_perimeter]),True,colour)
		chromosome_image = cv2.fillPoly(chromosome_image, np.int32([chromosome_perimeter]), colour)
	else :
		chromosome_image = cv2.polylines(chromosome_image, np.int32([chromosome_perimeter]),True,(255,255,255))
		chromosome_image = cv2.fillPoly(chromosome_image, np.int32([chromosome_perimeter]), (255,255,255))
	
	chromosome_image = cv2.flip(chromosome_image, 0)
	
	if plot :
		plt.imshow(chromosome_image)
		plt.title("scaled cv2 of chromosome (coloured={0})".format(isColoured))
		plt.show()
	
	return chromosome_image
	
def cv2fyV2(chromosome_perimeter, width, height, fillColour=None, bgColour=None, perimeterColour=(255,255,255), plot=False) :
	# NOTE! this function is ONLY a fix for renderChromosome done via Flask, DO NOT USE IT DURING NON-FLASK STUFF (or make sure you know what you do)

	chromosome_perimeter = chromosome_perimeter.round(0)
	chromosome_perimeter = chromosome_perimeter.astype(np.uint)
	chromosome_perimeter = chromosome_perimeter.transpose()
	
	chromosome_image = np.zeros((height, width, 3), np.uint8)
	if bgColour != None :
		chromosome_image[:][:] = bgColour
	
	chromosome_image = cv2.polylines(chromosome_image, np.int32([chromosome_perimeter]),True, perimeterColour)
	
	if fillColour != None :
		chromosome_image = cv2.fillPoly(chromosome_image, np.int32([chromosome_perimeter]), fillColour)
	
	chromosome_image = cv2.flip(chromosome_image, 0)
	
	if plot :
		plt.imshow(chromosome_image)
		plt.title("scaled cv2 of chromosome (coloured={0})".format(isColoured))
		plt.show()
	
	return chromosome_image
	
def scaleChromosomeToRefImage(chromoPerimeter, chromoBbox, refBbox, plot=False) :
	# move the bbox to [0,0]
	chromoPerimeter[0,:] -= chromoBbox['x0']
	chromoPerimeter[1,:] -= chromoBbox['y0']
	
	try :
		chromoPerimeter[0,:] = chromoPerimeter[0,:] * refBbox['width'] / chromoBbox['width']
	except Warning :
		print("scaleChromosomeToRefImage 7, chromoBbox['width']")
		chromoPerimeter[0,:] = chromoPerimeter[0,:] * refBbox['width'] / chromoBbox['width']
	
	try :
		chromoPerimeter[1,:] = chromoPerimeter[1,:] * refBbox['height'] / chromoBbox['height']
	except Warning :
		print("scaleChromosomeToRefImage 13, chromoBbox['height']=", chromoBbox['height']) 
		chromoPerimeter[1,:] = chromoPerimeter[1,:] * refBbox['height'] / chromoBbox['height']
		
	if plot :
		fig = plt.figure()
		plt.title("scaled chromosome (w{0} x h{1})".format(refBbox['width'], refBbox['height']))
		plt.scatter(x=chromoPerimeter[0,:], y=chromoPerimeter[1,:])
		plt.show()
		plt.cla()
		plt.clf()
		plt.clos('all')
		
	return chromoPerimeter
	
def chromosome2cv2(ind, img_bbox, plot=False) :
	# 1. conver the chromosome into an image perimeter
	chromosome_perimeter, chromosome_bbox, chromosome_colour, chromosome_isColoured = chromosome2image(ind)
	chromosome_isColoured = chromosome_isColoured[0]
	
	# 2. scale it wrt the target image's bounding box
	chromosome_perimeter = scaleChromosomeToRefImage(chromosome_perimeter, chromosome_bbox, img_bbox, plot)
	
	# 3. convert it into a cv2 image
	chromosome_rgb = cv2fy(chromosome_perimeter, img_bbox['width'], 
							 img_bbox['height'], chromosome_isColoured, 255.0 * np.asarray(chromosome_colour), plot)
	
	# 4. convert to grayscale
	chromosome_mask = cv2fy(chromosome_perimeter, img_bbox['width'], img_bbox['height'], False, None, False)
	chromosome_mask = cv2.cvtColor(chromosome_mask, cv2.COLOR_RGB2GRAY)
	
	if plot :
		plt.imshow(chromosome_mask)
		plt.title("bitmask chromosome to match")
		plt.show()
		
	return chromosome_mask, chromosome_rgb
	
def exploitChromosome(ind, indBehaviour, plot=False) :
	global nExploitChromosomes
	
	exploitedPopulation = [deepcopy(ind) for _ in range(nExploitChromosomes)]
	for i in range(len(exploitedPopulation)) :
		toolbox.mutate(exploitedPopulation[i])
		repairChromosome(exploitedPopulation[i])
	exploitedPopulationBehaviour = [getChromosomeBehaviour(x)[0] for x in exploitedPopulation]
		
	if plot :
		print("end of exploitation")
		print("selected chromosome")
		renderChromosomeAndBehaviour(ind, indBehaviour)
		print("exploited population and behaviour")
		for i in range(nExploitChromosomes) :
			renderChromosomeAndBehaviour(exploitedPopulation[i], exploitedPopulationBehaviour[i])
	
	return exploitedPopulation, exploitedPopulationBehaviour
	
def mutate(x, mutateProb, printDebug=False) :
	global chromosomeOrder, chromosomeDef, toolbox
	
	if printDebug :
		print(chromosomeToStr(x))  
		
	mutationOccurred = False
	
	# we assume that each chunk of a chromosome is a gene, so first we decide wheher to evolve each one and then the inside
	loop = True
	while loop :
		p = np.random.random()
		if printDebug :
			print("MP=", mutateProb, "p=", p)
		if p > mutateProb :
			if printDebug :
				print("end of mutation")
			loop = False
			continue
			
		# ok, we can mutate. pick a subgene
		mutationOccurred = True
		
		subGene = np.random.randint(0, len(chromosomeOrder))
		subGeneName = chromosomeOrder[subGene]
		x0 = chromosomeDef[subGene]['startIdx']
		x1 = x0 + chromosomeDef[subGene]['len']
		if printDebug :
			print("mutate subgene", subGene, subGeneName, chromosomeDef[subGene])
			print("from:", x[x0 : x1])
			
		if chromosomeDef[subGene]['dtype'] == float :
			# float mutation
			x[x0:x1] = toolbox.mutateSubGene_float(x[x0:x1])[0]
		elif chromosomeDef[subGene]['dtype'] == bool :
			# bitwise mutation
			x[x0:x1] = toolbox.mutateSubGene_bit(x[x0:x1])[0]
		else :
			# nothing yet!
			print("WARNING! allele value dtype", chromosomeDef[subGene]['dtype'], "not handled during mutation")	
			
		if printDebug :
			print("to:", x[x0 : x1])
			
	return mutationOccurred
	
def hierarchicalMutation(x, mutateProb, printDebug=False) :
	global chromosomeOrder, chromosomeDef, toolbox
	
	newX = deepcopy(x)

	mutationOccurred = False
	
	# the new mutation should be like this: 
	# - first mutate the main flags: is shaped/is coloured
	# - then decide to go inside: which of the two? both?
	# - so essentially mutate only if the flag is "on"
	# - maybe the best way is preparing a mutation strategy?
	
	chromosomeSplit, _ = splitChromosome(x)
	
	if printDebug :
		print("mutate chromosome. original:")
		print(chromosomeToStr(x))
		
	x_isShaped = chromosomeDef[chromosomeOrder.index('isShaped')]['startIdx']
	x_isColoured = chromosomeDef[chromosomeOrder.index('isColoured')]['startIdx']
	
	enabledPoints_x0 = chromosomeDef[chromosomeOrder.index('enabledPoints')]['startIdx']
	enabledPoints_x1 = enabledPoints_x0 + chromosomeDef[chromosomeOrder.index('enabledPoints')]['len']
	
	pointCoords_x0 = chromosomeDef[chromosomeOrder.index('pointCoords')]['startIdx']
	pointCoords_x1 = pointCoords_x0 + chromosomeDef[chromosomeOrder.index('pointCoords')]['len']
	
	#############################################################################################
	# 1. evolve shape/colour (based on p)
	p = np.random.random()
	if printDebug :
		print("MP=", mutateProb, "p=", p)
	
	if p <= mutateProb :
		mutationOccurred = True
		
		if printDebug: 
			print("1. change shape/colour flags")
		
		if printDebug :
			print("2-bit coords:", x_isShaped, x_isColoured)

		chShapeColour = [x[x_isShaped], x[x_isColoured]]

		if printDebug :
			print("from", chShapeColour)

		chShapeColour = toolbox.mutateSubGene_bit(chShapeColour)[0]

		if printDebug :
			print("to", chShapeColour)

		repairChromosome_shapedOrColoured(chShapeColour)

		if printDebug :
			print("repaired final", chShapeColour)

		newX[x_isShaped] = chShapeColour[0]
		newX[x_isColoured] = chShapeColour[1]
	
		if printDebug :
			print("newX:")
			print(chromosomeToStr(newX))
	else :
		if printDebug :
			print("do not evolve shape or colour, keep it as it is")
	
	if printDebug :
		print("2. retrieve the candidate second phase: do we evolve the bitmask? do we evolve colour? for sure we evolve thresholds")
	
	candidateSubgenes = []
	
	if newX[x_isShaped] == 1 :
		if printDebug :
			print("we can evolve its shape and threshold")
		candidateSubgenes.append('enabledPoints')
		candidateSubgenes.append('pointCoords')
		candidateSubgenes.append('shapeMatchingOverlap')
		
	if newX[x_isColoured] == 1 :
		if printDebug :
			print("we can evolve its colour and threshold")
		candidateSubgenes.append('colour')
		candidateSubgenes.append('colourMatchingOverlap')
		
	loop = True
	while loop :
		if printDebug :
			# now in loop we evolve this stuff:
			print("\ncandidate subgenes:", candidateSubgenes)
			
		if len(candidateSubgenes) == 0 :
			if printDebug :
				"nothing to evolve anymore"
			loop = False
			continue
		
		p = np.random.random()
		if printDebug :
			print("MP=", mutateProb, "p=", p)
		if p > mutateProb :
			if printDebug :
				print("end of mutation")
			loop = False
			continue

		mutationOccurred = True
		 
		sampledIdx = np.random.randint(0, len(candidateSubgenes))
		subGeneName = candidateSubgenes[sampledIdx]
		subGeneIdx = chromosomeOrder.index(subGeneName)
		x0 = chromosomeDef[subGeneIdx]['startIdx']
		x1 = x0 + chromosomeDef[subGeneIdx]['len']
		
		# if we evolve point coords then we should only evolve the points which are enabled
		oldPointCoords = None
		oldEnabledPoints = None
		
		if printDebug :
			print("sampledIdx =", sampledIdx)
			print("mutate subgene", subGeneIdx, subGeneName, chromosomeDef[subGeneIdx])
			print("from:", newX[x0 : x1])
		
		if subGeneName == "pointCoords" :
			if printDebug :
				print("whatch out for the enabled points only!")
			oldPointCoords = deepcopy(newX)
			oldEnabledPoints = newX[enabledPoints_x0 : enabledPoints_x1]
		
		if chromosomeDef[subGeneIdx]['dtype'] == float :
			# float mutation
			newX[x0:x1] = toolbox.mutateSubGene_float(x[x0:x1])[0]
		elif chromosomeDef[subGeneIdx]['dtype'] == bool :
			# bitwise mutation
			newX[x0:x1] = toolbox.mutateSubGene_bit(x[x0:x1])[0]
		else :
			# nothing yet!
			print("WARNING! allele value dtype", chromosomeDef[subGeneIdx]['dtype'], "not handled during mutation")
			

		if printDebug :
			print("to:", newX[x0 : x1])
			
		if subGeneName == "pointCoords" :
			if printDebug :
				print("restore points which were not enabled, i.e. those from")
				print(oldPointCoords[x0 : x1])
				print("based on flags:")
				print(oldEnabledPoints)
			for bit_idx in range(len(oldEnabledPoints)) :
				if printDebug :
					print("bit_idx:", bit_idx)
					print("bitFlag:", oldEnabledPoints[bit_idx])
					print("mutated value:", getPointCoords(newX, bit_idx))
				
				if oldEnabledPoints[bit_idx] == 0 :
					pc_x0 = chromosomeDef[chromosomeOrder.index('pointCoords')]['startIdx'] + 2*bit_idx
					pc_x1 = chromosomeDef[chromosomeOrder.index('pointCoords')]['startIdx'] + 2*bit_idx + 1
					originalCoord = getPointCoords(oldPointCoords, bit_idx)
					
					if printDebug :
						print("restore old value i.e 2coords", originalCoord, "in global indices", pc_x0, pc_x1)
						
					newX[pc_x0] = originalCoord[0]
					newX[pc_x1] = originalCoord[1]
					
				else :
					if printDebug :
						print("keep")
					
		if printDebug :
			print("newX:")
			print(chromosomeToStr(newX))
			
		# pop that subgene from the list, we evolve it only once
		candidateSubgenes.pop(sampledIdx)
		
	# final moving: newX becomes x
	x[0:len(x)] = newX[0:len(newX)]
	  
	# we better repair
	return mutationOccurred
	
def EA_generation(pop, popBehaviour, behaviourArchive, printEAprogress=False) :
	global toolbox, eliteSize, tools, crossoverProb, uniformCrossoverProb
	
	curBest = tools.selBest(pop, 1)[0]
	bestBehaviour = getChromosomeBehaviour(curBest)[0]
	fitnessBest = curBest.fitness.values
	
	if printEAprogress :
		print("population at the beginning of recombination, popSize=", len(pop))
		for i in range(len(pop)) :
			print(chromosomeToStr(pop[i]), popBehaviour[i], pop[i].fitness.values)
		print("BEST:", chromosomeToStr(curBest, 1), bestBehaviour, fitnessBest)
		print("behaviourAchive:")
		for x in behaviourArchive :
			print(x)
	
	# --------------------------------------------------------------------------------------
	# MATING POOL FORMATION
	# --------------------------------------------------------------------------------------
	
	# select and clone the next generation individuals
	offspring = toolbox.select(pop, len(pop))
	offspring = list(map(toolbox.clone, offspring))
	
	if printEAprogress :
		print("mating pool (i.e. before crossover)")
		for x in offspring :
			print(chromosomeToStr(x))
	
	# --------------------------------------------------------------------------------------
	# CROSSOVER
	# --------------------------------------------------------------------------------------
	
	# Apply crossover and mutation on the offspring
	# we deal with crossoverProbability, so things are a wee bit different
	crossedOverOffspring = []
	while len(crossedOverOffspring) != len(offspring) :
		# pick 2 parents
		parent1_idx = np.random.randint(0, len(offspring))
		parent2_idx = parent1_idx
		while parent2_idx == parent1_idx :
			parent2_idx = np.random.randint(0, len(offspring))

		if np.random.random() <= crossoverProb:
			child1 = deepcopy(offspring[parent1_idx])
			child2 = deepcopy(offspring[parent2_idx])
			del child1.fitness.values
			del child2.fitness.values

			if printEAprogress :
				print("mate parents", parent1_idx, "and ", parent2_idx,"!")
				print("parents:", type(child1), type(child2))
				print(chromosomeToStr(child1))
				print(chromosomeToStr(child2))
				print("-")
			
			toolbox.crossover(child1, child2)
			
			if printEAprogress :
				print("offspring")
				print(type(child1), type(child2))
				print(chromosomeToStr(child1))
				print(chromosomeToStr(child2))
				
			crossedOverOffspring.append(child1)
			crossedOverOffspring.append(child2)
			
	offspring = deepcopy(crossedOverOffspring)
	
	if printEAprogress :
		print("offspring post crossover pre mutation")
		for x in offspring :
			print(chromosomeToStr(x))
	
	# --------------------------------------------------------------------------------------
	# MUTATION
	# --------------------------------------------------------------------------------------
	
	for i in range(len(offspring)) :
		m = toolbox.mutate(offspring[i])
		if m :
			del offspring[i].fitness.values
			
	if printEAprogress :
		print("offspring post mutation pre repair")
		for x in offspring :
			print(chromosomeToStr(x))
	
	# --------------------------------------------------------------------------------------
	# REPAIR CHROMOSOME
	# --------------------------------------------------------------------------------------
	
	for i in range(len(offspring)) :
		repairChromosome(offspring[i])
	
	if printEAprogress :
		print("offspring post repair pre fitness calculation")
		for x in offspring :
			print(chromosomeToStr(x))
		
	# --------------------------------------------------------------------------------------
	# FITNESS CALCULATION
	# --------------------------------------------------------------------------------------
	
	# Evaluate the entire offrpsing population against the archive+current population!
	offBehaviour = [getChromosomeBehaviour(x)[0] for x in offspring]
	
	if printEAprogress :
		print("offspring behaviour") 
		for x in offBehaviour :
			print(x)
		print("population behaviour")
		for x in popBehaviour :
			print(x)
		print("archive") 
		for x in behaviourArchive :
			print(x)
	
	offFits = [toolbox.fitnessFunction_noveltySearch(i, offspring, offBehaviour, behaviourArchive + popBehaviour) 
			   for i in range(len(offspring))]

	for ind, fit in zip(offspring, offFits):
		ind.fitness.values = [fit]
	
	if printEAprogress :
		print("offspring post fitness pre merging")
		for i in range(len(offspring)) :
			print(chromosomeToStr(offspring[i]), offBehaviour[i], offspring[i].fitness.values)
	 
	# --------------------------------------------------------------------------------------
	# FINALISATION ELITISM + BEST OFFSPRINGS
	# --------------------------------------------------------------------------------------
	
	if printEAprogress :
		print("population and offsprings PRE merging, eliteSize=", eliteSize)
		print("population")
		for i in range(len(pop)) :
			print(i, chromosomeToStr(pop[i]), popBehaviour[i],  pop[i].fitness.values)
		print("offpsring")
		for i in range(len(offspring)) :
			print(i, chromosomeToStr(offspring[i]), offBehaviour[i],  offspring[i].fitness.values)
			
	elitePop = tools.selBest(pop, eliteSize)
	bestOffspring = tools.selBest(offspring, len(pop) - eliteSize)
	
	if printEAprogress :
		eliteBehaviour = [getChromosomeBehaviour(x)[0] for x in elitePop]
		bestOffspringBehaviour = [getChromosomeBehaviour(x)[0] for x in bestOffspring]
	
		print("ELITE")
		for i in range(len(elitePop)) :
			print(i, chromosomeToStr(elitePop[i]), eliteBehaviour[i], elitePop[i].fitness.values)
		print("BEST", len(bestOffspring), "OFFSPRINGS")
		for i in range(len(bestOffspring)) :
			print(chromosomeToStr(bestOffspring[i]), bestOffspringBehaviour[i], bestOffspring[i].fitness.values)
			
	pop = elitePop + bestOffspring
	popBehaviour = [getChromosomeBehaviour(x)[0] for x in pop]
	
	if printEAprogress :
		print("merged population (pre recalculated fitness)")
		for i in range(len(pop)) :
			print(chromosomeToStr(pop[i]), popBehaviour[i], pop[i].fitness.values)
	
	# we recalculate the fitness score because we now ignore the offsprings
	fits = [toolbox.fitnessFunction_noveltySearch(i, pop, popBehaviour, behaviourArchive) for i in range(len(pop))]
	for ind, fit in zip(pop, fits):
		ind.fitness.values = [fit]

	if printEAprogress :
		print("final merged population (with recalculated fitness)")
		for i in range(len(pop)) :
			print(chromosomeToStr(pop[i]), popBehaviour[i], pop[i].fitness.values)
	
	return pop, popBehaviour
	
def fitnessFunction_noveltySearch(idx, pop, popBehaviour, archive, K=2, printDebug=False) :
	# 1. extract the behaviour of x, corresponding to the matching/mismatching wrt the dataset
	if printDebug :
		print("population and behaviour (idx=", idx,")")
		for i in range(len(pop)) :
			ind = pop[i]
			if i == idx :
				print("* ", end='')
			else :
				print("  ", end='')
			print(chromosomeToStr(ind, rounding=2), popBehaviour[i])
			
		print("behaviourArchive")
		for arc in archive :
			print(arc)
	
	xBehaviour = np.asarray(popBehaviour[idx])
	
	# join the archive and the popbehaviour but not the current individual
	popAndArchive = popBehaviour[:idx] + popBehaviour[idx+1 :] + archive
	# optimise the popAndArchive
	uniqPopAndArchive = np.asarray([list(x) for x in list(set(tuple(row) for row in popAndArchive))])
	
	
	if printDebug :
		print("chromosome behaviour")
		print(xBehaviour)
		print("restof behaviours")
		for r in popAndArchive :
			print(r)
		print("unique rest of behaviour")
		for r in uniqPopAndArchive :
			print(r)
		
	# apply knn
	k = K
	if len(uniqPopAndArchive) == 1 :
		k = 1
	
	classifier = NearestNeighbors(k)
	classifier.fit(popAndArchive)
	dList, nList = classifier.kneighbors(xBehaviour.reshape(1, -1))
	
	# this is the novelty score, i.e. the mean of the k-nn
	ro = np.mean(dList)
	
	if printDebug :
		print("knn with k=", k, "returned:")
		print("dList =", dList)
		print("nList =", nList)
		print("i.e. rpo =", ro)	
		
	return ro
	
def crossover(x1, x2, crossoverProb, printDebug=False) :
	global toolbox
	# crossover is uniform in terms of subgenes. for now we do not perform crossover of allele values of different subgenes
	if printDebug :
		print("crossover of these 2 individuals")
		print("crossoverProb=", crossoverProb)
		print(chromosomeToStr(x1))
		print(chromosomeToStr(x2))
	# we need to represent x1 and x2 as lists of subgenes
	chSplit1, _ = splitChromosome(x1)
	chSplit2, _ = splitChromosome(x2)
	tmpX1 = [chSplit1[k] for k in chSplit1]
	tmpX2 = [chSplit2[k] for k in chSplit2]
	
	# now the crossover
	o1, o2 = toolbox.crossoverSubGene(tmpX1, tmpX2)
	
	# now we rebuild the offsprings
	o1 = [item for sublist in o1 for item in sublist]
	o2 = [item for sublist in o2 for item in sublist]
	if printDebug :
		print("crossed over into")
		print(chromosomeToStr(o1))
		print(chromosomeToStr(o2))
	
	x1[0:len(x1)] = o1[0:len(o1)]
	x2[0:len(o2)] = o2[0:len(o2)]
	
def EA_algorithm(globalLoop, pop, behaviourArchive, nGen, printEAprogress=False) :
	
	# Evaluate the entire population against the archive
	popBehaviour = [getChromosomeBehaviour(x)[0] for x in pop]
	fits = [toolbox.fitnessFunction_noveltySearch(i, pop, popBehaviour, behaviourArchive) for i in range(len(pop))]

	for ind, fit in zip(pop, fits):
		ind.fitness.values = [fit]

	if printEAprogress :
		print("population at loop", globalLoop, "(idx, chromosome, behaviour, fitnesses) PRE EVOLUTION")
		for i in range(populationSize) :
			print(chromosomeToStr(pop[i]),
				  popBehaviour[i], pop[i].fitness.values)
	
	for g in range(nGen) :
		#print("EVOLUTION GENERATION", (g+1), '/', nGen, 10*' ', end='\r')
		pop, popBehaviour = EA_generation(pop, popBehaviour, behaviourArchive, printEAprogress)
			
	if printEAprogress :
		print("\npopulation at loop", globalLoop, "(idx, chromosome, behaviour, fitnesses) POST EVOLUTION")
		for i in range(populationSize) :
			print(chromosomeToStr(pop[i]), popBehaviour[i], pop[i].fitness.values)
	
	return pop, popBehaviour