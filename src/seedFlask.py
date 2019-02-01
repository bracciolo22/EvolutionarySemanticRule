#----------------------------------------------------------------------------------------------------------------------
# IMPORTS
#----------------------------------------------------------------------------------------------------------------------

from flask import Flask, request, render_template, session, redirect, url_for, escape

import os
import secrets

import datetime

import io
import base64

import logging
from logging.handlers import RotatingFileHandler

import cv2

import numpy as np

from seedUtils import seedUtils

from deap import tools

from copy import deepcopy

#----------------------------------------------------------------------------------------------------------------------
# GLOBALS
#----------------------------------------------------------------------------------------------------------------------

debugMode = False
hasInitialised = False

resultsFolder = "evolution" 
baseResultsFolder = ".."
resultsPath = baseResultsFolder + "/" + resultsFolder + "/"

app = Flask(__name__)
app.secret_key = os.urandom(24)

menu = ['startEA']

userData = {}

dataset_rgb = []
dataset_mu = []

#----------------------------------------------------------------------------------------------------------------------
# RENDERING FUNCTIONS
#----------------------------------------------------------------------------------------------------------------------
	
def renderDatasetImg(image, title=None) :
	ble = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	ret, buffer = cv2.imencode('.png', ble)
	plot_url = base64.b64encode(buffer).decode()
					
	return plot_url
	
def renderDatasetImgMuRGB(muR, muG, muB, title=None) :
	ble = np.asarray([[[muR, muG, muB]]], dtype=np.uint8)
	ble = cv2.cvtColor(ble, cv2.COLOR_RGB2BGR)
	ret, buffer = cv2.imencode('.png', ble)
	plot_url = base64.b64encode(buffer).decode()
	
	return plot_url
	
def renderChromosome2(x) :
	chromosomeSplit, _ = seedUtils.splitChromosome(x)
	perimeter, bbox, colour, alpha = seedUtils.chromosome2image(x)
	isShaped = chromosomeSplit[seedUtils.chromosomeOrder.index('isShaped')][0]
	isColoured = chromosomeSplit[seedUtils.chromosomeOrder.index('isColoured')][0]

	if isShaped :
		p2 = deepcopy(perimeter)
		p2[0] = p2[0] - bbox['x0']
		p2[1] = p2[1] - bbox['y0']
		p2[0] = p2[0] / bbox['width']
		p2[1] = p2[1] / bbox['height']
		p2 *= 99
		p2 = np.uint(p2)

		if isColoured :
			colour = list(np.asarray(colour)*255)
		else :
			colour = None

		ble = seedUtils.cv2fyV2(p2, 100, 100, 
								bgColour=(255,255,255), 
								perimeterColour=(0,0,0), fillColour=colour)

	else :
		ble = np.asarray(255.0 * np.asarray(colour), dtype=np.uint8)
		ble = np.asarray([[ble]], dtype=np.uint8)
		
	ret, buffer = cv2.imencode('.png', ble)
	plot_url = base64.b64encode(buffer).decode()

	return plot_url
	
#----------------------------------------------------------------------------------------------------------------------
# SAVING RESULTS
#----------------------------------------------------------------------------------------------------------------------

def saveEvolutionResult(loopIndex, selectedChromosomeIdx, chromosomeStr, chromosomeBehaviour) :

	global resultsPath
	
	f = open(resultsPath + session['username'] + "_" + session['nickname'], "a")
	f.writelines(str(loopIndex) + "," + str(datetime.datetime.now()) + "," + str(selectedChromosomeIdx) + "," + chromosomeStr + "," + ";".join([str(x) for x in chromosomeBehaviour]) + "\n")
	f.close()
	
#----------------------------------------------------------------------------------------------------------------------
# STARTUP
#----------------------------------------------------------------------------------------------------------------------

def startApp() :
	global dataset_rgb, dataset_mu, hasInitialised, resultsPath
	
	hasInitialised = True
	
	print("initialising app")
	
	if resultsFolder not in os.listdir(baseResultsFolder) :
		os.mkdir(resultsPath)

	seedUtils.initDEAP()
	seedUtils.loadDataset(True)
	
	dataset_rgb = [renderDatasetImg(img, "original") for img in seedUtils.dataset_rgb]
	
	dataset_mu = []
	for i in range(len(seedUtils.dataset)) :
		muR = seedUtils.dataset_muR[i]
		muG = seedUtils.dataset_muG[i]
		muB = seedUtils.dataset_muB[i]
		dataset_mu.append(renderDatasetImgMuRGB(muR, muG, muB, "muRGB"))
		
#----------------------------------------------------------------------------------------------------------------------
# PAGES
#----------------------------------------------------------------------------------------------------------------------

@app.route('/')
def index() :
	global menu, resultsPath, userData, hasInitialised, debugMode
	
	if not hasInitialised :
		startApp()
	
	if 'username' in session:
	
		print(session['username'], session['nickname'])
	
		# init userData and results file
		userData[session['username']] = {
			'nickname' : session['nickname'],
			'population' : [],
			'populationBehaviour' : [],
			'behaviourArchive' : []
		}
	
		f = open(resultsPath + session['username'] + "_" + session['nickname'], "a")
		f.write("loop,datetime,chromosomeIndex,genotype,behaviour\n")
		f.close()
		
		if debugMode :
			return render_template('hello.html', debugMode=debugMode, menu=menu, datetime = str(datetime.datetime.now()))
		else :
			return redirect(url_for('startEA'))
	
	return redirect(url_for("login"))
	
@app.route('/login', methods=['GET', 'POST'])
def login():
	global debugMode

	if request.method == 'POST':
		session['username'] = request.form['username']
		session['nickname'] = request.form['nickname']
		return redirect(url_for('index'))
		
	hashCode = secrets.token_hex(nbytes=16)
	
	return render_template('login.html', debugMode=debugMode, hashCode=hashCode)
		
@app.route('/logout')
def logout():
	# remove the username from the session if it's there
	session.pop('username', None)
	return redirect(url_for('index'))
	
@app.route('/startEA')
def startEA() :
	global dataset_rgb, dataset_mu, userData

	pop, popB = seedUtils.createPopulation()
	
	userData[session['username']]['population'] = deepcopy(pop)
	userData[session['username']]['populationBehaviour'] = deepcopy(popB)
	
	populationStr = [seedUtils.chromosomeToStr(x) for x in userData[session['username']]['population']]
	
	populationImg = [renderChromosome2(x) for x in userData[session['username']]['population']]
	
	populationBehaviourStr = ["".join(str(x)) for x in userData[session['username']]['populationBehaviour']]
	
	tmp_behaviour = [seedUtils.getChromosomeBehaviour(x) for x in userData[session['username']]['population']]
	
	populationOverlapRatio = [tmp_behaviour[i][1] for i in range(len(userData[session['username']]['population']))]
	tmp_populationOverlapImg = [tmp_behaviour[i][2] for i in range(len(userData[session['username']]['population']))]
	populationColourRatio = [tmp_behaviour[i][3] for i in range(len(userData[session['username']]['population']))]
	
	populationOverlapThredhold = []
	populationColourThreshold = []
	populationOverlapImg = []
	
	for i in range(len(userData[session['username']]['population'])) :
		xSplit, _ = seedUtils.splitChromosome(userData[session['username']]['population'][i])
		th_i = xSplit[seedUtils.chromosomeOrder.index('shapeMatchingOverlap')]
		th_c = xSplit[seedUtils.chromosomeOrder.index('colourMatchingOverlap')]
		populationOverlapThredhold.append(np.round(th_i,2))
		populationColourThreshold.append(np.round(th_c,2))
		
		poi = []
		j = 0
		for _ in range(len(tmp_populationOverlapImg[i])) :
			if tmp_populationOverlapImg[i][j] is None :
				poi.append(None)
			else :
				try :
					ret, buffer = cv2.imencode('.png', tmp_populationOverlapImg[i][j])
					plot_url = base64.b64encode(buffer).decode()
					poi.append(plot_url)
				except ValueError :
					print("FECK", session['username'], "i=", i, "j=", j)
					j -= 1
					continue
			j += 1
					
		populationOverlapImg.append(poi)
		
	return render_template('EAinterface_toggle.html', 
							debugMode=debugMode,
							loopIndex = 0,
							population = userData[session['username']]['population'],
							populationBehaviour = userData[session['username']]['populationBehaviour'],
							dataset_rgb = dataset_rgb,
							dataset_mu = dataset_mu,
							populationStr = populationStr, 
							populationImg = populationImg,
							populationBehaviourStr = populationBehaviourStr,
							populationOverlapRatio = populationOverlapRatio,
							populationOverlapThredhold = populationOverlapThredhold,
							populationColourThreshold = populationColourThreshold,
							populationOverlapImg = populationOverlapImg,
							populationColourRatio = populationColourRatio,
							showImageOverlap = True,
							showColourOverlap = True)
	
@app.route('/userInput', methods=['GET', 'POST'])
def userInput() :

	global dataset_rgb, dataset_mu, resultsPath
	
	showImageOverlap = request.form.get("image_overlap") != None
	showColourOverlap = request.form.get("colour_overlap") != None
	
	selectedChromosomeIdx = int(request.form['submit'])
	loopIndex = int(request.form['loopIndex'])
	
	#--------------------------------------------------------------------------------------
	# PRESERVING WINNER
	#--------------------------------------------------------------------------------------
	
	chromosome = userData[session['username']]['population'][selectedChromosomeIdx]
	chromosomeStr = seedUtils.chromosomeToStr(chromosome)
	chromosomeBehaviour = userData[session['username']]['populationBehaviour'][selectedChromosomeIdx]
	
	saveEvolutionResult(loopIndex, selectedChromosomeIdx, chromosomeStr, chromosomeBehaviour)
	
	loopIndex += 1
	
	#--------------------------------------------------------------------------------------
	# EXPLOITING
	#--------------------------------------------------------------------------------------
	
	# i.e. just mutation
	exploitedPopulation, exploitedPopulationBehaviour = seedUtils.exploitChromosome(
															userData[session['username']]['population'][selectedChromosomeIdx], 
															userData[session['username']]['populationBehaviour'][selectedChromosomeIdx])
		
	#--------------------------------------------------------------------------------------
	# EXPORING
	#--------------------------------------------------------------------------------------
	
	# i.e. novelty search
	exploredPopulation, exploredPopulationBehaviour = seedUtils.EA_algorithm(-1, 
														userData[session['username']]['population'], 
														userData[session['username']]['behaviourArchive'], 
														seedUtils.nEvolutionaryGenerationsPerLoop)   
														
	#--------------------------------------------------------------------------------------
	# MERGING: exploit + winner + explored
	#--------------------------------------------------------------------------------------
	
	# i.e. combine the two sets into the new population: exploited + best of population
	bestExploredPopulation = tools.selBest(exploredPopulation, seedUtils.nExploreChromosomes)
	bestExploredPopulationBehaviour = [seedUtils.getChromosomeBehaviour(x)[0] for x in bestExploredPopulation]
	
	userData[session['username']]['population'] = deepcopy(exploitedPopulation + [chromosome] + bestExploredPopulation[:-1])
	userData[session['username']]['populationBehaviour'] = deepcopy(exploitedPopulationBehaviour + [chromosomeBehaviour] + bestExploredPopulationBehaviour[:-1])
	
	populationStr = [seedUtils.chromosomeToStr(x) for x in userData[session['username']]['population']]	
	populationImg = [renderChromosome2(x) for x in userData[session['username']]['population']]
	populationBehaviourStr = ["".join(str(x)) for x in userData[session['username']]['populationBehaviour']]
	
	#--------------------------------------------------------------------------------------
	# UPDATE ARCHIVE
	#--------------------------------------------------------------------------------------
	
	ble = userData[session['username']]['behaviourArchive'] + userData[session['username']]['populationBehaviour']
	bleStr = [z.split('-') for z in list(set(['-'.join([str(x) for x in y]) for y in ble]))]
	ble = [[int(x) for x in y] for y in bleStr]
	
	userData[session['username']]['behaviourArchive'] = deepcopy(ble)
	
	#--------------------------------------------------------------------------------------
	# FINALISATION
	#--------------------------------------------------------------------------------------
	
	tmp_behaviour = [seedUtils.getChromosomeBehaviour(x) for x in userData[session['username']]['population']]
	
	populationOverlapRatio = [tmp_behaviour[i][1] for i in range(len(userData[session['username']]['population']))]
	tmp_populationOverlapImg = [tmp_behaviour[i][2] for i in range(len(userData[session['username']]['population']))]
	populationColourRatio = [tmp_behaviour[i][3] for i in range(len(userData[session['username']]['population']))]
	
	populationOverlapThredhold = []
	populationColourThreshold = []
	populationOverlapImg = []
	
	for i in range(len(userData[session['username']]['population'])) :
	
		xSplit, _ = seedUtils.splitChromosome(userData[session['username']]['population'][i])
		th_i = xSplit[seedUtils.chromosomeOrder.index('shapeMatchingOverlap')]
		th_c = xSplit[seedUtils.chromosomeOrder.index('colourMatchingOverlap')]
		populationOverlapThredhold.append(th_i)
		populationColourThreshold.append(th_c)
		
		poi = []
		j = 0
		for _ in range(len(tmp_populationOverlapImg[i])) :
			if tmp_populationOverlapImg[i][j] is None :
				poi.append(None)
			else :
				#continue
				try :
					ret, buffer = cv2.imencode('.png', tmp_populationOverlapImg[i][j])
					plot_url = base64.b64encode(buffer).decode()
					poi.append(plot_url)
				except ValueError :
					print("FECK", session['username'], "i=", i, "j=", j)
					j -= 1
					continue
			j += 1
					
		populationOverlapImg.append(poi)
	
	
	return render_template('EAinterface_toggle.html', 
							debugMode=debugMode,
							loopIndex = loopIndex,
							population = userData[session['username']]['population'],
							populationBehaviour = userData[session['username']]['populationBehaviour'],
							dataset_rgb = dataset_rgb,
							dataset_mu = dataset_mu,
							populationStr = populationStr, 
							populationImg = populationImg,
							populationBehaviourStr = populationBehaviourStr,
							populationOverlapRatio = populationOverlapRatio,
							populationOverlapThredhold = populationOverlapThredhold,
							populationColourThreshold = populationColourThreshold,
							populationOverlapImg = populationOverlapImg,
							populationColourRatio = populationColourRatio,
							showImageOverlap = showImageOverlap,
							showColourOverlap = showColourOverlap)

	

#----------------------------------------------------------------------------------------------------------------------
# MAIN
#----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

	app.debug = debugMode
	
	startApp()
	
	app.run("0.0.0.0", 5000)