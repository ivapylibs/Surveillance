#!/usr/bin/python
#================================= design01 ================================
##@file
# @brief  Test configuration node for method instantiation.
# 
# Tests configuration creation, saving, and sub-algorithm instantiation.
# 
# Execution:
# ----------
# Just run the script.  Outputs should be self explanatory.
#
# @ingroup  TestSurveillance_Layers
#
# @author   Patricio A. Vela,       pvela@gatech.edu
# @date     2023/06/30              [created]
#
# @quitf
#
# NOTE:     tabstop = 4, indent = 2, conversion to spaces. 85 columns.
#
#================================= design01 ================================


import Surveillance.layers.PuzzleScene as PS

import detector.bgmodel.inCorner    as bgColor
import detector.bgmodel.onWorkspace as bgDepth
import detector.fgmodel.Gaussian as fgColor

configRaw = PS.CfgPuzzleScene.get_default_settings()
config = PS.CfgPuzzleScene()

print('The config is:')
print(config)
print('---------------')
print('Now saving. YAML file should agree with above text.')

with open('design01.yaml','w') as file:
      file.write(config.dump())
      file.close()

print('Based on the configuration, several algorithm class instances should')
print('be created.  Two background model detectors and one foreground model')
print('detector.\n')

bgDetColor = bgColor.inCorner.buildFromCfg(config.workspace.color)
print(bgDetColor)

bgDetDepth = bgDepth.onWorkspace.buildFromCfg(config.workspace.depth)
print(bgDetDepth)

fgDet = fgColor.Gaussian.buildFromCfg(config.glove)
print(fgDet)

print('\nThese are also instantiated within the PuzzleScene detector instance.')
print('If configuration is good, then below should be a PuzzleScene detector.')
print('instance plus confirmation that detectors instantiated.\n')

layDet = PS.Detectors(config)
print(layDet)
print(layDet.workspace)
print(layDet.depth)
print(layDet.glove)

#
#================================= design01 ================================
