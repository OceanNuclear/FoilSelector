#!/bin/sh
if [ $# -eq 0 ]
then
    echo "usage: ./init.sh <NameOfDirectoryToBeCreated>"
else
    mkdir ${1}
    mkdir ${1}/DataPreparation/
    mkdir ${1}/Selection/
    mkdir ${1}/TestEffectiveness/
    cp run_python.sh ${1}
    cp GetMaterialSpeadsheet.py ${1}/DataPreparation/
    cp Filter.py ${1}/Selection/
    cp TestEffectiveness.py ${1}/TestEffectiveness/
    echo Now go and populate ${1}/DataPreparation with your raw data.
    #I need to write test to check that everything works
fi