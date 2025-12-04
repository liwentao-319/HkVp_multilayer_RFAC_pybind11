import os
from glob import glob

stations = glob("seismic_data/2F_TB*")
for station in stations:
    files = glob(f"{station}/*.BHR.sac")
    for file in files:
        os.rename(file,file.replace(".BHR.",".HHR."))
    files = glob(f"{station}/*.BHT.sac")
    for file in files:
        os.rename(file,file.replace(".BHT.",".HHT."))
    files = glob(f"{station}/*.BHZ.sac")
    for file in files:
        os.rename(file,file.replace(".BHZ.",".HHZ."))
