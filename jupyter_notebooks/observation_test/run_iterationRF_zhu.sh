bindir=/home/liwentao/packages_install/hk_zhu
cp $bindir/iter_decon bin/iteration_zhu
maindatadir=seismic_data_RFAC

stations=`ls -d ${maindatadir}/TLM_M*`
for station in ${stations[@]}
do

stationname=${station##*/}
gausses=(2 5 7)
Z_files=`ls ${station}/*.BHZ.sac`
for gauss in ${gausses[@]}
do

for Z_file in ${Z_files[@]}
do

./bin/iteration_zhu  -C-5/30/130 -F3/$gauss/-5 -T0.1  -N100 $Z_file  ${Z_file/.BHZ./.BHR.}
#move the Rrf 
mv ${Z_file/.BHZ./.BHR.}i ${Z_file/.BHZ./.BHR.}_g${gauss}


done
done
done

