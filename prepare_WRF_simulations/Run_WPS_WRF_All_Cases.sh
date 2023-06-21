
#!/bin/bash

shopt -s extglob
# loop through all case directory
for d in $PWD/Present_Case*/ ; do   
    CaseDir=$d
    echo -\ working on\ $d
    cd $d
    ./cam2wrf.exe
    
    ln -s $CaseDir/FILE* /home/shixm/Lab/WPS/
    cp namelist.wps /home/shixm/Lab/WPS
    cp namelist.input /home/shixm/Lab/WRF/run

    cd /home/shixm/Lab/WPS
    ./geogrid.exe
    mpirun -np 24 ./metgrid.exe

    cd /home/shixm/Lab/WRF/run
    ln -s /home/shixm/Lab/WPS/geo_em*nc .
    ln -s /home/shixm/Lab/WPS/met_em*nc .
    mpirun -np 24 ./real.exe

    mv wrfinput* $d
    mv wrflow* $d
    mv wrfbdy* $d
    rm met_em*nc
    rm geo_em*nc
    rm /home/shixm/Lab/WPS/met_em*nc
    rm /home/shixm/Lab/WPS/geo_em*nc
    rm /home/shixm/Lab/WPS/FILE*

    cd $d
    rm CAM*.nc
    rm CLM*.nc
    rm FILE*
done
    
    
