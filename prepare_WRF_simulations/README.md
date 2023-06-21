## Extract CESM2 time slices and generate WRF's met_em files
The deep learning model predicts the dates when extreme events may happen and the potential precipitation maximum. The NCL scripts here extract the CESM2 variables we need for running a WRF simulation on those dates. 

The script "Run_WPS_WRF_All_Cases.sh" run a program called "cam2wrf.exe" to convert CESM data into intermediate files which can be used by WPS to generate the met_em* files, which are then used by real.exe to generate wrfinput and wrfbdy files.

The cam2wrf code is from Prof. Eli Tziperman <eli@EPS.Harvard.edu>
(https://groups.seas.harvard.edu/climate/eli/Level2/etc.html)
A zipped copy of the source code is deposited here and see "compile.note" for compiling parameters.
