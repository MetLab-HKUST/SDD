; This script extract South China rainfall and generate data that can be used for
; machine learning.
;   The region of focus is 20N to 25N, 110E to 120E, slightly larger than the
; bounds of the Guangdong.

begin

  files = systemfunc("ls APHRO_V1901_MA/data/APHRO_MA_025deg_V1901*.nc")
  in = addfiles(files, "r")

  precip = in[:]->precip(:, {20.0:25.0}, {110.0:120.0})
  printVarSummary(precip)
  
  time = in[:]->time
  date = cd_calendar(time, -2)
  lon = in[0]->lon
  lat = in[0]->lat

  out = addfile("SouthChina_Rainfall_APHRO_V1901_MA.nc", "c")
  out->precip = precip 
  out->date = date 
  
end 