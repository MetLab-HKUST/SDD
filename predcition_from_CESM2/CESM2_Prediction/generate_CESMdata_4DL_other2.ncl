; Interpolate CESM data on pressure levels to new 1x1 deg grids,
; making the data ready for deep learning prediction

begin
 
  files = systemfunc("ls ssp585_vwind/*.nc")    ;================
  print(files)

  in = addfiles(files, "r")

  var0 = in[:]->V    ;================
  varSiz = dimsizes(var0)
  printVarSummary(var0) 

  newVar = new((/toint(floor(varSiz(0)/4)), varSiz(1), 52, 57/), float)

  newLat = fspan(-3, 48, 52)
  newLon = fspan(87, 143, 57)
  
  do i = 0, varSiz(0)-2, 4 
    varNow = var0(i:i+3, :, :, :)
    varDayAve = dim_avg_n_Wrap(varNow, 0)
    varNew = linint2(var0&lon, var0&lat, varDayAve, False, newLon, newLat, 0)
    newVar(i/4, :, :, :) = varNew 
  end do
  printVarSummary(newVar)

  ; write out date in NetCDF format              
  klev  = dimsizes(var0&lev_p)                                               
  nlat  = dimsizes(newLat)  
  nlon  = dimsizes(newLon)      

  diro = "./"     
  ;================================                
  filo = "DL_CESM2_SSP585_V_Wind.nc"     
  ;================================    
  system("rm -f " + diro + filo)    
  fout  = addfile (diro + filo, "c")     

  setfileoption(fout,"DefineMode",True)
  setfileoption("nc", "Format",  "NetCDF4")
  setfileoption(fout, "CompressionLevel", 6)

  fAtt               = True          
  fAtt@creation_date = systemfunc ("date")        
  fileattdef( fout, fAtt )          

  dimNames = (/"time", "level", "lat", "lon"/)  
  dimSizes = (/ -1   ,  klev, nlat,  nlon/) 
  dimUnlim = (/ True , False, False, False/)   
  filedimdef(fout, dimNames, dimSizes, dimUnlim)

  filevardef(fout, "time",  typeof(var0&time), "time") 
  filevardef(fout, "level", typeof(var0&lev_p), "level")                           
  filevardef(fout, "lat"  , typeof(newLat), "lat")                          
  filevardef(fout, "lon"  , typeof(newLon), "lon")             
  ; ============================================================             
  filevardef(fout, "V"    ,  typeof(newVar), (/"time", "level", "lat", "lon"/))
  ; ============================================================

  setfileoption(fout, "DefineMode", False)

  fout->time = var0&time(0:varSiz(0)-2:4)      
  fout->level  = (/var0&lev_p/)
  fout->lat    = (/newLat/)
  fout->lon    = (/newLon/)
  ;======================== 
  fout->V      = (/newVar/)
  ;========================

end  



