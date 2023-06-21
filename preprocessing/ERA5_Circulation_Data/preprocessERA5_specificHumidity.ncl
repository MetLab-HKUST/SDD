begin 

  precipFile = addfile("SouthChina_Precip_4DL.nc", "r")
  prDate = precipFile->date 
  prLon = precipFile->lon 
  prLat = precipFile->lat

  ; Get raw EAR5 data files
  era5Files = systemfunc("ls ERA5_SpecificHumidity_*.nc")
  ;==================================================
  in = addfiles(era5Files, "r")
  ; Link and get the variable
  var = in[:]->q
  ;================================================== 
  lon = in[0]->LONGITUDE 
  lat = in[0]->LATITUDE 
  time = var&time 
  date = cd_calendar(time, -2)
  varSiz = dimsizes(var)

  ; Next we loop through all precipitation samples' locations 
  ; and interpolate ERA5 data on a 48x48 grid centered at the 
  ; same location
  numSamples = dimsizes(prDate)
  system("echo loops to go " + numSamples(0))
  
  var4DL = new((/numSamples(0), varSiz(1), 48, 48/), float)
  var4DL!0 = "sample"
  var4DL!1 = "level"
  var4DL!2 = "lat"
  var4DL!3 = "lon"
  var4DL&sample = ispan(0, numSamples(0)-1, 1)
  var4DL&level = var&level 
  var4DL&lat = fspan(-23.5, 23.5, 48)
  var4DL&lon = fspan(-23.5, 23.5, 48) 

  j = 0    ; index for ERA5 data
  preSampDate = prDate(0)
  do i = 0, numSamples(0)-1
    sampLat = prLat(i) 
    sampLon = prLon(i) 
    sampDate = prDate(i)
    if (.not.(sampDate .eq. preSampDate)) then
      j = j + 4
      system("echo date = " + sampDate)
      system("echo \ \ at sample \#" + i)
    end if 
    if (.not.(date(j) .eq. sampDate)) then
      system("echo FATAL ... Dates do not match")
      exit()
    end if
    varDayAve = dim_avg_n(var(j:j+3, :, :, :), 0)
    newLat = fspan(sampLat-23.5, sampLat+23.5, 48)
    newLon = fspan(sampLon-23.5, sampLon+23.5, 48)
    varInterp = linint2(lon, lat, varDayAve, False, newLon, newLat, 0)
    var4DL(i, :, :, :) = (/varInterp/)

    preSampDate = sampDate 
  end do 

  printVarSummary(var4DL)

; write out date in NetCDF format              
  klev  = dimsizes(var4DL&level)                                               
  nlat  = dimsizes(var4DL&lat)  
  nlon  = dimsizes(var4DL&lon)      

  diro = "./"                     
  filo = "DL_ERA5_SpecificHumidity.nc"     
  ;================================    
  system("/bin/rm -f " + diro + filo)    
  fout  = addfile (diro + filo, "c")     

  setfileoption(fout,"DefineMode",True)
  setfileoption("nc", "Format",  "NetCDF4")
  setfileoption(fout, "CompressionLevel", 6)

  fAtt               = True          
  fAtt@creation_date = systemfunc ("date")        
  fileattdef( fout, fAtt )          

  dimNames = (/"sample", "level", "lat", "lon"/)  
  dimSizes = (/ -1   ,  klev, nlat,  nlon/) 
  dimUnlim = (/ True , False, False, False/)   
  filedimdef(fout,dimNames,dimSizes,dimUnlim)

  filevardef(fout, "sample", typeof(var4DL&sample), "sample") 
  filevardef(fout, "level" , typeof(var4DL&level), "level")                           
  filevardef(fout, "lat"  ,  typeof(var4DL&lat), "lat")                          
  filevardef(fout, "lon"  ,  typeof(var4DL&lon), "lon")                          
  filevardef(fout, "Q"    ,  typeof(var4DL), getvardims(var4DL))
  ; ============================================================

  setfileoption(fout, "DefineMode", False)

  fout->sample = (/var4DL&sample/)     
  fout->level  = (/var4DL&level/)
  fout->lat    = (/var4DL&lat/)
  fout->lon    = (/var4DL&lon/) 
  fout->Q      = (/var4DL/)
  ;========================

end  


