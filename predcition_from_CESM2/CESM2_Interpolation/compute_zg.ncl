; This script reads the CESM data and calculate geopotential height based on
; the hypsometric equation. 

begin 

  husfiles = systemfunc("ls -1 ./ssp585/hus/*nc")
  tafiles = systemfunc("ls -1 ./ssp585/ta/*nc")
  numFiles = dimsizes(tafiles)

  do i = 0, numFiles(0)-1
    system("echo ========")
    system("echo working on file\#\ " + (i+1) + " / " + numFiles(0))

    inhus := addfile(husfiles(i), "r")
    inta :=  addfile(tafiles(i), "r")
    ta := inta->ta(:, :, {-7.0:52.0}, {83.0:147.0})
    tbot := ta(:, 0, :, :)
    hus := inhus->hus(:, :, {-7.0:52.0}, {83.0:147.0})
    tv := temp_virtual(ta, hus, (/1,0,1/))
    tv = tv(:, ::-1, :, :)
    
    a = inta->a 
    b = inta->b
    aa = new(33, double)
    bb = new(33, double)
    a_bnds = inta->a_bnds
    b_bnds = inta->b_bnds
    aa(0:31) = a_bnds(:, 1)
    aa(32) = a_bnds(31, 0)
    bb(0:31) = b_bnds(:, 1)
    bb(32) = b_bnds(31, 0)
    
    ps := inta->ps(:, {-7.0:52.0}, {83.0:147.0}) 
    p0 = (inta->p0) 

    in = addfile("./orog_fx_CESM2_historical_r11i1p1f1_gn.nc", "r")
    phis = (in->orog({-7.0:52.0}, {83.0:147.0})) * 9.8

    Z := cz2ccm(ps, phis, tv, p0, a, b, aa, bb)
    Z = Z(:, ::-1, :, :)
    copy_VarCoords(ta, Z)
    
    printVarSummary(Z)
    ; note that for deep learning it is needed to be geopotential, Z = Z * 9.8

    date := cd_calendar(Z&time, -3)
    dateLen = dimsizes(date)
    dateStart = date(0)
    dateEnd = date(dateLen(0)-1)

    filo = "./ssp585/zg/CESM2_SSP585_Geopotential_Height_" + dateStart + "to" + dateEnd + ".nc"
    system("echo \*\* writing\ " + filo)

    ; write out date in NetCDF format              
    klev  = dimsizes(Z&lev)                                               
    nlat  = dimsizes(Z&lat)  
    nlon  = dimsizes(Z&lon)      

    ;================================    
    system("/bin/rm -f " + filo)    
    fout  = addfile(filo, "c")     

    setfileoption(fout,"DefineMode",True)
    setfileoption("nc", "Format",  "NetCDF4")
    setfileoption(fout, "CompressionLevel", 6)

    fAtt               = True          
    fAtt@creation_date = systemfunc ("date")        
    fileattdef( fout, fAtt )          

    dimNames = (/"time", "lev", "lat", "lon"/)  
    dimSizes = (/ -1   ,  klev, nlat,  nlon/) 
    dimUnlim = (/ True , False, False, False/)   
    filedimdef(fout,dimNames, dimSizes, dimUnlim)

    filevardef(fout, "time", typeof(Z&time),"time") 
    filevardef(fout, "lev" , typeof(Z&lev), "lev")                           
    filevardef(fout, "lat" , typeof(Z&lat), "lat")                          
    filevardef(fout, "lon" , typeof(Z&lon), "lon")                          
    filevardef(fout, "Zg"  , typeof(Z), getvardims(Z))
    ;================================

    setfileoption(fout, "DefineMode", False)

    fout->time = (/Z&time/)     
    fout->lev  = (/Z&lev/)
    fout->lat  = (/Z&lat/)
    fout->lon  = (/Z&lon/) 
    fout->Zg   = (/Z/)
  end do 

end 