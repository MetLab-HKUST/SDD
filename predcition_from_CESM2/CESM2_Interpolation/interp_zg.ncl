; This script reads the CESM data and interpolates it to the desired pressure
; levels, making the data usable by the DL model.

begin 

  files = systemfunc("ls -1 CESM2/ssp585/zg/*nc")
  taFiles = systemfunc("ls -1 CESM2/ssp585/ta/*nc")
  numFiles = dimsizes(files)

  do i = 0, numFiles(0)-1
    system("echo ========")
    system("echo working on file\#\ " + (i+1) + " / " + numFiles(0))
    system("echo \ \ " + files(i))
    system("echo \ \ " + taFiles(i)) 

    in := addfile(files(i), "r")
    inta :=  addfile(taFiles(i), "r")
    tbot := inta->ta(:, 0, {-7.0:52.0}, {83.0:147.0})
    zg0 := in->Zg  ;(:, :, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(zg0)
    zg := zg0(:, ::-1, :, :)
    delete(zg0)   
    a = inta->a(::-1) 
    b = inta->b(::-1) 
    ps := inta->ps(:, {-7.0:52.0}, {83.0:147.0}) 
    p0 = (inta->p0) * 0.01

    in = addfile("CESM2/orog_fx_CESM2_historical_r11i1p1f1_gn.nc", "r")
    phis = (in->orog({-7.0:52.0}, {83.0:147.0})) * 9.8

    lev_p = (/250.0, 500.0, 600.0, 700.0, 850.0, 925.0/)
    varflg = -1
    ; variable to interpolate: 1 = temperature, -1 = geopotential height, 0 = all others. 

    Z := vinth2p_ecmwf(zg, a, b, lev_p, ps, 1, p0, 1, True, varflg, tbot, phis) 
    Z&time = tbot&time 
    
    date := cd_calendar(tbot&time, -3)
    dateLen = dimsizes(date)
    dateStart = date(0)
    dateEnd = date(dateLen(0)-1)

    fileName = "ssp585_geopotential_height/CESM2_SSP585_Geopotential_Height_" + dateStart + "to" + dateEnd + ".nc"
    system("echo \ \ writing\ " + fileName)

    out := addfile(fileName, "c")
    out->Zg = Z
  end do 

end 