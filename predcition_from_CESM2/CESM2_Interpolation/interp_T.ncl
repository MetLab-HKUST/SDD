; This script reads the CESM data and interpolates it to the desired pressure
; levels, making the data usable by the DL model.

begin 

  files = systemfunc("ls -1 CESM2/ssp585/ta/*nc")
  numFiles = dimsizes(files)

  do i = 0, numFiles(0)-1 
    system("echo working on file\#\ " + (i+1) + " / " + numFiles(0))

    in := addfile(files(i), "r")
    ta0 := in->ta(:, :, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(ta0)
    ta := ta0(:, ::-1, :, :)
    tbot := ta0(:, 0, :, :)
    delete(ta0)   
    a = in->a(::-1) 
    b = in->b(::-1) 
    ps := in->ps(:, {-7.0:52.0}, {83.0:147.0}) 
    p0 = (in->p0) * 0.01

    in = addfile("CESM2/orog_fx_CESM2_historical_r11i1p1f1_gn.nc", "r")
    phis = (in->orog({-7.0:52.0}, {83.0:147.0})) * 9.8

    lev_p = (/250.0, 500.0, 600.0, 700.0, 850.0, 925.0/)
    varflg = 1 
    ; variable to interpolate: 1 = temperature, -1 = geopotential height, 0 = all others. 

    T := vinth2p_ecmwf(ta, a, b, lev_p, ps, 1, p0, 1, True, varflg, tbot, phis) 

    date := cd_calendar(T&time, -3)
    dateLen = dimsizes(date)
    dateStart = date(0)
    dateEnd = date(dateLen(0)-1)

    fileName = "ssp585_temperature/CESM2_SSP585_Temperature_" + dateStart + "to" + dateEnd + ".nc"
    system("echo \ \ writing\ " + fileName)

    out := addfile(fileName, "c")
    out->T = T
  end do 

end 