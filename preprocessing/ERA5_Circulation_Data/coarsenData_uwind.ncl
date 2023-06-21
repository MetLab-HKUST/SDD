begin

  files = systemfunc("ls ERA-5/u_wind/adaptor*nc")
  numFiles = dimsizes(files)
  print(numFiles)

  do f = 0, numFiles(0)-1
    system("echo working on file\#" + f + " ...") 
    in = addfile(files(f), "r")
    varShort := in->u
    printVarSummary(varShort)

    var := short2flt(varShort)
    printVarSummary(var)

    newLat = fspan(-5.5, 50.5, 57)
    ; print(newLat)
    newLon = fspan(84.5, 145.5, 62)
    ; print(newLon)

    newVar := area_hi2lores_Wrap(var&longitude, var&latitude, var, False, 1.0, newLon, newLat, False)
    printVarSummary(newVar)

    date := cd_calendar(var&time, -3)
    dateLen = dimsizes(date)
    dateStart = date(0)
    dateEnd = date(dateLen(0)-1)

    fileName = "ERA5_Uwind_" + dateStart + "to" + dateEnd + ".nc"

    out =  addfile(fileName, "c")
    out->u = newVar
  end do

end

