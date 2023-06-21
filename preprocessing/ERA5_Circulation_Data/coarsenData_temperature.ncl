begin

  files = systemfunc("ls ERA-5/temperature/adaptor*nc")
  numFiles = dimsizes(files)
  print(numFiles)

  do f = 0, numFiles(0)-1
    system("echo working on file\#" + f + " ...") 
    in = addfile(files(f), "r")
    varShort := in->t
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

    fileName = "ERA5_Temperature_" + dateStart + "to" + dateEnd + ".nc"

    out =  addfile(fileName, "c")
    out->t = newVar
  end do

end

