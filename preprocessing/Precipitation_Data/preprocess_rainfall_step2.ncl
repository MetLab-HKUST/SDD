begin

  in = addfile("SouthChina_Rainfall_APHRO_V1901_MA.nc", "r")
  precip = in->precip 
  printVarSummary(precip)
  lon = precip&lon 
  lat = precip&lat 
  date = in->date 

  precip4pt = new((/6574*5*10, 16/), float)
  precip4pt@_FillValue = precip@_FillValue
  precip4pt@units = "mm"
  lon4ptCen = new(6574*5*10, float)
  lat4ptCen = new(6574*5*10, float)
  date4pt = new(6574*5*10, typeof(date))

  do day=0,6573
    do j=0,4
      do i=0,9
        precip4pt(day*5*10+j*10+i, :) = ndtooned(precip(day, j*4:j*4+3, i*4:i*4+3))
        lon4ptCen(day*5*10+j*10+i) = doubletofloat(avg(lon(i*4:i*4+3)))
        lat4ptCen(day*5*10+j*10+i) = doubletofloat(avg(lat(j*4:j*4+3)))
        date4pt(day*5*10+j*10+i) = date(day)
      end do
    end do
    print("working on " + date4pt(day*5*10+4*10+9))
  end do

  print(max(lon4ptCen))
  print(min(lon4ptCen))
  ;     119.5
  ;     110.5
  print(max(lat4ptCen))
  print(min(lat4ptCen))
  ;     24.5
  ;     20.5


  numElements = dim_num_n(.not.ismissing(precip4pt), 1)
  keepID = numElements.ge.3
  print("total number of blocks = 328,700")
  print("useful blocks to keep  = " + num(keepID))
  keepID := ind(keepID)
  
  precipBlock := precip4pt(keepID, :)
  precipBlock@_FillValue = precip4pt@_FillValue
  precipBlock@units = "mm"
  precipBlock@long_name = "precip at the block four neighboring points, at least two points have non-mising values"
  printVarSummary(precipBlock)
  
  lon := lon4ptCen(keepID)
  lat := lat4ptCen(keepID)
  date:= date4pt(keepID)

  precipAvg = dim_avg_n_Wrap(precipBlock, 1)
  precipMax = dim_max_n_Wrap(precipBlock, 1)
  precipMin = dim_min_n_Wrap(precipBlock, 1)
  precipVar = dim_variance_n_Wrap(precipBlock, 1)

  out = addfile("SouthChina_Precip_4DL.nc", "c")
  out->precipBlock = precipBlock
  out->precipAvg = precipAvg
  out->precipMax = precipMax
  out->precipMin = precipMin
  out->precipVar = precipVar
  out->lon = lon
  out->lat = lat
  out->date = date 

end 

