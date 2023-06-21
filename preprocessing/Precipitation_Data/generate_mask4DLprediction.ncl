; This script generate land mask for the South China region, for use
; later in CESM prediction with deep learning.


begin

  in = addfile("SouthChina_Rainfall_APHRO_V1901_MA.nc", "r")
  precip = in->precip 
  printVarSummary(precip)
  lon = precip&lon 
  lat = precip&lat 
  date = in->date

  landMask = precip(0, ::4, ::4)
  latMask = lat(::4)
  lonMask = lon(::4)
  landMask@long_name = "Land mask; 1: land, 0: ocean"
  landMask@units = ""
  printVarSummary(landMask)
  
  do j = 0, 4
    do i = 0, 9
      precip4pt = ndtooned(precip(100, j*4:j*4+3, i*4:i*4+3))
      lon4ptCen = doubletofloat(avg(lon(i*4:i*4+3)))
      lat4ptCen = doubletofloat(avg(lat(j*4:j*4+3)))
      numElements = dim_num_n(.not.ismissing(precip4pt), 0)
      latMask(j) = lat4ptCen
      lonMask(i) = lon4ptCen
      if (numElements.ge.3) then
        landMask(j, i) = 1
      else
        landMask(j, i) = 0
      end if
    end do
  end do

  landMask&lat = latMask
  landMask&lon = lonMask

  totalNum = sum(ndtooned(landMask))
  landMask@total_land_points = totalNum
  
  printVarSummary(landMask)
  
  out = addfile("SouthChina_LandMask_4DL.nc", "c")
  out->land_mask = landMask

end 

