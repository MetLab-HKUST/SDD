begin

  in = addfile("SouthChina_Precip_4DL.nc", "r")
  precipAvg = in->precipAvg 
  precipVar = in->precipVar
  precipMax = in->precipMax
  
  out = addfile("DL_Precip_4NormalMax_Dist.nc", "c")
  out->precipAvg = precipAvg
  out->precipVar = precipVar
  out->precipMax = precipMax

end 