; Extract CESM2 data and write out to a number of files, each of which contains
; one time slice and will be used for the CAM2WRF program.

; % Period: 2081-2100
; %        FirstDay        LastDay      DailyMax_DL
; %       ___________    ___________    ___________
; % 
; % 1     01-Jun-2099    02-Jun-2099      341.62   
; % 2     18-Aug-2089    20-Aug-2089      272.71   
; % 3     13-Aug-2093    14-Aug-2093       268.8   
; % 4     30-Jun-2100    30-Jun-2100      266.49   
; % 5     29-Jul-2099    29-Jul-2099      248.43   
; % 6     19-Jul-2082    19-Jul-2082      241.87   
; % 7     24-Aug-2086    24-Aug-2086      233.37   
; % 8     06-Aug-2099    06-Aug-2099      226.03   
; % 9     29-Oct-2098    29-Oct-2098       219.3   
; %10     14-Aug-2084    15-Aug-2084      215.68   
; %11     26-Jun-2091    26-Jun-2091      206.61   
; %12     27-Jun-2093    28-Jun-2093      190.88   
; %13     23-Aug-2093    23-Aug-2093      190.85   
; %14     03-Jul-2087    04-Jul-2087      186.97   
; %15     09-Nov-2082    09-Nov-2082      186.96   
; %16     15-Sep-2082    15-Sep-2082      174.02   
; %17     24-Oct-2086    24-Oct-2086      171.12   
; %18     15-Sep-2085    15-Sep-2085      165.34   
; %19     11-Jun-2095    11-Jun-2095      159.74   
; %20     15-Jul-2099    15-Jul-2099      158.69

load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin

  startDates = (/2099053100, 2089081700, 2093081200, 2100062900, 2099072800, 2082071800, 2086082300, 2099080500, 2098102800, 2084081300, 2091062500, 2093062600, 2093082200, 2087070200, 2082110800, 2082091400, 2086102300, 2085091400, 2095061000, 2099071400/)
  endDates   = (/2099060400, 2089082200, 2093081600, 2100070200, 2099073100, 2082072100, 2086082600, 2099080800, 2098103100, 2084081700, 2091062800, 2093063000, 2093082500, 2087070600, 2082111100, 2082091700, 2086102600, 2085091700, 2095061300, 2099071700/)
  ;                  1            2           3          4            5          6            7           8           9          10          11          12          13          14           15          16          17          18          19          20

  hus_files = systemfunc("ls hus/hus_6hrLev_CESM2_ssp585_r11i1p1f1_gn_20[7-9]*nc")
  ta_files  = systemfunc("ls ta/ta_6hrLev_CESM2_ssp585_r11i1p1f1_gn_20[7-9]*nc")
  ua_files  = systemfunc("ls ua/ua_6hrLev_CESM2_ssp585_r11i1p1f1_gn_20[7-9]*nc")
  va_files  = systemfunc("ls va/va_6hrLev_CESM2_ssp585_r11i1p1f1_gn_20[7-9]*nc")
  zg_files  = systemfunc("ls zg/CESM2_SSP585_Geopotential_Height_20[7-9]*.nc")
  psl_files = systemfunc("ls surface_soil_sea/psl_6hrPlevPt_CESM2_ssp585_r11i1p1f1_gn_20[7-9]*.nc")
  tas_files = systemfunc("ls surface_soil_sea/tas_day_CESM2_ssp585_r11i1p1f1_gn_20[7-9]*.nc")
  land_file = "surface_soil_sea/sftlf_fx_CESM2_ssp585_r11i1p1f1_gn.nc"
  tsl_file  = "surface_soil_sea/tsl_Eday_CESM2_ssp585_r11i1p1f1_gn_20650101-21010101.nc"
  tos_file  = "surface_soil_sea/tos_Oday_CESM2_ssp585_r11i1p1f1_gn_20650102-21010101.nc"
  mrsol_file= "surface_soil_sea/mrsol_Eday_CESM2_ssp585_r11i1p1f1_gn_20650101-21010101.nc"

  do icase = 0, 19
    start_date = startDates(icase)
    end_date   = endDates(icase)
    folder_name= "Future_Case" + sprinti("%0.3i", (icase+1)) 
    
    in := addfiles(ta_files, "r")  
    
    time := in[:]->time
    date := cd_calendar(time, -3)
    start_id = ind(date .eq. start_date)
    end_id   = ind(date .eq. end_date)

    ta := in[:]->ta(start_id:end_id, :, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(ta)
    ta_size = dimsizes(ta)
    ta_time := ta&time
    ta_date := date(start_id:end_id)

    dblquote = integertochar(34)  
    utc_date := cd_calendar(ta&time, 0)
    year   := tointeger(utc_date(:,0))    ; Convert to integer for
    month  := tointeger(utc_date(:,1))    ; use sprinti 
    day    := tointeger(utc_date(:,2))
    hour   := tointeger(utc_date(:,3))
    minute := tointeger(utc_date(:,4))
    second := tointeger(utc_date(:,5))
    date_str := dblquote + sprinti("%0.4i", year) + "-" + sprinti("%0.2i", month) + "-" + sprinti("%0.2i", day) + \
      "_" + sprinti("%0.2i", hour) + ":" + sprinti("%0.2i", minute) + ":" + sprinti("%0.2i", second) + \
      dblquote
    date_str_wrf := "'" + sprinti("%0.4i", year) + "-" + sprinti("%0.2i", month) + "-" + sprinti("%0.2i", day) + \
      "_" + sprinti("%0.2i", hour) + ":" + sprinti("%0.2i", minute) + ":" + sprinti("%0.2i", second) + "'"
             
    in := addfiles(ua_files, "r")
    ua := in[:]->ua(start_id:end_id, :, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(ua)
    
    in := addfiles(va_files, "r")
    va := in[:]->va(start_id:end_id, :, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(va)
    
    in := addfiles(zg_files, "r")
    zg := in[:]->Zg(start_id:end_id, :, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(zg)
    
    in := addfiles(hus_files, "r")
    hus := in[:]->hus(start_id:end_id, :, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(hus)
    
    ps := in[:]->ps(start_id:end_id, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(ps)
    ps_size = dimsizes(ps)
    lat := in[0]->lat
    lon := in[0]->lon
    
    p0 := in[0]->p0
    a = in[0]->a
    b = in[0]->b
    printVarSummary(a)
    printVarSummary(b)
    pres := pres_hybrid_ccm(ps, p0, a, b)
    
    q := mixhum_convert(hus, "q", (/0, 0/))
    RH := relhum(ta, q, pres)
    RH@units = "percent"
    printVarSummary(RH)
  
    aa = new(33, double)
    bb = new(33, double)
    a_bnds = in[0]->a_bnds
    b_bnds = in[0]->b_bnds
    aa(0:31) = a_bnds(:, 1)
    aa(32) = a_bnds(31, 0)
    bb(0:31) = b_bnds(:, 1)
    bb(32) = b_bnds(31, 0)

    in := addfile(land_file, "r")
    land_frac := ps    ; copy axis
    land_frac = (/conform(ps, (in->sftlf({-7.0:52.0}, {83.0:147.0}))*0.01, (/1,2/))/)
    land_frac@units = "fraction"
    printVarSummary(land_frac)
    
    in := addfiles(psl_files, "r")
    psl := in[:]->psl(start_id:end_id, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(psl)
    
    ; daily data for the following
    start_date0 := start_date
    end_date0   := end_date
    start_date  := start_date/100 - 1
    end_date    := end_date/100
    in := addfiles(tas_files, "r")
    time := in[:]->time
    date := cd_calendar(time, -2)
    start_id = ind(date .eq. start_date)
    end_id   = ind(date .eq. end_date)
  
    tas0 := in[:]->tas(start_id:end_id, {-7.0:52.0}, {83.0:147.0})
    printVarSummary(tas0)
    tas0_size = dimsizes(tas0)
  
    tas := ps
    xi := fspan(0, tas0_size(0)-1, tas0_size(0)) - 0.5
    xo := fspan(0, (ps_size(0)-1)/4, ps_size(0))
    tas = (/linint1_n(xi, tas0, False, xo, 0, 0)/)
    tas@units = "K"
    printVarSummary(tas)

    in := addfile(tsl_file, "r")
    time := in->time
    date := cd_calendar(time, -2)
    start_id = ind(date .eq. start_date)
    end_id   = ind(date .eq. end_date)
    tsl0 := in->tsl(start_id:end_id, 0:19, {-7.0:52.0}, {83.0:147.0})

    tsl := ta(:, 0:19, {-7.0:52.0}, {83.0:147.0})
    tsl!1 = "depth"
    tsl&depth = in->depth(0:19)
    copy_VarAtts(tsl0, tsl)
    tsl = (/linint1_n(xi, tsl0, False, xo, 0, 0)/)
    printVarSummary(tsl)
  
    in := addfile(tos_file, "r")
    time := in->time
    date := cd_calendar(time, -2)
    start_id = ind(date .eq. start_date)
    end_id   = ind(date .eq. end_date)
    tos0 := in->tos(start_id:end_id, :, :)
    tos0 = tos0 + 273.15
    printVarSummary(tos0)
    ; regrid is needed here because tos is from POP grid
    methods = "patch"
    tos0@lat2d = in->lat
    tos0@lon2d = in->lon
    DstGridName = "CAM"   
    ; specify name of weight file(s) to be generated; name of destination grid; dst directory
    WgtFileDir  = "./"
    WgtFileName = "POP_gx1v3_to_"+DstGridName+"."+methods+".nc"
    Opt                  = True
    Opt@SrcFileName      = "POP_gx1v3_SCRIP.nc"   ; source grid description file name
    Opt@DstFileName      = "CCSM4_SCRIP.nc"       ; destination grid description file name
    Opt@ForceOverwrite   = True
    Opt@SrcMask2D        = where(.not.ismissing(tos0(0,:,:)),1,0)  ; land/ocean mask
    Opt@DstTitle         = "CAM Grid Resolution"
    Opt@DstGridLat       = lat  
    Opt@DstGridLon       = lon
    Opt@PrintTimings     = True
    Opt@WgtFileName      = WgtFileName
    Opt@InterpMethod     = methods
    tos0_regrid := ESMF_regrid(tos0, Opt)
    printVarSummary(tos0_regrid)
  
    ts0 := tas0
    ts0 = (/tsl0(:, 0, {-7.0:52.0}, {83.0:147.0})/)
    ts0 = where(ismissing(tos0_regrid(:, {-7.0:52.0}, {83.0:147.0})), ts0, tos0_regrid(:, {-7.0:52.0}, {83.0:147.0}))
    ts0 = where(ismissing(ts0), tas0, ts0)
    printVarSummary(ts0)
    
    ts := ps
    ts = (/linint1_n(xi, ts0, False, xo, 0, 0)/)
    ts@units = "K"
    printVarSummary(ts)

    in := addfile(mrsol_file, "r")
    time := in->time
    date := cd_calendar(time, -2)
    start_id = ind(date .eq. start_date)
    end_id   = ind(date .eq. end_date)
    mrsol0 := in->mrsol(start_id:end_id, 0:19, {-7.0:52.0}, {83.0:147.0})  
    depth_bnds = in->depth_bnds 
    thickness := depth_bnds(:,1) - depth_bnds(:,0)
    mrsol0 = mrsol0 / conform(mrsol0, thickness, 1)
    printVarSummary(mrsol0)
    
    mrsol := ta(:, 0:19, {-7.0:52.0}, {83.0:147.0})
    mrsol!1 = "depth"
    depth = in->depth(0:19)
    mrsol&depth = depth
    copy_VarAtts(mrsol0, mrsol)
    mrsol@units = "kg m-3"
    mrsol = (/linint1_n(xi, mrsol0, False, xo, 0, 0)/)
    printVarSummary(mrsol)

    num_slices := ta_size(0)

    ; create "CAM" file first
    ;========================
    diro = folder_name
    system("rm -rf " + diro)    ; remove if exists
    system("mkdir " + diro)
    
    do i = 0, num_slices-1
      
      ntim  = 1
      nlev  = dimsizes(ta&lev)                                               
      nlat  = dimsizes(ta&lat)  
      nlon  = dimsizes(ta&lon)      
      
      filo = "CAM_" + ta_date(i) + ".nc"    ; Output file
      fout = addfile(diro + "/" + filo, "c") 
      
      if (i.eq.0) then
        myfolder = systemfunc("pwd")
        cam_filenames := (/dblquote+myfolder+"/"+diro+"/"+filo+dblquote/)
      else
        cam_filenames := array_append_record(cam_filenames, (/dblquote+myfolder+"/"+diro+"/"+filo+dblquote/), 0)
      end if
    
      ; explicitly declare file definition mode. 
      setfileoption(fout, "DefineMode", True)
      
      ; create global attributes of the file
      fAtt               = True         ; assign file attributes
      fAtt@title         = "DL Extreme Case: " + folder_name   
      fAtt@source        = "CESM SSP5-8.5 r11i1p1f1"
      fAtt@creation_date = systemfunc ("date")        
      fileattdef(fout, fAtt)            ; copy file attributes    

      ; predefine the coordinate variables and their dimensionality
      ; Note: to get an UNLIMITED record dimension, we set the dimensionality
      ; to -1 (or the actual size) and set the dimension name to True.
      dimNames = (/"time", "lat", "lon", "lev"/)  
      dimSizes = (/ ntim ,  nlat,  nlon, nlev /) 
      dimUnlim = (/ True , False, False, False/)   
      filedimdef(fout, dimNames, dimSizes, dimUnlim)

      ; predefine the the dimensionality of the variables to be written out

      ; Here we are using NCL functions to facilitate defining 
      ; each variable's dimension name(s) and type. 
      ; The following could be replaced with explicit, user defined dimension 
      ; names different from those associated with the variable in memory. 
      ; Say, PS(time,lat,lon) in the NCL script. They could be redefined for the file via: 
      ; filevardef(fout, "PS"   ,typeof(PS) ,(/"TIME","latitude","longitude"/)) 
      ;===================================================================
      filevardef(fout, "time", typeof(ta&time), getvardims(ta&time)) 
      filevardef(fout, "lev",  typeof(ta&lev),  getvardims(ta&lev))                           
      filevardef(fout, "lat",  typeof(ta&lat),  getvardims(ta&lat))                          
      filevardef(fout, "lon",  typeof(ta&lon),  getvardims(ta&lon))                          

      filevardef(fout, "nlat", "integer",  getvardims(ta&lat))                          
      filevardef(fout, "nlon", "integer",  getvardims(ta&lon))                          

      filevardef(fout, "T",      typeof(ta), getvardims(ta))
      filevardef(fout, "RELHUM", typeof(RH), getvardims(ta))
      filevardef(fout, "U",      typeof(ua), getvardims(ua))
      filevardef(fout, "V",      typeof(va), getvardims(va))
      filevardef(fout, "Z3",     typeof(zg), getvardims(zg))
      filevardef(fout, "PRES",   typeof(zg), getvardims(zg))

      filevardef(fout, "PS",      typeof(ps),        getvardims(ps))
      filevardef(fout, "PSL",     typeof(psl),       getvardims(psl))
      filevardef(fout, "LANDFRAC",typeof(land_frac), getvardims(land_frac))
      filevardef(fout, "TS",      typeof(ts),        getvardims(ts))
      filevardef(fout, "TREFHT",  typeof(tas),       getvardims(tas))

      filevardef(fout, "hyam", typeof(a), getvardims(a))
      filevardef(fout, "hybm", typeof(b), getvardims(b))

      ; Copy attributes associated with each variable to the file
      ; All attributes associated with each variable will be copied.
      ;====================================================================
      filevarattdef(fout, "time", ta&time)
      filevarattdef(fout, "lev",  ta&lev) 
      filevarattdef(fout, "lat",  ta&lat) 
      filevarattdef(fout, "lon",  ta&lon)
      
      filevarattdef(fout, "T",     ta)  
      filevarattdef(fout, "RELHUM",RH)  
      filevarattdef(fout, "U",     ua)  
      filevarattdef(fout, "V",     va)  
      filevarattdef(fout, "Z3",    zg)  
      
      filevarattdef(fout, "PS",  ps)  
      filevarattdef(fout, "PSL", psl)  
      filevarattdef(fout, "LANDFRAC", land_frac)  
      filevarattdef(fout, "TS",  ts)  
      filevarattdef(fout, "TREFHT", tas)  
    
      filevarattdef(fout, "hyam", a)  
      filevarattdef(fout, "hybm", b)  

      setfileoption(fout,"DefineMode",False)

      ; output only the data values since the dimensionality and such have
      ; been predefined. The "(/", "/)" syntax tells NCL to only output the
      ; data values to the predefined locations on the file.
      fout->time = (/ta_time(i)/)
      fout->lev  = (/ta&lev(::-1)/)
      fout->lat  = (/ta&lat/)
      fout->lon  = (/ta&lon/)
      fout->nlat = ispan(1, nlat, 1)
      fout->nlon = ispan(1, nlon, 1)
      fout->P0   = p0
      
      fout->T(0, :, :, :)     = (/ta(i,::-1,:,:)/)
      fout->RELHUM(0, :, :, :)= (/RH(i,::-1,:,:)/)
      fout->U(0, :, :, :)     = (/ua(i,::-1,:,:)/)
      fout->V(0, :, :, :)     = (/va(i,::-1,:,:)/)
      fout->Z3(0, :, :, :)    = (/zg(i,::-1,:,:)/)
      fout->PRES(0,:,:, :)    = (/pres(i,::-1,:,:)/)
      fout->PS(0, :, :)       = (/ps(i,:,:)/)
      fout->PSL(0, :, :)      = (/psl(i,:,:)/)
      fout->LANDFRAC(0, :, :) = (/land_frac(i,:,:)/)
      fout->TS(0, :, :)       = (/ts(i,:,:)/)
      fout->TREFHT(0, :, :)   = (/tas(i,:,:)/)

      fout->hyam = (/a(::-1)/)
      fout->hybm = (/b(::-1)/)

    end do
  
    ; create "CLM" file 
    ;========================
    do i = 0, num_slices-1

      ntim  = 1
      nlev  = dimsizes(tsl&depth)                                               
      nlat  = dimsizes(tsl&lat)  
      nlon  = dimsizes(tsl&lon)      
      
      filo = "CLM_" + ta_date(i) + ".nc"    ; Output file
      fout = addfile(diro + "/" + filo, "c") 
      
      if (i.eq.0) then
        myfolder = systemfunc("pwd")
        clm_filenames := (/dblquote+myfolder+"/"+diro+"/"+filo+dblquote/)
      else
        clm_filenames := array_append_record(clm_filenames, (/dblquote+myfolder+"/"+diro+"/"+filo+dblquote/), 0)
      end if
    
      ; explicitly declare file definition mode. 
      setfileoption(fout, "DefineMode", True)

      ; create global attributes of the file
      fAtt               = True         ; assign file attributes
      fAtt@title         = "DL Extreme Case: " + folder_name   
      fAtt@source        = "CESM SSP5-8.5 r11i1p1f1"
      fAtt@creation_date = systemfunc ("date")        
      fileattdef(fout, fAtt)            ; copy file attributes    
      
      ; predefine the coordinate variables and their dimensionality
      ; Note: to get an UNLIMITED record dimension, we set the dimensionality
      ; to -1 (or the actual size) and set the dimension name to True.
      dimNames = (/"time", "lat", "lon", "depth"/)  
      dimSizes = (/ ntim ,  nlat,  nlon, nlev /) 
      dimUnlim = (/ True , False, False, False/)   
      filedimdef(fout, dimNames, dimSizes, dimUnlim)

      ; predefine the the dimensionality of the variables to be written out

      ; Here we are using NCL functions to facilitate defining 
      ; each variable's dimension name(s) and type. 
      ; The following could be replaced with explicit, user defined dimension 
      ; names different from those associated with the variable in memory. 
      ; Say, PS(time,lat,lon) in the NCL script. They could be redefined for the file via: 
      ; filevardef(fout, "PS"   ,typeof(PS) ,(/"TIME","latitude","longitude"/)) 
      ;===================================================================
      filevardef(fout, "time", typeof(tsl&time), getvardims(tsl&time)) 
      filevardef(fout, "depth",typeof(tsl&depth),  getvardims(tsl&depth))                           
      filevardef(fout, "lat",  typeof(tsl&lat),  getvardims(tsl&lat))                          
      filevardef(fout, "lon",  typeof(tsl&lon),  getvardims(tsl&lon))                          
      
      filevardef(fout, "TSOI",   typeof(tsl),   getvardims(tsl))
      filevardef(fout, "H2OSOI", typeof(mrsol), getvardims(mrsol))
    
      ;===================================================================
      ; Copy attributes associated with each variable to the file
      ; All attributes associated with each variable will be copied.
      ;====================================================================
      filevarattdef(fout, "time", tsl&time)
      filevarattdef(fout, "depth",tsl&depth) 
      filevarattdef(fout, "lat",  tsl&lat) 
      filevarattdef(fout, "lon",  tsl&lon)
      
      filevarattdef(fout, "TSOI",   tsl)  
      filevarattdef(fout, "H2OSOI", mrsol)  
      
      setfileoption(fout,"DefineMode",False)

      ; output only the data values since the dimensionality and such have
      ; been predefined. The "(/", "/)" syntax tells NCL to only output the
      ; data values to the predefined locations on the file.
      fout->time = (/ta_time(i)/)
      fout->depth= (/tsl&depth/)
      fout->lat  = (/tsl&lat/)
      fout->lon  = (/tsl&lon/)
    
      fout->TSOI(0, :, :, :)   = (/tsl(i,:,:,:)/)
      fout->H2OSOI(0, :, :, :) = (/mrsol(i,:,:,:)/)

    end do

    print(cam_filenames)
    print(clm_filenames)
    print(date_str)
  
    write_table("cam2wrf.input", "w", [/cam_filenames, clm_filenames, date_str/], \
    "%s %s %s")

    system("mv cam2wrf.input " + diro)

    system("cp ../CAM2WRF/cam2wrf.exe " + diro)
    system("cp ../CAM2WRF/CAM_netcdf_to_WRF_intermediate.f90 " + diro)
    system("cp ../CAM2WRF/compile.note " + diro)

    ;==================================
    ; create WRF and WPS namelist files
    date_length = dimsizes(date_str_wrf)
    start_str = date_str_wrf(0)
    end_str   = date_str_wrf(date_length(0)-1)
    
    system("cp namelist.wps.sdd_template namelist.wps")   
    system("sed -i " + dblquote + "s/^\ start\_date.*/\ start\_date\ =\ " + start_str + ", " + start_str + ", " + \
      start_str + ", " + start_str + ",\ /g " + dblquote + " " + "namelist.wps")
    system("sed -i " + dblquote + "s/^\ end\_date.*/\ end\_date\ =\ \ \ " + end_str + ", " + end_str + ", " + \
      end_str + ", " + end_str + ",\ /g " + dblquote + " " + "namelist.wps")
    system("mv namelist.wps " + diro)

    system("cp namelist.input.sdd_template namelist.input")
    start_year_str = sprinti("%0.4i", year(0))
    start_month_str= sprinti("%0.2i", month(0))
    start_day_str  = sprinti("%0.2i", day(0))
    end_year_str = sprinti("%0.4i", year(date_length(0)-1))
    end_month_str= sprinti("%0.2i", month(date_length(0)-1))
    end_day_str  = sprinti("%0.2i", day(date_length(0)-1))
    system("sed -i " + dblquote + "s/^\ start\_year.*/\ start\_year\ =\ " + start_year_str + ", " + \
      start_year_str + ", " + start_year_str + ", " + start_year_str + ",\ /g " + dblquote + " " + "namelist.input")
    system("sed -i " + dblquote + "s/^\ start\_month.*/\ start\_month\ =\ " + start_month_str + ", " + \
      start_month_str + ", " + start_month_str + ", " + start_month_str + ",\ /g " + dblquote + " " + "namelist.input")
    system("sed -i " + dblquote + "s/^\ start\_day.*/\ start\_day\ =\ " + start_day_str + ", " + \
      start_day_str + ", " + start_day_str + ", " + start_day_str + ",\ /g " + dblquote + " " + "namelist.input")
    system("sed -i " + dblquote + "s/^\ end\_year.*/\ end\_year\ =\ " + end_year_str + ", " + \
      end_year_str + ", " + end_year_str + ", " + end_year_str + ",\ /g " + dblquote + " " + "namelist.input")
    system("sed -i " + dblquote + "s/^\ end\_month.*/\ end\_month\ =\ " + end_month_str + ", " + \
      end_month_str + ", " + end_month_str + ", " + end_month_str + ",\ /g " + dblquote + " " + "namelist.input")
    system("sed -i " + dblquote + "s/^\ end\_day.*/\ end\_day\ =\ " + end_day_str + ", " + \
      end_day_str + ", " + end_day_str + ", " + end_day_str + ",\ /g " + dblquote + " " + "namelist.input")
    system("sed -i " + dblquote + "s/^\ run\_days.*/\ run_days\ =\ " + (end_id-start_id-1) + ",\ /g " + dblquote + " " + "namelist.input")

    system("mv namelist.input " + diro)     
  end do
  
end