; Dump prepare_basemaps /sfq intermediates for Python parity.
pro dump_idl_sfq
  compile_opt idl2
  outdir = '/tmp/sfq_parity_20240512'
  file_field = '/Users/gelu/jsoc_cache/2024-05-11/hmi.B_720s.20240512_000000_TAI.field.fits'
  file_inclination = '/Users/gelu/jsoc_cache/2024-05-11/hmi.B_720s.20240512_000000_TAI.inclination.fits'
  file_azimuth = '/Users/gelu/jsoc_cache/2024-05-11/hmi.B_720s.20240512_000000_TAI.azimuth.fits'
  file_disambig = '/Users/gelu/jsoc_cache/2024-05-11/hmi.B_720s.20240512_000000_TAI.disambig.fits'
  file_continuum = '/Users/gelu/jsoc_cache/2024-05-11/hmi.Ic_noLimbDark_720s.20240512_000000_TAI.continuum.fits'
  center_arcsec = [0d, 0d]
  size_pix = [64l, 64l, 64l]
  dx_km = 1400d

  files = [file_field, file_inclination, file_azimuth]
  read_sdo, files, index, data, /uncomp_delete
  ind = where(finite(data, /nan), count)
  if count gt 0 then data[ind] = 0
  wcs0 = fitshead2wcs(index[0])
  wcs2map, data[*,*,0], wcs0, map
  map2wcs, map, wcs0

  dsun_obs = wcs0.position.dsun_obs
  dx_deg = dx_km*1d3 / wcs_rsun() * 180d/!dpi
  wcs_convert_from_coord, wcs0, center_arcsec, 'HG', lon, lat, /carrington
  wcs = wcs_2d_simulate(size_pix[0], size_pix[1], cdelt=dx_deg, crval=[lon,lat], $
                        type='CR', projection='cea', date_obs=index[0].date_obs)

  nx = wcs.naxis[0]
  ny = wcs.naxis[1]
  pix = lonarr(2, 4)
  pix[0,*] = [0, 0, nx-1, nx-1]
  pix[1,*] = [0, ny-1, ny-1, 0]
  crd = wcs_get_coord(wcs, pix)
  wcs_convert_from_coord, wcs, crd, 'HG', lonc, latc, /carrington
  wcs_convert_to_coord, wcs0, crd_ref, 'HG', lonc, latc, /carrington
  pix_ref = wcs_get_pixel(wcs0, crd_ref)
  xrange = round(minmax(pix_ref[0,*])) + [-1,1]
  yrange = round(minmax(pix_ref[1,*])) + [-1,1]

  field_s = data[xrange[0]:xrange[1], yrange[0]:yrange[1], 0]
  inclination_s = data[xrange[0]:xrange[1], yrange[0]:yrange[1], 1]
  azimuth_s = data[xrange[0]:xrange[1], yrange[0]:yrange[1], 2]
  az_before = azimuth_s

  bz = field_s*cos(inclination_s*(!dpi/180d))
  bx = field_s*sin(inclination_s*(!dpi/180d))*sin(azimuth_s*!dpi/180d)
  by = -field_s*sin(inclination_s*(!dpi/180d))*cos(azimuth_s*!dpi/180d)
  bx = rotate(bx, 2)
  by = rotate(by, 2)
  bz = rotate(bz, 2)
  pos = [min(crd_ref[0,*]), min(crd_ref[1,*]), max(crd_ref[0,*]), max(crd_ref[1,*])]
  rsun_arcsec = wcs_rsun()/wcs0.position.dsun_obs*180d*60d*60d/!dpi

  bx0 = bx & by0 = by & bz0 = bz
  sfq_disambig, bx, by, bz, pos, rsun_arcsec, /hmi, /silent
  by = -rotate(by, 2)
  bx = rotate(bx, 2)
  az = atan(bx, by)
  az_after = az*180d/!dpi

  save, filename=outdir+'/idl_sfq_dump.sav', $
        xrange, yrange, pos, rsun_arcsec, lon, lat, dx_deg, $
        pix_ref, crd_ref, az_before, az_after, field_s, inclination_s, $
        bx0, by0, bz0, bx, by, dsun_obs, nx, ny
  print, 'WROTE ', outdir+'/idl_sfq_dump.sav'
  print, 'xrange=', xrange, ' yrange=', yrange
  print, 'pos=', pos, ' rsun=', rsun_arcsec
  print, 'lon,lat=', lon, lat
  print, 'crop shape=', size(az_after, /dim)
  exit
end
