import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    'product_type': ['reanalysis'],
    'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', 'evaporation', 'runoff', 'convective_precipitation', 'convective_rain_rate', 'total_precipitation', 'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3', 'soil_temperature_level_4', 'soil_type', 'high_vegetation_cover', 'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 'low_vegetation_cover', 'type_of_high_vegetation', 'type_of_low_vegetation', '2m_dewpoint_temperature', '2m_temperature', 'surface_pressure', 'skin_temperature', 'clear_sky_direct_solar_radiation_at_surface', 'downward_uv_radiation_at_the_surface', 'forecast_logarithm_of_surface_roughness_for_heat', 'surface_net_solar_radiation', 'instantaneous_surface_sensible_heat_flux', 'surface_latent_heat_flux', 'near_ir_albedo_for_direct_radiation', 'near_ir_albedo_for_diffuse_radiation', 'top_net_thermal_radiation', 'cloud_base_height', 'high_cloud_cover', 'low_cloud_cover', 'medium_cloud_cover', 'total_cloud_cover', 'total_column_cloud_liquid_water'],
    'year': ['2023'],
    'month': ['06', '07', '08'],
    'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
    'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
    'data_format': 'grib',
    'download_format': 'unarchived',
    'area': [45, 19, 34, 28]
}

times = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']

for time in times:
    request = {
        'product_type': ['reanalysis'],
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', 'evaporation', 'runoff',
                     'convective_precipitation', 'convective_rain_rate', 'total_precipitation',
                     'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3',
                     'soil_temperature_level_4', 'soil_type', 'high_vegetation_cover',
                     'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 'low_vegetation_cover',
                     'type_of_high_vegetation', 'type_of_low_vegetation', '2m_dewpoint_temperature', '2m_temperature',
                     'surface_pressure', 'skin_temperature', 'clear_sky_direct_solar_radiation_at_surface',
                     'downward_uv_radiation_at_the_surface', 'forecast_logarithm_of_surface_roughness_for_heat',
                     'surface_net_solar_radiation', 'instantaneous_surface_sensible_heat_flux',
                     'surface_latent_heat_flux', 'near_ir_albedo_for_direct_radiation',
                     'near_ir_albedo_for_diffuse_radiation', 'top_net_thermal_radiation', 'cloud_base_height',
                     'high_cloud_cover', 'low_cloud_cover', 'medium_cloud_cover', 'total_cloud_cover',
                     'total_column_cloud_liquid_water'],
        'year': ['2023'],
        'month': ['06', '07', '08'],
        'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
        'time': [time],
        'data_format': 'grib',
        'download_format': 'unarchived',
        'area': [45, 19, 34, 28]
    }
    client = cdsapi.Client()
    r = client.retrieve(dataset, request).download()
    print(time, r)


