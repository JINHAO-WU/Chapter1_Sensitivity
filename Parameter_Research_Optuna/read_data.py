"""Data readers for observation and CMIP6 ENSO experiments."""

import xarray as xr


def read_observation_data(
    obs_sst_path,
    obs_enso_path,
    obs_ohc_path,
    lat_range,
    lon_range,
    time_range,
    include_sst=True,
    include_ohc=False,
):
    """Read observation SST/OHC fields and ENSO index for the configured domain."""
    time_start, time_end = time_range
    data = {}

    if include_sst:
        sst = xr.open_dataset(obs_sst_path)["sst"].fillna(0)
        data["sst"] = _subset_field(sst, lat_range, lon_range, time_start, time_end)

    if include_ohc:
        ohc = xr.open_dataset(obs_ohc_path)["ohc300"].fillna(0)
        data["ohc"] = _subset_field(ohc, lat_range, lon_range, time_start, time_end)

    enso = xr.open_dataset(obs_enso_path)["sst"].fillna(0)
    data["enso"] = enso.loc[time_start:time_end].squeeze()
    return data


def read_cmip6_member(member_name, cmip6_sst_dir, cmip6_enso_dir, lat_range, lon_range):
    """Read one CMIP6 member and return SST plus ENSO index."""
    sst_path = cmip6_sst_dir / f"sst_{member_name}.nc"
    enso_path = cmip6_enso_dir / f"nino34_{member_name}.nc"

    sst = xr.open_dataset(sst_path)["sst"].fillna(0)
    enso = xr.open_dataset(enso_path)["sst"].fillna(0)

    return {
        "sst": _subset_field(sst, lat_range, lon_range, None, None),
        "enso": enso.loc[:].squeeze(),
    }


def read_cmip6_dataset(member_names, cmip6_sst_dir, cmip6_enso_dir, lat_range, lon_range):
    """Read multiple CMIP6 members into a dictionary keyed by member name."""
    return {
        member_name: read_cmip6_member(member_name, cmip6_sst_dir, cmip6_enso_dir, lat_range, lon_range)
        for member_name in member_names
    }


def _subset_field(field, lat_range, lon_range, time_start, time_end):
    """Apply time and lat/lon subsets to a gridded DataArray."""
    lat_start, lat_end = lat_range
    lon_start, lon_end = lon_range
    if time_start is None or time_end is None:
        return field.loc[:, lat_start:lat_end, lon_start:lon_end]
    return field.loc[time_start:time_end, lat_start:lat_end, lon_start:lon_end]
