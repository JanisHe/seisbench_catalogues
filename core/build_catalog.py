import os
import glob
import sys
import pickle

import obspy
import pandas as pd
import yaml
import joblib
import torch
import pyocto
import pathlib
import shutil
import tqdm
import seisbench  # noqa

from typing import Union
import seisbench.models as sbm  # noqa
from obspy.geodetics import degrees2kilometers

from core.functions import (date_list, load_stations, daily_picks, associate_pyocto, count_picks,
                            picks_per_station, get_tmp_picks, associate_gamma, read_velocity_model, area_limits)
from core.utils import nll_wrapper, merge_catalogs

# seisbench.use_backup_repository()


def daily_catalog(julday: int,
                  year: int,
                  stations: pd.DataFrame,
                  channel_code: str,
                  seisbench_model,
                  association_method: str = "pyocto",
                  sampling_rate: (None, float) = None,
                  pathname: str = "tmp_picks",
                  **parameters
                  ):

    # Loop over each station_picks
    for station in stations["id"]:
        print(station)
        # Picking
        daily_picks(
            julday=julday,
            year=year,
            starttime=obspy.UTCDateTime(parameters["data"]["starttime"]),
            endtime=obspy.UTCDateTime(parameters["data"]["endtime"]),
            sds_path=parameters["data"]["sds_path"],
            network=station.split(".")[0],
            station=station.split(".")[1],
            channel_code=channel_code,
            seisbench_model=seisbench_model,
            sampling_rate=sampling_rate,
            pathname=pathname,
            **parameters["picking"]
        )

    # Collect picks for single day
    picks = get_tmp_picks(dirname=pathname,
                          julday=julday)

    # Association stuff
    if association_method.lower() == "pyocto":
        catalog = associate_pyocto(
            station_json=stations,
            picks=picks,
            **parameters["association"])
    elif association_method.lower() == "gamma":
        catalog = associate_gamma(
            picks=picks,
            stations=stations,
            ncpu=parameters["nworkers"],
            p_vel=None,
            s_vel=None,
            **parameters["association"]
        )

    # Save catalog in tmp_dir
    if len(catalog) > 0:
        catalog.write(filename=os.path.join(pathname, f"{year}_{julday}.xml"),
                      format="QUAKEML")


def load_single_model(phasenet_model: str):
    if os.path.isfile(phasenet_model):
        pn_model = torch.load(phasenet_model,
                              map_location=torch.device('cpu'),
                              weights_only=False)
    else:
        pn_model = sbm.PhaseNet.from_pretrained(phasenet_model,
                                                update=True)

    return pn_model


def load_models(phasenet_model: Union[str, dict]):
    if isinstance(phasenet_model, str):
        pn_model = load_single_model(phasenet_model=phasenet_model)
    elif isinstance(phasenet_model, dict):
        pn_model = []
        for value in phasenet_model.values():
            pn_model.append(load_single_model(
                phasenet_model=value
            ))

    return pn_model


def main(parfile):
    """

    :param parfile:
    :return:
    """
    with open(parfile, "r") as f:
        parameters = yaml.safe_load(f)

    # Copy parfile and json_file to results
    # TODO: Instead of taking ".." take the whole path of the project with pathlib
    dirname = os.path.join("..", "results", pathlib.Path(parameters["filename"]).stem)
    if os.path.isdir(dirname) is False:
        os.makedirs(dirname)
    try:
        shutil.copyfile(src=parfile,
                        dst=os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.yml'))
        shutil.copyfile(src=parameters["data"]["stations"],
                        dst=os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.json'))
    except shutil.SameFileError as e:
        print(e)
        print("Keeping old file and do not overwrite")

    # Copy PhaseNet model and load PhaseNet model(s)
    # If parameter phasenet_model is of type str, then one model is loaded, otherwise if phasenet_model is of type
    # dict all models in dict are loaded and semblance picking is applied
    # shutil.copyfile(src=parameters["picking"]["phasenet_model"],
    #                 dst=os.path.join(dirname, pathlib.Path(parameters["picking"]["phasenet_model"]).name))
    pn_model = load_models(phasenet_model=parameters["picking"].pop("phasenet_model"))
    # pn_model.filter_kwargs["type"] = "bandpass"

    # Load stations
    stations = load_stations(station_json=parameters["data"]["stations"])

    # Loop with joblib over all stations, get waveforms for each day and pick
    dates = date_list(start_date=obspy.UTCDateTime(parameters["data"]["starttime"]),
                      end_date=obspy.UTCDateTime(parameters["data"]["endtime"]))

    # Check sampling rate in parameters
    if parameters["data"].get("sampling_rate"):
        sampling_rate = parameters["data"]["sampling_rate"]
    else:
        sampling_rate = None

    # Create directory for temporary picks
    tmp_pick_dirname = os.path.join(dirname, "tmp_picks")
    if os.path.isdir(tmp_pick_dirname):
        shutil.rmtree(tmp_pick_dirname)
    os.makedirs(tmp_pick_dirname)

    # Set up velocity model
    association_method = parameters["association"].pop("method")
    if parameters["nonlinloc"].get("velocity_model"):
        vel_model = read_velocity_model(filename=parameters["nonlinloc"]["velocity_model"])
        # Create velocity model either for GaMMA or PyOcto
        if association_method.lower() == "pyocto":
            vel_model_path = os.path.join(dirname,
                                          f'{pathlib.Path(parameters["filename"]).stem}.pyocto')
            # TODO: Define xdist and zdist from lat lon in parameters
            #       Decrease delta for a more accurate initial localisation
            limits = area_limits(stations=stations)
            max_degree = max([abs(limits["latitude"][1] - limits["latitude"][0]),
                              abs(limits["longitude"][1] - limits["longitude"][0])])
            pyocto.VelocityModel1D.create_model(model=vel_model,
                                                delta=1,
                                                xdist=degrees2kilometers(degrees=max_degree),
                                                zdist=max(parameters["association"]["zlim"]),
                                                path=vel_model_path)
            velocity_model = pyocto.VelocityModel1D(path=vel_model_path,
                                                    tolerance=2.0)
            parameters["association"]["velocity_model"] = velocity_model
        elif association_method.lower() == "gamma":
            eikonal = {"vel": {"p": vel_model["vp"].to_list(),
                               "s": vel_model["vs"].to_list(),
                               "z": vel_model["depth"].to_list()},
                       "h": 1.0}
            parameters["association"]["eikonal"] = eikonal
        else:
            msg = f"Method {parameters['association']['method']} is not implemented."
            raise ValueError(msg)


    #######################################################################

    # TODO: Since processes can be doubled be careful with number of workers for association
    # Define joblib pool for multiprocessing
    joblib_pool = joblib.Parallel(n_jobs=parameters.get("nworkers"))

    # Loop over each date in dates with joblib
    joblib_pool(
        joblib.delayed(daily_catalog)
        (
            julday=date[1],
            year=date[0],
            stations=stations,
            channel_code="*",
            dirname=dirname,
            sampling_rate=sampling_rate,
            pathname=tmp_pick_dirname,
            seisbench_model=pn_model,
            association_method=association_method,
            **parameters
        )
        for date in dates
    )

    # Merge daily catalogs
    catalogs = glob.glob(os.path.join(tmp_pick_dirname, "*.xml"))
    catalog = merge_catalogs(catalogs=catalogs)
    print(f"\nDetected {len(catalog)} events after association.\n")

    # Run NLL
    # Relocate earthquakes in catalog with NonLinLoc
    # TODO: NLL package can used pre calculated travel times. Try to find pre calcualted travel times, instead
    #       of computing new. Especially worthful, when a accurate velocoty model is used
    # if parameters.get("nonlinloc") and len(catalog) > 0:
    #     catalog = nll_wrapper(catalog=catalog,
    #                           station_json=stations,
    #                           nll_basepath=parameters["nonlinloc"]["nll_basepath"],
    #                           nll_executable=parameters["nonlinloc"]["nll_executable"],
    #                           vel_model=parameters["nonlinloc"]["velocity_model"])


    # Collect all picks from temporary saved picks and convert picks of each station_picks to single dataframe
    picks = get_tmp_picks(dirname=tmp_pick_dirname)
    picks_station = picks_per_station(seisbench_picks=picks)

    # Assign associated events to final catalog and create output files
    # Save picks as pickle and catalog as xmlin separate directory
    catalog_filename = os.path.join(dirname, parameters["filename"])
    catalog.write(filename=catalog_filename, format="QUAKEML")
    print(f"Wrote {len(catalog)} events to final catalogue.")
    with open(os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.picks'), "wb") as handle:
        pickle.dump(obj=picks, file=handle)

    # Save picks for each station_picks
    # First, delete existing files
    if os.path.isdir(os.path.join(dirname, "picks")) is True:
        shutil.rmtree(os.path.join(dirname, "picks"))
    if os.path.isdir(os.path.join(dirname, "picks")) is False:
        os.makedirs(os.path.join(dirname, "picks"))

    # Note, picks for each station_picks are saved in csv-format, since functions for plotting are written to read these
    # files. For a later association, all picks are saved as a pickle file.
    for trace_id, station_picks in picks_station.items():
        station_pick_fname = os.path.join(dirname, "picks", f"{trace_id}.csv")
        station_picks.to_csv(path_or_buf=station_pick_fname, mode='w')

    # Write summary file
    with open(os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.summary'), "w") as f:
        phase_count = count_picks(picks=picks)
        f.write(f"Number of events in catalog: {len(catalog)}\n")
        for id in phase_count.keys():
            string = f"{id}: "
            for phase in phase_count[id]:
                string += f"{phase}: {phase_count[id][phase]} "
            f.write(string + "\n")

    # Delete temporary pick files
    shutil.rmtree(tmp_pick_dirname)



if __name__ == "__main__":
    if len(sys.argv) <= 1:
        parfile = "../parfiles/parfile.yml"
    elif len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) is False:
        msg = "The given file {} does not exist. Perhaps take the full path of the file.".format(sys.argv[1])
        raise FileNotFoundError(msg)
    else:
        parfile = sys.argv[1]

    # Start to run catalogue building
    main(parfile=parfile)
