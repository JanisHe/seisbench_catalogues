import os
import sys
import pickle

import obspy
import yaml
import joblib
import torch
import pyocto
import pathlib
import shutil
import tqdm

import pandas as pd
import seisbench.models as sbm

from core.functions import (date_list, load_stations, daily_picks, associate_pyocto, count_picks,
                            picks_per_station, get_tmp_picks, associate_gamma, read_velocity_model)
from core.utils import nll_wrapper


def main(parfile):
    """

    :param parfile:
    :return:
    """
    with open(parfile, "r") as f:
        parameters = yaml.safe_load(f)

    # Copy parfile and json_file to results
    dirname = os.path.join("..", "results", pathlib.Path(parameters["filename"]).stem)
    if os.path.isdir(dirname) is False:
        os.makedirs(dirname)
    shutil.copyfile(src=parfile,
                    dst=os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.yml'))
    shutil.copyfile(src=parameters["data"]["stations"],
                    dst=os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.json'))

    # Load PhaseNet model
    if os.path.isfile(parameters["picking"].get("phasenet_model")):
        pn_model = torch.load(parameters["picking"].pop("phasenet_model"),
                              map_location=torch.device('cpu'))
    else:
        pn_model = sbm.PhaseNet.from_pretrained(parameters["picking"].pop("phasenet_model"))

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
    #######################################################################

    # Start phase picking
    joblib_pool = joblib.Parallel(n_jobs=parameters.get("nworkers"), backend="threading")
    with tqdm.tqdm(total=len(stations["id"]) * len(dates)) as pbar:
        for station in stations["id"]:
            pbar.set_postfix_str(f"Picking phases at station {station}")   # Update progressbar

            # Start parallel picking over days for each station
            joblib_pool(
                joblib.delayed(daily_picks)(
                    julday=date[1],
                    year=date[0],
                    starttime=obspy.UTCDateTime(parameters["data"]["starttime"]),
                    endtime=obspy.UTCDateTime(parameters["data"]["endtime"]),
                    sds_path=parameters["data"]["sds_path"],
                    network=station.split(".")[0],
                    station=station.split(".")[1],
                    channel_code="*",
                    seisbench_model=pn_model,
                    output_format="pyocto",
                    sampling_rate=sampling_rate,
                    pathname=tmp_pick_dirname, **parameters["picking"]
                )
                for date in dates)
            pbar.update(len(dates))

    # Collect all picks from temporary saved picks
    picks = get_tmp_picks(dirname=tmp_pick_dirname)

    # Convert picks of each station to single dataframe
    picks_station = picks_per_station(seisbench_picks=picks)
    #######################################################################

    # Association
    # TODO: If nonlinloc in parameters, load velocity model and define 1D velocity model for PyOcto and
    #       use GaMMA1D.
    if parameters["association"].get("method").lower() == "pyocto":
        parameters["association"].pop("method")
        try:
            if parameters["nonlinloc"].get("velocity_model"):
                # Create velocity model for PyOcto
                vel_model = read_velocity_model(filename=parameters["nonlinloc"]["velocity_model"])
                vel_model_path = os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.pyocto')
                # TODO: Define xdist and zdist from lat lon in parameters
                pyocto.VelocityModel1D.create_model(model=vel_model,
                                                    delta=1.,
                                                    xdist=30,
                                                    zdist=20,
                                                    path=vel_model_path)
                velocity_model = pyocto.VelocityModel1D(path=vel_model_path,
                                                        tolerance=2.0)
        except KeyError:
            velocity_model = pyocto.VelocityModel0D(**parameters["0D_velocity_model"])

        # Generate catalogue
        catalog = associate_pyocto(
            station_json=stations,
            picks=picks,
            velocity_model=velocity_model,
            **parameters["association"])
    elif parameters["association"].get("method").lower() == "gamma":
        parameters["association"].pop("method")

        # Define velocity model for eikonal solver in GaMMA
        try:
            if parameters["nonlinloc"].get("velocity_model"):
                # Read velocity model
                vel_model = read_velocity_model(filename=parameters["nonlinloc"]["velocity_model"])
                eikonal = {"vel": {"p": vel_model["vp"].to_list(),
                                   "s": vel_model["vs"].to_list(),
                                   "z": vel_model["depth"].to_list()},
                           "h": 1.0}

        except KeyError:
            eikonal = None

        # Start association
        catalog = associate_gamma(
            picks=picks,
            stations=stations,
            ncpu=parameters["nworkers"],
            p_vel=parameters.get("0D_velocity_model").get("p_velocity"),
            s_vel=parameters.get("0D_velocity_model").get("s_velocity"),
            eikonal=eikonal,
            **parameters["association"]
        )
    else:
        msg = f"Method {parameters['association']['method']} is not implemented."
        raise ValueError(msg)

    print(f"\nDetected {len(catalog)} events after association.\n")
    #######################################################################

    # Relocate earthquakes in catalog with NonLinLoc
    # TODO: NLL package can used pre calculated travel times. Try to find pre calcualted travel times, instead
    #       of computing new. Especially worthful, when a accurate velocoty model is used
    if parameters.get("nonlinloc") and len(catalog) > 0:
        catalog = nll_wrapper(catalog=catalog,
                              station_json=stations,
                              nll_basepath=parameters["nonlinloc"]["nll_basepath"],
                              vel_model=parameters["nonlinloc"]["velocity_model"])

    # TODO: Add filtering methods (see Notizbuch fuer Paper)
    # Assign associated events to final catalog and create output files
    # Save picks as pickle and catalog as xmlin separate directory
    catalog_filename = os.path.join(dirname, parameters["filename"])
    catalog.write(filename=catalog_filename, format="QUAKEML")
    print(f"Wrote {len(catalog)} events to final catalogue.")
    with open(os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.picks'), "wb") as handle:
        pickle.dump(obj=picks, file=handle)

    # Save picks for each station
    # First, delete existing files
    if os.path.isdir(os.path.join(dirname, "picks")) is True:
        shutil.rmtree(os.path.join(dirname, "picks"))
    if os.path.isdir(os.path.join(dirname, "picks")) is False:
        os.makedirs(os.path.join(dirname, "picks"))

    # Note, picks for each station are saved in csv-format, since functions for plotting are written to read these
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
