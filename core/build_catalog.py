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

from functions import (date_list, load_stations, daily_picks, associate_pyocto, count_picks, picks2df,
                       picks_per_station, get_tmp_picks)


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

    joblib_pool = joblib.Parallel(n_jobs=parameters.get("nworkers"))
    for station in stations["id"]:
        print("Picking phases at station", station)
        picks_generator = joblib_pool(joblib.delayed(daily_picks)(
            julday=date[1], year=date[0], starttime=obspy.UTCDateTime(parameters["data"]["starttime"]),
            endtime=obspy.UTCDateTime(parameters["data"]["endtime"]), sds_path=parameters["data"]["sds_path"],
            network=station.split(".")[0], station=station.split(".")[1], channel_code="*",
            seisbench_model=pn_model, output_format="pyocto", sampling_rate=sampling_rate,
            pathname=tmp_pick_dirname, **parameters["picking"]
        )
                                      for date in tqdm.tqdm(dates))

        # for index in range(len(picks_generator)):
        #     try:
        #         picks += picks_generator[index].picks
        #     except AttributeError:
        #         print("Did not find picks at station", station)

    # Collect all picks from temporary saved picks and delete temporary picks
    picks = get_tmp_picks(dirname=tmp_pick_dirname)
    shutil.rmtree(tmp_pick_dirname)

    # Convert picks to pandas dataframe
    # picks_df = picks2df(seisbench_picks=picks)

    # Convert picks of each station to single dataframe
    picks_station = picks_per_station(seisbench_picks=picks)

    # velocity_model = pyocto.VelocityModel0D(
    #     p_velocity=parameters["p_velocity"],
    #     s_velocity=parameters["s_velocity"],
    #     tolerance=parameters["tolerance"],
    #     association_cutoff_distance=parameters["cutoff_distance"],
    # )
    velocity_model = pyocto.VelocityModel0D(**parameters["velocity_model"])

    # Generate catalogue
    catalog = associate_pyocto(
        station_json=stations,
        picks=picks,
        velocity_model=velocity_model,
        **parameters["association"])
    print(f"Detected {len(catalog)} events. ")

    # Save picks as pickle and catalog as xmlin separate directory
    catalog_filename = os.path.join(dirname, parameters["filename"])
    catalog.write(filename=catalog_filename, format="QUAKEML")
    with open(os.path.join(dirname, f'{pathlib.Path(parameters["filename"]).stem}.picks'), "wb") as handle:
        pickle.dump(obj=picks, file=handle)

    # Save picks for each station
    # First, delete existing files
    if os.path.isdir(os.path.join(dirname, "picks")) is True:
        shutil.rmtree(os.path.join(dirname, "picks"))
    if os.path.isdir(os.path.join(dirname, "picks")) is False:
        os.makedirs(os.path.join(dirname, "picks"))
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
