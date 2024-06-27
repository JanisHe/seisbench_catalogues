import os
import glob
import json
import pathlib
import tqdm
import pickle
import datetime
import obspy
import pyocto

import pandas as pd
import numpy as np
from pyproj import Proj

from obspy.core.event.origin import Pick, Origin
from obspy.core.event.event import Event
from obspy.core.event.base import WaveformStreamID, Comment
from gamma.utils import association


def date_list(start_date, end_date):
    """
    Returns a full date list, containing all dates between start- and
    enddate. If startdate == enddate, the returned list has length 1
    and contains just one date. Takes datetime.datetime object as input
    values
    """

    if not isinstance(start_date, datetime.datetime):
        start_date = obspy.UTCDateTime(start_date).datetime

    if not isinstance(end_date, datetime.datetime):
        end_date = obspy.UTCDateTime(end_date).datetime

    date_list = []
    date = start_date
    while date <= end_date:
        date = datetime.datetime(date.year, date.month, date.day)
        date_obspy = obspy.UTCDateTime(date)
        date_list.append((date_obspy.year, date_obspy.julday))
        date = date + datetime.timedelta(days=1)

    return date_list


def get_waveforms(sds_path: str, station: str, network: str, channel_code: str,
                  starttime: obspy.UTCDateTime, endtime: obspy.UTCDateTime, location: str = "*"):
    dates = date_list(start_date=starttime.datetime, end_date=endtime.datetime)
    sds_pathname = "{sds_path}/{year}/{network}/{station}/{channel}*/*{julday}"

    stream = obspy.Stream()
    for i, d in enumerate(dates):
        jd = "{:03d}".format(d[1])
        pathname = sds_pathname.format(sds_path=sds_path, year=d[0],
                                       network=network, station=station,
                                       channel=channel_code, julday=jd)
        try:
            st = obspy.read(pathname)
            st.merge(fill_value=0)
        except Exception:
            st = obspy.Stream()

        # Trim stream at start- and enddate
        if st:
            if i == 0:
                st.trim(starttime=starttime)
            if i == len(dates) - 1:
                st.trim(endtime=endtime)

        stream += st

    stream.merge()

    return stream


def get_daily_waveforms(julday: int, year: int, starttime: obspy.UTCDateTime,
                        endtime: obspy.UTCDateTime, sds_path: str,
                        station: str, network: str, channel_code: str,
                        sampling_rate: (None, float) = None):
    """
    
    :param julday: 
    :param year: 
    :param starttime: 
    :param endtime: 
    :param sds_path: 
    :param station: 
    :param network: 
    :param channel_code:
    :param sampling_rate
    :return: 
    """
    sds_pathname = "{sds_path}/{year}/{network}/{station}/{channel}*/*{julday}"
    jd = "{:03d}".format(julday)
    pathname = sds_pathname.format(sds_path=sds_path, year=year,
                                   network=network, station=station,
                                   channel=channel_code, julday=jd)
    try:
        stream = obspy.read(pathname)
        stream.merge(fill_value=0)
    except Exception:
        stream = obspy.Stream()

    # Trim stream
    if julday == starttime.julday:
        stream.trim(starttime=starttime)
    if julday == endtime.julday:
        stream.trim(endtime=endtime)

    # Resample stream
    if sampling_rate:
        stream.resample(sampling_rate=sampling_rate)

    return stream


def picking(seisbench_model, stream: obspy.Stream, batch_size: int = 512, P_threshold: float = 0.1,
            S_threshold: float = 0.1, output_format: str = "pyocto", **kwargs):

    if output_format.lower() not in ["gamma", "pyocto"]:
        msg = "Output_format must be either pyocto or gamma."
        raise ValueError(msg)

    # if len(stream) != 3:
    #     msg = "Can only pick data from one station with 3 channels."
    #     raise ValueError(msg)

    picks = seisbench_model.classify(stream,
                                     batch_size=batch_size,
                                     P_threshold=P_threshold,
                                     S_threshold=S_threshold,
                                     **kwargs)

    if output_format.lower() == "gamma":
        picks_df = []
        for pick in picks:
            picks_df.append(
                {"id": f"{stream[0].stats.network}.{stream[0].stats.station}",
                 "timestamp": pick.peak_time.datetime,
                 "prob": pick.peak_value,
                 "type": pick.phase.lower()}
            )

        return picks_df
    else:
        return picks


def daily_picks(julday: int, year: int, starttime: obspy.UTCDateTime,
                endtime: obspy.UTCDateTime, sds_path: str,
                station: str, network: str, channel_code: str,
                seisbench_model, batch_size: int = 512, P_threshold: float = 0.1,
                S_threshold: float = 0.1, output_format: str = "pyocto",
                sampling_rate: (None, float) = None,
                pathname: str = "tmp_picks", **kwargs):
    """

    :param julday:
    :param year:
    :param starttime:
    :param endtime:
    :param sds_path:
    :param station:
    :param network:
    :param channel_code:
    :param seisbench_model:
    :param batch_size:
    :param P_threshold:
    :param S_threshold:
    :param output_format:
    :param pathname:
    :param kwargs:
    :return:
    """
    stream = get_daily_waveforms(julday=julday, year=year, starttime=starttime,
                                 endtime=endtime, sds_path=sds_path, station=station,
                                 network=network, channel_code=channel_code,
                                 sampling_rate=sampling_rate)

    # Pick with seisbench model
    if len(stream) > 0:
        picks = picking(seisbench_model=seisbench_model, batch_size=batch_size,
                        P_threshold=P_threshold, S_threshold=S_threshold,
                        output_format=output_format, stream=stream, **kwargs)

        # Save picks as pickle
        filename = os.path.join(pathname, f"{network}_{station}_{year}_{julday}.pick")
        with open(filename, "wb") as handle:
            pickle.dump(picks, handle)


def convert_station_json(stations: dict) -> pd.DataFrame:
    """

    :param stations:
    :return:
    """
    station_df = []
    for station in stations.keys():
        station_df.append(
            {"id": station,
             "latitude": stations[station]["coords"][0],
             "longitude": stations[station]["coords"][1],
             "elevation": stations[station]["coords"][2]}
        )

    return pd.DataFrame(station_df)


def load_stations(station_json: str):
    """

    :param station_json:
    :return:
    """
    with open(station_json) as f_json:
        stations = json.load(f_json)
        stations = convert_station_json(stations)

    return stations


def area_limits(stations: (pd.DataFrame, dict)) -> dict:
    """

    :param stations:
    :return:
    """
    # Convert stations to pandas dataframe
    if isinstance(stations, dict):
        stations = convert_station_json(stations=stations)

    limits = {
        "latitude": (np.min(stations["latitude"] - 0.2), np.max(stations["latitude"]) + 0.2),
        "longitude": (np.min(stations["longitude"] - 0.2), np.max(stations["longitude"]) + 0.2)
    }

    # Add center to limits
    limits["center"] = ((limits["latitude"][1] - limits["latitude"][0]) / 2 + limits["latitude"][0],
                        (limits["longitude"][1] - limits["longitude"][0]) / 2 + limits["longitude"][0])

    return limits


def sort_events(events: list):
    # Create list with all dates
    dates = [event.origins[0].time.datetime for event in events]

    # Sort event_list by dates
    sorted_events = [x for _, x in zip(sorted(dates), events)]

    return sorted_events


def count_picks(picks: list) -> dict:
    """

    :param picks:
    :return:
    """
    counts = {"TOTAL": {}}
    for pick in picks:
        phase = pick.phase
        if pick.trace_id not in counts.keys():
            # Allocate empty dict for each station
            counts.update({pick.trace_id: {}})
        # Update phase
        if pick.phase not in counts[pick.trace_id].keys():
            counts[pick.trace_id].update({pick.phase: 0})
        if pick.phase not in counts["TOTAL"].keys():
            counts["TOTAL"].update({pick.phase: 0})

        # Update phase counter
        counts[pick.trace_id][phase] += 1
        counts["TOTAL"][phase] += 1

    return counts


def associate_pyocto(station_json: (str, pd.DataFrame), picks, velocity_model, **kwargs):
    """

    :param station_json:
    :param velocity_model:
    :param time_before:
    :param n_picks:
    :param n_p_and_s_picks:
    :param zlim:
    :return:
    """

    # Load station_json
    if isinstance(station_json, str):
        stations = load_stations(station_json=station_json)
    elif isinstance(station_json, pd.DataFrame):
        stations = station_json

    # Define limits of area from stations
    limits = area_limits(stations=stations)

    # Set up association
    associator = pyocto.OctoAssociator.from_area(
        lat=limits["latitude"],
        lon=limits["longitude"],
        velocity_model=velocity_model,
        **kwargs
    )

    # Convert stations to required format
    stations = associator.transform_stations(stations=stations)

    # Start association
    events, assignments = associator.associate_seisbench(picks, stations)
    associator.transform_events(events)

    # Proof whether events have been detected
    if len(events) == 0:
        return obspy.Catalog()

    # Assign correct time to event
    events["time"] = events["time"].apply(datetime.datetime.fromtimestamp, tz=datetime.timezone.utc)
    assignments["time"] = assignments["time"].apply(datetime.datetime.fromtimestamp, tz=datetime.timezone.utc)

    # Create picks for catalog
    picks_dct = {}
    for idx in range(len(assignments["event_idx"])):
        waveform_id = WaveformStreamID(network_code=assignments["station"][idx].split(".")[0],
                                       station_code=assignments["station"][idx].split(".")[1],
                                       location_code=assignments["station"][idx].split(".")[2])
        pick = Pick(time=obspy.UTCDateTime(assignments["time"][idx]),
                    waveform_id=waveform_id,
                    phase_hint=assignments["phase"][idx])

        if assignments["event_idx"][idx] in picks_dct.keys():
            picks_dct[assignments["event_idx"][idx]].append(pick)
        else:
            picks_dct.update({assignments["event_idx"][idx]: [pick]})

    # TODO: Filter picks by logic operation, e.g. spick-time - p-picktime > 0

    # Create obspy catalog
    event_lst = []
    for index in events["idx"]:
        origin = Origin(time=obspy.UTCDateTime(events["time"][index]), longitude=events["longitude"][index],
                        latitude=events["latitude"][index], depth=events["depth"][index] * 1e3)
        ev = Event(picks=picks_dct[index], force_resource_id=True, origins=[origin])
        event_lst.append(ev)

    # Sort events by date
    event_lst = sort_events(events=event_lst)

    return obspy.Catalog(event_lst)


def associate_gamma(picks, stations, ncpu=4, use_dbscan=True, use_amplitude=False,
                    zlim=(0, 30), p_vel=5.9, s_vel=3.5, method="BGMM",
                    inital_points=[1, 1, 1], covariance_prior=(5, 5),
                    oversample_factor=10, dbscan_eps=4, dbscan_min_samples=5,
                    min_picks_per_eq=8, min_p_picks_per_eq=4,
                    min_s_picks_per_eq=4, max_sigma11=2.0, max_sigma22=1.0,
                    max_sigma12=1.0):
    """

    :param picks:
    :param stations:
    :param ncpu:
    :param use_dbscan:
    :param use_amplitude:
    :param zlim:
    :param p_vel:
    :param s_vel:
    :param method:
    :param inital_points:
    :param covariance_prior:
    :param oversample_factor:
    :param dbscan_eps:
    :param dbscan_min_samples:
    :param min_picks_per_eq:
    :param min_p_picks_per_eq:
    :param min_s_picks_per_eq:
    :param max_sigma11:
    :param max_sigma22:
    :param max_sigma12:
    :return:
    """
    # Load station_json
    if isinstance(stations, str):
        stations = load_stations(station_json=stations)

    # Convert picks to required format
    picks = picks_per_station(seisbench_picks=picks,
                              association_method="gamma")

    # Get grid from latitude and longitude
    config = area_limits(stations=stations)

    # Create projection from pyproj. Use a string that is then transfered to Proj (https://proj.org)
    proj = Proj(f"+proj=sterea +lat_0={config['center'][0]} +lon_0={config['center'][1]} +units=km")

    # Compute boundaries for x and y
    ylim1, xlim1 = proj(latitude=config['latitude'][0], longitude=config['longitude'][0])
    ylim2, xlim2 = proj(latitude=config['latitude'][1], longitude=config['longitude'][1])
    config['x(km)'] = [xlim1, xlim2]
    config["y(km)"] = [ylim1, ylim2]

    # Read gamma_score from kwargs
    # gamma_score_threshold = kwargs.get("gamma_score")
    # if not gamma_score_threshold:
    #     gamma_score_threshold = -99

    # Return empty event list, when no picks are found
    if len(picks) == 0:
        return []

    # Make config dict for GaMMA
    # Config for earthquake location
    config["ncpu"] = ncpu
    config["dims"] = ['x(km)', 'y(km)', 'z(km)']
    config["use_dbscan"] = use_dbscan
    config["use_amplitude"] = use_amplitude
    config["z(km)"] = zlim
    config["vel"] = {"p": p_vel, "s": s_vel}
    config["method"] = method
    config["initial_points"] = inital_points
    config["covariance_prior"] = covariance_prior

    config["bfgs_bounds"] = (
                             (config["x(km)"][0] - 1, config["x(km)"][1] + 1),
                             (config["y(km)"][0] - 1, config["y(km)"][1] + 1),
                             (0, config["z(km)"][1] + 1),  # x
                             (None, None),  # t
                            )

    # The initial number of clusters is determined by
    # (Number of picks)/(Number of stations) * (oversampling factor).
    if config["method"] == "BGMM":
        config["oversample_factor"] = oversample_factor  # default is 10
    if config["method"] == "GMM":
        config["oversample_factor"] = oversample_factor

    # DBSCAN
    config["dbscan_eps"] = dbscan_eps
    config["dbscan_min_samples"] = dbscan_min_samples

    # filtering
    config["min_picks_per_eq"] = min_picks_per_eq
    config["min_p_picks_per_eq"] = min_p_picks_per_eq
    config["min_s_picks_per_eq"] = min_s_picks_per_eq
    config["max_sigma11"] = max_sigma11
    config["max_sigma22"] = max_sigma22
    config["max_sigma12"] = max_sigma12

    # Transform station coordinates
    stations[["x(km)", "y(km)"]] = stations.apply(
        lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)),
        axis=1)
    stations["z(km)"] = stations["elevation"].apply(lambda depth: -depth / 1e3)

    # Do association
    catalogs, assignments = association(picks, stations, config, method=config["method"])

    catalog = pd.DataFrame(catalogs)
    assignments = pd.DataFrame(assignments, columns=["pick_index", "event_index", "gamma_score"])
    if len(catalog) > 0:
        catalog.sort_values(by=["time"], inplace=True, ignore_index=True)
    else:
        msg = "Did not associate events"
        raise Exception(msg)

    # Transform earthquake locations to lat long
    event_lat = []
    event_long = []
    for x, y in zip(catalog["x(km)"], catalog["y(km)"]):
        long, lat = proj(x, y, inverse=True)
        event_lat.append(lat)
        event_long.append(long)
    catalog["latitude"] = event_lat
    catalog["longitude"] = event_long

    # Create earthquake catalog
    # Empty event lst
    event_lst = []
    # Loop over each event in GaMMA catalog
    for event_count in range(len(catalog)):
        event_idx = catalog["event_index"][event_count]
        event_picks = [picks.iloc[i] for i in assignments[assignments["event_index"] == event_idx]["pick_index"]]

        # Create dictionary for each station that contains P and S phases
        # Create list with obspy picks class
        picks_lst = []
        for p in event_picks:
            waveform_id = WaveformStreamID(network_code=p["id"].split(".")[0],
                                           station_code=p["id"].split(".")[1])
            comments = Comment(text=f'probability={p["prob"]}')
            pick = Pick(time=obspy.UTCDateTime(p["timestamp"]),
                        waveform_id=waveform_id,
                        phase_hint=p["type"].upper(),
                        comments=[comments])
            picks_lst.append(pick)

        origin = Origin(time=catalog["time"][event_count],
                        longitude=catalog["longitude"][event_count],
                        latitude=catalog["latitude"][event_count],
                        depth=catalog["z(km)"][event_count] * 1e3)
        comment = Comment(text="EQTransformer for picking and GaMMA for association")
        gamma_score = Comment(text=f"gamma_score = {catalog['gamma_score'][event_count]}")
        ev = Event(picks=picks_lst,
                   force_resource_id=True,
                   origins=[origin],
                   comments=[comment, gamma_score])

        # Append event to final event list
        event_lst.append(ev)

        # Remove events that are beneath a certain gamma_score threshold value
        # if catalog["gamma_score"][event_count] >= gamma_score_threshold:
        #     event_lst.append(ev)

    return event_lst


def pyocto_picks(seisbench_picks: list) -> dict:
    """

    :param seisbench_picks:
    :return:
    """
    station_picks = {}
    for pick in seisbench_picks:
        if pick.trace_id not in station_picks.keys():
            station_picks.update(
                {pick.trace_id: {
                    "trace_id": [],
                    "start_time": [],
                    "peak_time": [],
                    "end_time": [],
                    "peak_value": [],
                    "phase": []
                }
                }
            )

        # Write values to lists for each station
        station_picks[pick.trace_id]["trace_id"].append(pick.trace_id)
        station_picks[pick.trace_id]["start_time"].append(pick.start_time)
        station_picks[pick.trace_id]["peak_time"].append(pick.peak_time)
        station_picks[pick.trace_id]["end_time"].append(pick.end_time)
        station_picks[pick.trace_id]["peak_value"].append(pick.peak_value)
        station_picks[pick.trace_id]["phase"].append(pick.phase)

    # Convert each station dictionary to pandas Dataframe
    for trace_id, picks in station_picks.items():
        station_picks[trace_id] = pd.DataFrame(picks)

    return station_picks


def gamma_picks(seisbench_picks: list) -> pd.DataFrame:
    """

    :param seisbench_picks:
    :return:
    """
    picks = []
    for pick in seisbench_picks:
        picks.append({
            "id": pick.trace_id,
            "timestamp": obspy.UTCDateTime(pick.peak_value).datetime,
            "prob": pick.peak_value,
            "type": pick.phase.lower()
        })

    return pd.DataFrame(picks)


def picks_per_station(seisbench_picks: list,
                      association_method: str = "pyocto") -> (dict, pd.DataFrame):
    """

    :param seisbench_picks:
    :param association_method
    :return:
    """
    if association_method == "pyocto":
        return pyocto_picks(seisbench_picks=seisbench_picks)
    elif association_method == "gamma":
        return gamma_picks(seisbench_picks=seisbench_picks)
    else:
        msg = f"Method {association_method} for phase association is not implemented."
        raise ValueError(msg)


def get_tmp_picks(dirname):
    """

    :param dirname:
    :return:
    """
    files = glob.glob(os.path.join(dirname, "*.pick"))
    picks_dict = {}
    for index, filename in enumerate(files):
        with open(filename, "rb") as handle:
            picks = pickle.load(handle)

        if index == 0:
            all_picks = picks.picks
        else:
            all_picks += picks.picks

        # # Write all picks in one dataframe and one dataframe for each station
        # if len(pick_df) > 0:
        #     trace_id = pick_df["trace_id"][0]
        #     if trace_id in picks_dict.keys():
        #         picks_dict[trace_id] = pd.concat(objs=[pick_df, picks_dict[trace_id]],
        #                                          ignore_index=True)
        #     else:
        #         picks_dict.update({trace_id: pick_df})

    return all_picks


def tmp_picks(dirname):
    df = pd.read_csv(glob.glob(f"{dirname}/*.csv")[0])

    station_picks = {}
    for index in range(len(df)):
        if df["trace_id"][index] not in station_picks.keys():
            station_picks.update(
                {df["trace_id"][index]: {
                    "trace_id": [],
                    "start_time": [],
                    "peak_time": [],
                    "end_time": [],
                    "peak_value": [],
                    "phase": []
                }
                }
            )

        # Write values to lists for each station
        station_picks[df["trace_id"][index]]["trace_id"].append(df["trace_id"][index])
        station_picks[df["trace_id"][index]]["start_time"].append(df["start_time"][index])
        station_picks[df["trace_id"][index]]["peak_time"].append(df["peak_time"][index])
        station_picks[df["trace_id"][index]]["end_time"].append(df["end_time"][index])
        station_picks[df["trace_id"][index]]["peak_value"].append(df["peak_value"][index])
        station_picks[df["trace_id"][index]]["phase"].append(df["phase"][index])

    # Convert each station dictionary to pandas Dataframe
    for trace_id, picks in station_picks.items():
        station_picks[trace_id] = pd.DataFrame(picks)
        if os.path.isdir(os.path.join(dirname, "picks")) is False:
            os.makedirs(os.path.join(dirname, "picks"))
        station_picks[trace_id].to_csv(path_or_buf=os.path.join(dirname, "picks", f"{trace_id}.csv"))


def picks2df(seisbench_picks: list):
    """

    :param seisbench_picks:
    :return:
    """
    dataframe = {"trace_id": [],
                 "start_time": [],
                 "peak_time": [],
                 "end_time": [],
                 "peak_value": [],
                 "phase": []}
    for pick in seisbench_picks:
        dataframe["trace_id"].append(pick.trace_id)
        dataframe["start_time"].append(pick.start_time)
        dataframe["peak_time"].append(pick.peak_time)
        dataframe["end_time"].append(pick.end_time)
        dataframe["peak_value"].append(pick.peak_value)
        dataframe["phase"].append(pick.phase)

    return pd.DataFrame(dataframe)


def origin_diff(origin1, origin2):
    return origin1.time - origin2.time


def compare_picks(result_dir1, result_dir2, filename: str, residual: float = 0.5):
    """
    Comparing picks between two results

    :param result_dir1:
    :param result_dir2:
    :param residual:
    :return:
    """
    same_picks = 0
    same_s = 0
    same_p = 0
    total_picks1 = 0
    total_picks1_p = 0
    total_picks1_s = 0

    picks_result_dir1 = glob.glob(os.path.join(result_dir1, "picks", "*.csv"))
    # Loop over each station
    for filename_picks in picks_result_dir1:
        # Read pick dataframes from results directories
        picks1 = pd.read_csv(filename_picks)
        picks2 = pd.read_csv(os.path.join(result_dir2, "picks", f"{pathlib.Path(filename_picks).stem}.csv"))

        # Actual total picks
        total_picks1 += len(picks1)

        # Loop over each pick in picks2 and search in picks1 for similar pick
        start = 0
        for p2_index in tqdm.tqdm(range(len(picks2)), postfix=filename_picks):
            for p1_index in range(start, len(picks1)):
                # if (
                #     picks2["trace_id"][p2_index] == picks1["trace_id"][p1_index] and
                #     abs(obspy.UTCDateTime(picks2["peak_time"][p2_index]) - obspy.UTCDateTime(picks1["peak_time"][p1_index])) <= residual and
                #     picks2["phase"][p2_index] == picks1["phase"][p1_index]
                # ):
                if (
                        picks2["trace_id"][p2_index] == picks1["trace_id"][p1_index] and
                        abs(obspy.UTCDateTime(picks2["peak_time"][p2_index]) - obspy.UTCDateTime(
                            picks1["peak_time"][p1_index])) <= residual
                ):
                    same_picks += 1
                    # start = p1_index
                    if picks1["phase"][p1_index] == "P":
                        same_p += 1
                    elif picks1["phase"][p1_index] == "S":
                        same_s += 1
                    break

    with open(filename, "w") as f:
        f.write("Comparision of picks\n")
        f.write(result_dir1 + "\n")
        f.write(result_dir2 + "\n")
        f.write(f"Same picks: {same_picks} (of {total_picks1})\n")
        f.write(f"Same P picks: {same_p}\n")
        f.write(f"Same S picks: {same_s}\n")


def compare_catalogs(catalog1, catalog2, origin_time_diff=2.5, verbose=False):
    same_events_cat1 = []
    same_events_cat2 = []  # Length of cat1 and cat2 should be the same
    diff_events_cat1 = []
    diff_events_cat2 = []

    # Loop over each event in catalog1
    for event_cat1 in catalog1:
        found_double = False
        for event_cat2 in catalog2:
            # Check origin time
            if (abs(
                    origin_diff(origin1=event_cat1.origins[0], origin2=event_cat2.origins[0])) <=
                    origin_time_diff
            ):
                # Add events to new event lists
                same_events_cat1.append(event_cat1)
                same_events_cat2.append(event_cat2)
                found_double = True
                break

        # Did not find an event in catalog2 which is also part of catalog1
        if found_double is False:
            diff_events_cat1.append(event_cat1)

    # Check length of same events
    if len(same_events_cat1) != len(same_events_cat2):
        msg = "Something is wrong with same events"
        raise ValueError(msg)

    # Loop over all events of catalog2 and write them to diff_events_cat2 if not in same_events_cat2
    for event in catalog2:
        if event not in same_events_cat2:
            diff_events_cat2.append(event)

    # Sort same events to avoid wrong order for comparison
    same_events_cat1 = sort_events(events=same_events_cat1)
    same_events_cat2 = sort_events(events=same_events_cat2)

    if verbose is True:
        print(f"Detected {len(same_events_cat1)} that are in both catalogues.")
        print(f"Found {len(diff_events_cat1)} that are not in catalog2 but in catalog1.")
        print(f"Found {len(diff_events_cat2)} that are not in catalog1 but in catalog2.")

    return diff_events_cat1, diff_events_cat2


def compare_catalogs_from_result_dir(
        result_dir1, result_dir2, origin_time_diff: float = 2.5, verbose=False
):

    # Read catalogs from both directories
    catalog1 = obspy.read_events(os.path.join(result_dir1, "*.xml"))
    catalog2 = obspy.read_events(os.path.join(result_dir2, "*.xml"))

    return compare_catalogs(
        catalog1=catalog1,
        catalog2=catalog2,
        origin_time_diff=origin_time_diff,
        verbose=verbose
    )






if __name__ == "__main__":
    # compare_catalogs(result_dir1="/home/jheuel/nextcloud/code/seisbench_catalogues/results/steadbasis",
    #                  result_dir2="/home/jheuel/nextcloud/code/seisbench_catalogues/results/steadbasis_induced")

    compare_picks(result_dir1="/home/jheuel/code/seisbench_catalogues/results/steadbasis",
                  result_dir2="/home/jheuel/code/seisbench_catalogues/results/steadbasis_induced_tl",
                  filename="stead_comparison.txt",
                  residual=0.4)
