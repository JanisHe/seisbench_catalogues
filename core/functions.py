import os
import glob
import json
import pathlib
import warnings
import gc

import tqdm
import pickle
import datetime
import obspy
import pyocto

import pandas as pd
import numpy as np
from pyproj import Proj
from typing import Union
import seisbench.models as sbm  # noqa

from obspy.core.event.origin import Pick, Origin
from obspy.core.event.event import Event
from obspy.core.event.base import WaveformStreamID, Comment, QuantityError
from obspy.geodetics.base import locations2degrees, degrees2kilometers, gps2dist_azimuth
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
        # TODO: Add location code to read correct data, if given in station_json
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


def get_daily_waveforms(julday: int,
                        year: int,
                        starttime: obspy.UTCDateTime,
                        endtime: obspy.UTCDateTime,
                        sds_path: str,
                        station: str,
                        network: str,
                        channel_code: str,
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
    pathname = sds_pathname.format(sds_path=sds_path,
                                   year=year,
                                   network=network,
                                   station=station,
                                   channel=channel_code,
                                   julday=jd)

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


def semblance_ensemble_annotations(seisbench_models: list,
                                   stream: obspy.Stream,
                                   delta_t: int = 50,
                                   v: float = 2,
                                   **kwargs) -> obspy.Stream():
    """

    :param seisbench_models:
    :param stream:
    :param delta_t:
    :param v:
    :param kwargs:
    :return:
    """
    # Check sampling rate of all models (if sampling rates are not equal, an error is raised)
    if not all(model.sampling_rate == seisbench_models[0].sampling_rate for model in seisbench_models):
        msg = "Sampling rate of loaded models are not equal but equal sampling rates are required."
        raise ValueError(msg)

    # Obtain picks from multiple models
    for index, model in enumerate(seisbench_models):
        prediction = model.annotate(stream=stream,
                                    **kwargs)

        # Allocate empty array for all predictions
        # Note, last row is for summed up prediction by ensemble method (i.e. len(seisbench_models) + 1)
        if index == 0:
            predictions_p = np.empty(shape=(len(seisbench_models) + 1, prediction.select(channel=f"*_P")[0].stats.npts))
            predictions_s = np.empty(shape=(len(seisbench_models) + 1, prediction.select(channel=f"*_S")[0].stats.npts))

        # Add prediction to empty array
        predictions_p[index, :] = prediction.select(channel=f"*_P")[0].data
        predictions_s[index, :] = prediction.select(channel=f"*_S")[0].data

    # Obtain sums for coherence
    sum_probs_p = np.sum(predictions_p[:-1, :], axis=0) ** 2
    sum_probs_s = np.sum(predictions_s[:-1, :], axis=0) ** 2

    sum_square_probs_p = np.sum(predictions_p[:-1, :] ** 2, axis=0)
    sum_square_probs_s = np.sum(predictions_s[:-1, :] ** 2, axis=0)

    # Apply coherence-based ensemble estimation method
    for i in range(predictions_p.shape[1]):
        weight_p = np.max(predictions_p[:-1, i])
        weight_s = np.max(predictions_s[:-1, i])

        lower_bound_sum = i - delta_t
        upper_bound_sum = i + delta_t
        if lower_bound_sum < 0:
            lower_bound_sum = 0
        if upper_bound_sum >= predictions_p.shape[1]:
            upper_bound_sum = predictions_p.shape[1] - 1

        coherence_p = np.sum(sum_probs_p[lower_bound_sum: upper_bound_sum]) / (
                len(seisbench_models) * np.sum(sum_square_probs_p[lower_bound_sum: upper_bound_sum]))
        coherence_s = np.sum(sum_probs_s[lower_bound_sum: upper_bound_sum]) / (
                len(seisbench_models) * np.sum(sum_square_probs_s[lower_bound_sum: upper_bound_sum]))

        # Multiply coherence by weight and apply power of v
        coherence_p = weight_p * coherence_p ** v
        coherence_s = weight_s * coherence_s ** v

        # Write coherence to final array
        predictions_p[-1, i] = coherence_p
        predictions_s[-1, i] = coherence_s

    # Create final obspy trace for model by copying last prediction and changing the data arrays
    coherence_stream = prediction.copy()
    coherence_stream.select(channel=f"*_P")[0].data = predictions_p[-1, :]
    coherence_stream.select(channel=f"*_S")[0].data = predictions_s[-1, :]
    coherence_stream.select(channel=f"*_N")[0].data = (np.ones(predictions_p.shape[1]) - predictions_p[-1, :] -
                                                       predictions_s[-1, :])

    return coherence_stream


def semblance_ensemble_picking(seisbench_models: list,
                               stream: obspy.Stream,
                               delta_t = 50,
                               v = 2,
                               P_threshold: float = 0.2,
                               S_threshold: float = 0.2,
                               **kwargs):

    coherence_stream = semblance_ensemble_annotations(seisbench_models=seisbench_models,
                                                      stream=stream,
                                                      delta_t=delta_t,
                                                      v=v,
                                                      **kwargs)

    # Classify model as usual with seisbench
    picks = seisbench_models[0].classify_aggregate(annotations=coherence_stream,
                                                   argdict=dict(P_threshold=P_threshold,
                                                                S_threshold=S_threshold)
                                                   )

    return picks


def picking(seisbench_model: Union[list, sbm.phasenet.PhaseNet],
            stream: obspy.Stream,
            batch_size: int = 512,
            P_threshold: float = 0.1,
            S_threshold: float = 0.1,
            **kwargs):

    if isinstance(seisbench_model, list) is True:
        picks = semblance_ensemble_picking(seisbench_models=seisbench_model,
                                           stream=stream,
                                           P_threshold=P_threshold,
                                           S_threshold=S_threshold,
                                           **kwargs
                                           )
    else:
        picks = seisbench_model.classify(stream,
                                         batch_size=batch_size,
                                         P_threshold=P_threshold,
                                         S_threshold=S_threshold,
                                         **kwargs
                                         )

    return picks


def daily_picks(julday: int,
                year: int,
                starttime: obspy.UTCDateTime,
                endtime: obspy.UTCDateTime,
                sds_path: str,
                station: str,
                network: str,
                channel_code: str,
                seisbench_model,
                batch_size: int = 512,
                P_threshold: float = 0.1,
                S_threshold: float = 0.1,
                sampling_rate: (None, float) = None,
                pathname: str = "tmp_picks",
                **kwargs):
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
    :param pathname:
    :param kwargs:
    :return:
    """
    stream = get_daily_waveforms(julday=julday,
                                 year=year,
                                 starttime=starttime,
                                 endtime=endtime,
                                 sds_path=sds_path,
                                 station=station,
                                 network=network,
                                 channel_code=channel_code,
                                 sampling_rate=sampling_rate)
    print(julday, year, station)

    # Pick with seisbench model
    if len(stream) > 0:
        picks = picking(seisbench_model=seisbench_model,
                        batch_size=batch_size,
                        P_threshold=P_threshold,
                        S_threshold=S_threshold,
                        stream=stream,
                        **kwargs)

        # Save picks as pickle
        filename = os.path.join(pathname, f"{network}_{station}_{year}_{julday}.pick")
        with open(filename, "wb") as handle:
            pickle.dump(picks, handle)

        del picks
        gc.collect()


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

    zipped_pairs = zip(dates, events)
    try:
        sorted_events = [x for _, x in sorted(zipped_pairs)]
    except TypeError:
        print(zipped_pairs)
        print("# BEGIN MESSAGE #\nDid not sort events\n# END MESSAGE #")
        return events

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


def event_picks(event):
    """

    :param event:
    :return:
    """
    picks = {}
    for pick in event.picks:
        id = f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}.{pick.waveform_id.location_code}"
        if id in picks.keys():
            picks[id].update({pick["phase_hint"]: pick["time"]})
        else:
            picks.update({id: {pick["phase_hint"]: pick["time"]}})

    return picks


def add_distances(picks: dict, stations: (str, pd.DataFrame),
                  event: obspy.core.event.event.Event,
                  hypocentral_distance: bool = True):
    """

    :param picks:
    :param stations:
    :param event:
    :param hypocentral_distance
    :return:
    """
    if isinstance(stations, str):
        stations = load_stations(station_json=stations)

    for key in picks:
        try:
            dataframe_index = list(stations["id"]).index(key)
        except ValueError:
            continue

        distance = locations2degrees(lat1=event.origins[0].latitude,
                                     long1=event.origins[0].longitude,
                                     lat2=stations["latitude"][dataframe_index],
                                     long2=stations["longitude"][dataframe_index])

        # Convert from degree to km
        distance = degrees2kilometers(degrees=distance)

        # Estimate hypocentral distance, otherwise it is epicentral distance
        if hypocentral_distance is True:
            distance = np.sqrt(event.origins[0].depth / 1e3 ** 2 + distance ** 2)

        # Add distance to pick dictionary
        picks[key].update({"distance_km": distance})

    # Sort picks with respect to distance
    try:
        picks = dict(sorted(picks.items(), key=lambda item: item[1]["distance_km"]))
    except KeyError:
        msg = "Did not assign distances to picks."
        warnings.warn(msg)

    return picks


def association_score(event, client, station_json, cutoff_radius=10):
    """
    Association score as described in Park et al., 2022: Basement fault activation before larger earthquakes in
    Oklahoma and Kansas.

    :param event:
    :param client:
    :param station_json:
    :param cutoff_radius:
    :return:
    """
    # Load station_json
    # TODO: Make copy of stations???
    if isinstance(station_json, str):
        stations = load_stations(station_json=station_json)
    elif isinstance(station_json, pd.DataFrame):
        stations = station_json

    # Get dictionary of picks and add distance
    picks = event_picks(event=event)

    # Obtain distance between localisation and each station
    distances_km = [degrees2kilometers(degrees=locations2degrees(
        lat1=event.origins[0].latitude,
        long1=event.origins[0].longitude,
        lat2=stations["latitude"][index],
        long2=stations["longitude"][index])) for index in range(len(stations))]
    weights = [np.min([1 / distance, 1 / cutoff_radius]) for distance in distances_km]
    stations = stations.assign(distance_km=distances_km)
    stations = stations.assign(weights=weights)

    # Add picks to stations dataframe
    stations = stations.assign(P_weights=[float(0)] * len(stations))
    stations = stations.assign(S_weights=[float(0)] * len(stations))
    for station, pick in picks.items():
        index = stations[stations["id"] == station].index.values[0]
        for phase, time in pick.items():
            stations.loc[index, f"{phase}_weights"] = stations.loc[index, "weights"]

    # Check whether data are available for each station
    for index, id in enumerate(stations["id"]):
        network, station, location = id.split(".")
        stream = client.get_waveforms(network=network,
                                      station=station,
                                      location=location,
                                      channel="*",
                                      starttime=event.origins[0].time - 10,
                                      endtime=event.origins[0].time + 30)

        # Remove row from station dataframe if now data are available
        if len(stream) == 0:
            stations = stations.drop(index)

    # Determine association score
    score = 1 / (2 * np.sum(stations["weights"])) * (np.sum(stations["P_weights"]) + np.sum(stations["S_weights"]))

    return score


def associate_pyocto(station_json: (str, pd.DataFrame),
                     picks,
                     velocity_model,
                     **kwargs):
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

    # Merge events to pick assignments
    assignments = pd.merge(events, assignments, left_on="idx", right_on="event_idx", suffixes=("", "_pick"))

    # Create final obspy catalogue from events and picks
    # Picks are stored in a dictionary for each event by idx of event
    picks_dct = {}
    origins = {}
    for i in range(len(assignments)):
        event_idx = assignments.loc[i, "idx"]
        if event_idx not in picks_dct.keys():
            picks_dct.update({event_idx: []})  # List with all picks for an event

        if event_idx not in origins.keys():
            origins.update({event_idx: Origin(time=obspy.UTCDateTime(assignments.loc[i, "time"]),
                                              latitude=assignments.loc[i, "longitude"],
                                              longitude=assignments.loc[i, "longitude"],
                                              depth=assignments.loc[i, "depth"] * 1000)
                            })
        # Create obspy pick
        network_code, station_code, location = assignments.loc[i, "station"].split(".")
        waveform_id = WaveformStreamID(network_code=network_code,
                                       station_code=station_code,
                                       location_code=location)
        pick = Pick(time=obspy.UTCDateTime(assignments.loc[i, "time_pick"]),
                    waveform_id=waveform_id,
                    phase_hint=assignments.loc[i, "phase"]
                    )
        # Append pick to event_idx in dictionary
        picks_dct[event_idx].append(pick)

    # Create obspy catalogue from picks_dct and origins
    event_list = []
    for event_idx in origins.keys():
        event_list.append(Event(origins=[origins[event_idx]],
                                picks=picks_dct[event_idx],
                                force_resource_id=True)
                          )

    # Sort events by date
    event_list = sort_events(events=event_list)

    return obspy.Catalog(event_list)


def associate_gamma(picks,
                    stations,
                    ncpu=4,
                    use_dbscan=True,
                    use_amplitude=False,
                    zlim=(0, 30),
                    p_vel: Union[float, None] = 5.0,
                    s_vel: Union[float, None] = 2.9,
                    eikonal: Union[dict, None] = None,
                    method="BGMM",
                    inital_points=[1, 1, 1],
                    covariance_prior=(5, 5),
                    oversample_factor=10,
                    dbscan_eps=4,
                    dbscan_min_samples=5,
                    min_picks_per_eq=8,
                    min_p_picks_per_eq=4,
                    min_s_picks_per_eq=4,
                    max_sigma11=2.0,
                    max_sigma22=1.0,
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
    :param eikonal:
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
    config["xlim_degree"] = config.pop("latitude")
    config["ylim_degree"] = config.pop("longitude")

    # Create projection from pyproj. Use a string that is then transfered to Proj (https://proj.org)
    proj = Proj(f"+proj=sterea +lat_0={config['center'][0]} +lon_0={config['center'][1]} +units=km")

    # Compute boundaries for x and y
    ylim1, xlim1 = proj(latitude=config['xlim_degree'][0], longitude=config['ylim_degree'][0])
    ylim2, xlim2 = proj(latitude=config['xlim_degree'][1], longitude=config['ylim_degree'][1])
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
    config["method"] = method
    config["initial_points"] = inital_points
    config["covariance_prior"] = covariance_prior

    # Write velocity model to config
    # If eikonal is defined, then eikonal is preferred
    if p_vel and s_vel and not eikonal:
        config["vel"] = {"p": p_vel, "s": s_vel}
    elif eikonal:
        config["eikonal"] = eikonal
        config["eikonal"]["xlim"] = config["x(km)"]
        config["eikonal"]["ylim"] = config["y(km)"]
        config["eikonal"]["zlim"] = config["z(km)"]

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
        warnings.warn(msg)
        return obspy.Catalog([])

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
                        time_errors=QuantityError(),
                        comments=[comments])
            pick.time_errors["uncertainty"] = (p["end_time"] - p["start_time"]) / 2
            picks_lst.append(pick)

        origin = Origin(time=catalog["time"][event_count],
                        longitude=catalog["longitude"][event_count],
                        latitude=catalog["latitude"][event_count],
                        depth=catalog["z(km)"][event_count] * 1e3)
        gamma_score = Comment(text=f"gamma_score = {catalog['gamma_score'][event_count]}")
        ev = Event(picks=picks_lst,
                   force_resource_id=True,
                   origins=[origin],
                   comments=[gamma_score])

        # Append event to final event list
        event_lst.append(ev)

        # Remove events that are beneath a certain gamma_score threshold value
        # if catalog["gamma_score"][event_count] >= gamma_score_threshold:
        #     event_lst.append(ev)

    return obspy.Catalog(event_lst)


def read_velocity_model(filename: str) -> pd.DataFrame:
    """

    :param filename:
    :return:
    """
    return pd.read_csv(filename)


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
            "timestamp": obspy.UTCDateTime(pick.peak_time).datetime,
            "prob": pick.peak_value,
            "type": pick.phase.lower(),
            "start_time": pick.start_time,
            "end_time": pick.end_time
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


def get_tmp_picks(dirname,
                  julday=None):
    """

    :param dirname:
    :return:
    """
    if julday:
        files = glob.glob(os.path.join(dirname, f"*{julday}.pick"))
    else:
        files = glob.glob(os.path.join(dirname, "*.pick"))
    for index, filename in enumerate(files):
        with open(filename, "rb") as handle:
            picks = pickle.load(handle)

        if index == 0:
            all_picks = picks.picks
        else:
            all_picks += picks.picks

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


def _get_origin_attrib(
    event: Event,
    attribute: str
) -> Union[float, obspy.UTCDateTime]:
    """ Get an event origin attribute. """
    try:
        return (event.preferred_origin() or event.origins[-1]
                ).__getattribute__(attribute)
    except (IndexError, AttributeError):
        return None


def _get_magnitude_attrib(
    event: Event,
    attribute: str,
    magnitude_type: str = None
) -> Union[float, str]:
    """ Get a magnitude attribute. """
    if magnitude_type:
        magnitude = [magnitude for magnitude in event.magnitudes
                     if magnitude.magnitude_type == magnitude_type]
        if len(magnitude) == 0:
            return None
        magnitude = magnitude[0]
    else:
        try:
            magnitude = event.preferred_magnitude() or event.magnitudes[-1]
        except IndexError:
            return None
    try:
        return magnitude.__getattribute__(attribute)
    except AttributeError:
        return None


def _get_arrival_for_amplitude(amplitude,
                               event: Event):
    ori = event.preferred_origin() or event.origins[-1]
    if amplitude.pick_id is None:
        print("Amplitude not matched to pick")
        return None
    pick = amplitude.pick_id.get_referred_object()
    if pick is None:
        print("Amplitude not matched to pick")
        return None
    # Need an arrival on this station to get the distance
    arrival = [arr for arr in ori.arrivals
                if arr.pick_id.get_referred_object().waveform_id.station_code == pick.waveform_id.station_code
                and arr.distance]
    if len(arrival) == 0:
        print(f"No arrival found for {pick.waveform_id.station_code}, skipping")
        return None
    arrival = arrival[0]
    return arrival


def _get_amplitude_value(amplitude) -> float:
    # Convert to m
    amp = amplitude.generic_amplitude
    if amplitude.type == "IAML":
        # Apply Wood Anderson sensitivity - ISAPEI standard is removed
        amp *= 2080.0
    if amplitude.unit is None:
        warnings.warn(
            "No amplitude unit specified, assuming this is m - "
            "use with caution!!!")
    elif amplitude.unit != "m":
        raise NotImplementedError(
            "Only written to work with SI displacement units.")
    return amp


def summarize_catalog(catalog: obspy.Catalog,
                      magnitude_type: str = None) -> pd.DataFrame:
    """
    Summarize a catalog into a sparse dataframe of origin information
    """
    if magnitude_type:
        events = [ev for ev in catalog.events
                  if magnitude_type in [mag.magnitude_type
                                        for mag in ev.magnitudes]]
    else:
        events = catalog.events
    event_ids = [ev.resource_id.id.split('/')[-1] for ev in events]
    origin_times = [_get_origin_attrib(ev, "time") for ev in events]
    latitudes = [_get_origin_attrib(ev, "latitude") for ev in events]
    longitudes = [_get_origin_attrib(ev, "longitude") for ev in events]
    depths = [_get_origin_attrib(ev, "depth") for ev in events]
    magnitudes = [
        _get_magnitude_attrib(ev, "mag", magnitude_type=magnitude_type)
        for ev in events]
    magnitude_types = [
        _get_magnitude_attrib(ev, "magnitude_type",
                              magnitude_type=magnitude_type)
        for ev in events]

    catalog_df = pd.DataFrame(data=dict(
        event_id=event_ids, origin_time=origin_times, latitude=latitudes,
        longitude=longitudes, depth=depths, magnitude=magnitudes,
        magnitude_type=magnitude_types))

    return catalog_df


def find_matching_events(
        catalog_1: obspy.Catalog,
        catalog_2: obspy.Catalog,
        time_difference: float = 5.0,
        epicentral_difference: float = 20.0,
        depth_difference: float = 40.0,
        magnitude_type: str = None,
) -> dict:
    """
    Find matching events between two catalogs.

    https://bitbucket.org/calum-chamberlain/utilities/src/b6699258edb6639eea31815a437a329041ed87f0/cjc_utilities/magnitude_inversion/magnitude_inversion.py#lines-241

    Parameters
    ----------
    catalog_1
        A catalog to compare to catalog_2
    catalog_2
        A catalog to compare to catalog_1
    time_difference
        Maximum allowed difference in origin time in seconds between matching
        events
    epicentral_difference
        Maximum allowed difference in epicentral distance in km between
        matching events
    depth_difference
        Maximum allowed difference in depth in km between matching events.
    magnitude_type
        Magnitude type for comparison, will only return events in catalog_1
        with this magnitude

    Returns
    -------
    Dictionary of matching events ids. Keys will be from catalog_1.
    """
    df_1 = summarize_catalog(catalog_1, magnitude_type=magnitude_type)
    df_2 = summarize_catalog(catalog_2)
    if len(df_2) == 0:
        return None

    swapped = False
    if len(df_1) > len(df_2):
        # Flip for efficiency, will loop over the shorter of the two and use
        # more efficient vectorized methods on the longer one
        df_1, df_2 = df_2, df_1
        swapped = True

    timestamp = min(min(df_1.origin_time), min(df_2.origin_time))
    comparison_times = np.array([t - timestamp for t in df_2.origin_time])

    print("Starting event comparison.")
    matched_quakes = dict()
    for i in tqdm.tqdm(range(len(df_1))):
        origin_time = obspy.UTCDateTime(df_1.origin_time[i])
        origin_seconds = origin_time - timestamp
        deltas = np.abs(comparison_times - origin_seconds)
        index = np.argmin(deltas)
        delta = deltas[index]
        if delta > time_difference:
            continue  # Too far away in time
        depth_sep = abs(df_1.depth[i] - df_2.depth[index]) / 1000.0  # to km
        if depth_sep > depth_difference:
            continue  # Too far away in depth
        # distance check
        dist, _, _ = gps2dist_azimuth(
            lat1=df_2.latitude[index],
            lon1=df_2.longitude[index],
            lat2=df_1.latitude[i],
            lon2=df_1.longitude[i])
        dist /= 1000.
        if dist > epicentral_difference:
            continue  # Too far away epicentrally
        matched_id = df_2.event_id[index]
        if matched_id in matched_quakes.keys():
            # Check whether this is a better match
            if delta > matched_quakes[matched_id]["delta"] or \
                    dist > matched_quakes[matched_id]["dist"] or \
                    depth_sep > matched_quakes[matched_id]["depth_sep"]:
                continue  # Already matched to another, better matched event
        matched_quakes.update(
            {matched_id: dict(
                delta=delta, dist=dist, depth_sep=depth_sep,
                matched_id=df_1.event_id[i])})

    # We just want the event mapper
    if not swapped:
        return {key: value["matched_id"]
                for key, value in matched_quakes.items()}
    else:
        return {value["matched_id"]: key
                for key, value in matched_quakes.items()}


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


def find_pick(dataframe, datetime, pick_resiudal=0.5):
    picktimes, phases = [], []
    for index in range(len(dataframe)):
        if abs(obspy.UTCDateTime(dataframe["peak_time"][index]) - obspy.UTCDateTime(datetime)) <= pick_resiudal:
            picktime, phase = dataframe["peak_time"][index], dataframe["phase"][index]
            picktimes.append(obspy.UTCDateTime(picktime))
            phases.append(phase)

    return picktimes, phases


def manual_seisbench(cat_man: obspy.Catalog,
                     result_dir_sb: str,
                     residual: float=0.3):
    """
    Compares picks of manual catalogue and catalogue (result_dir) created with seisbench catalogue project.
    """
    # Read all picks from SeisBench catalogue
    files = glob.glob(os.path.join(result_dir_sb, "picks", "*"))
    df_dict_sb = {}
    for filename in files:
        trace_id = pathlib.Path(filename).stem
        df_dict_sb.update({trace_id: pd.read_csv(filename)})

    # Loop over each event in manual catalog
    event_dict = {}  # key: origin time of event, second dict with phase in catalog and True or False,
                     # Maybe third dict with residuals
    for event in cat_man:
        event_dict.update({str(event.origins[-1].time): {"phase_manuel": [],
                                                         "pick_time": [],
                                                         "waveform_id": [],
                                                         "in_picklist": [],
                                                         "residual": [],
                                                         "num_pred_p_picks": 0,
                                                         "num_pred_s_picks": 0,
                                                         "waveform_id_pick": [],
                                                         "waveform_id_P_S": [],
                                                         "p_rate": 0.0,
                                                         "s_rate": 0.0}
                           })
        count_p = 0
        count_s = 0
        count_true_p = 0
        count_true_s = 0
        for pick in event.picks:  # Loop over each pick and compare pick with seisbench picks
            trace_id = (f"{pick.waveform_id.network_code}.{pick.waveform_id.station_code}."
                        f"{pick.waveform_id.location_code}")
            # Find whether pick is in SeisBench picks
            if trace_id in df_dict_sb.keys():
                time, phase = find_pick(dataframe=df_dict_sb[trace_id],
                                        datetime=pick.time,
                                        pick_resiudal=residual)

                # Append manual phase to dictionary
                event_dict[str(event.origins[-1].time)]["phase_manuel"].append(pick.phase_hint)
                event_dict[str(event.origins[-1].time)]["pick_time"].append(pick.time)
                event_dict[str(event.origins[-1].time)]["waveform_id"].append(trace_id)

                # True picks
                if pick.phase_hint.lower() in ["p", "pg", "pn"]:
                    count_true_p += 1
                elif pick.phase_hint.lower() in ["s", "sg", "sn"]:
                    count_true_s += 1

                if len(time) > 0:
                    found_pick = False
                    for p, t in zip(phase, time):
                        pick_residual = t - pick.time
                        if pick.phase_hint.lower() in ["p", "pg", "pn"] and p.lower() == "p":
                            event_dict[str(event.origins[-1].time)]["in_picklist"].append(True)
                            event_dict[str(event.origins[-1].time)]["residual"].append(pick_residual)
                            event_dict[str(event.origins[-1].time)]["waveform_id_pick"].append(f"{trace_id}_P")
                            count_p += 1
                            found_pick = True
                            break
                        elif pick.phase_hint.lower() in ["s", "sg", "sn"] and p.lower() == "s":
                            event_dict[str(event.origins[-1].time)]["in_picklist"].append(True)
                            event_dict[str(event.origins[-1].time)]["residual"].append(pick_residual)
                            event_dict[str(event.origins[-1].time)]["waveform_id_pick"].append(f"{trace_id}_S")
                            count_s += 1
                            found_pick = True
                            break
                    if found_pick is False:  # If phase of predicted pick does match phase of manual pick
                        event_dict[str(event.origins[-1].time)]["in_picklist"].append(False)
                        event_dict[str(event.origins[-1].time)]["residual"].append(np.nan)
                else:
                    event_dict[str(event.origins[-1].time)]["in_picklist"].append(False)
                    event_dict[str(event.origins[-1].time)]["residual"].append(np.nan)

        # Update P and S rate, i.e. how many picks have been detected in percent
        # event_dict[str(event.origins[-1].time)]["p_rate"] = count_p / count_true_p * 100
        # event_dict[str(event.origins[-1].time)]["s_rate"] = count_s / count_true_s * 100
        event_dict[str(event.origins[-1].time)]["num_pred_p_picks"] = count_p
        event_dict[str(event.origins[-1].time)]["num_pred_s_picks"] = count_s

        # Count waveforms with both P and S arrivals
        for id_P in event_dict[str(event.origins[-1].time)]["waveform_id_pick"]:
            if id_P[-2:] == "_P":  # Look up for S wave ID
                id_S = f"{id_P[:-2]}_S"
                if id_S in event_dict[str(event.origins[-1].time)]["waveform_id_pick"]:
                    event_dict[str(event.origins[-1].time)]["waveform_id_P_S"].append(id_P[:-2])

    return event_dict



if __name__ == "__main__":
    # compare_catalogs(result_dir1="/home/jheuel/nextcloud/code/seisbench_catalogues/results/steadbasis",
    #                  result_dir2="/home/jheuel/nextcloud/code/seisbench_catalogues/results/steadbasis_induced")

    # compare_picks(result_dir1="/home/jheuel/code/seisbench_catalogues/results/steadbasis",
    #               result_dir2="/home/jheuel/code/seisbench_catalogues/results/steadbasis_induced_tl",
    #               filename="stead_comparison.txt",
    #               residual=0.4)

    from obspy.clients.filesystem.sds import Client
    import torch
    import shutil

    event_dict = manual_seisbench(cat_man=obspy.read_events("/home/jheuel/work/ais/catalogs/sequence_march2024_subset.xml"),
                                  result_dir_sb="/home/jheuel/nextcloud/code/seisbench_catalogues/results/rittershoffen_bpfilter")

    model = torch.load("/home/jheuel/code/train_phasenet/models/propulate_rittershoffen/models/rittershoffen_0.09777.pt",
                       map_location=torch.device("cpu"))
    client = Client("/home/jheuel/data/SDS")

    # Determine if event can be associated successfully
    associated_events = 0
    for time, event in event_dict.items():
        p_rate = event["p_rate"]
        s_rate = event["s_rate"]
        num_p_picks = event["num_pred_p_picks"]
        num_s_picks = event["num_pred_s_picks"]
        p_s_picks = len(event["waveform_id_P_S"])

        if p_rate >= 0.5 and s_rate >= 0.5 and num_p_picks >= 4 and num_s_picks >= 4 and p_s_picks >= 3:
            associated_events += 1
        else:
            # shutil.copyfile(src=f"/home/jheuel/Pictures/sequence_march2024/rittershoffen_bpfilter/{time}.png",
            #                 dst=f"/home/jheuel/Pictures/sequence_march2024/not_associated/{time}.png")
            print(time)

    print("Number of possible associated events:", associated_events)
    print("Number of not possible associated events:", len(event_dict) - associated_events)


    # # Plot missed picks
    # for time, event in event_dict.items():
    #     if event["p_rate"] < 50 or event["s_rate"] < 50:
    #         for in_picklist, waveform_id, pick_time, phase in zip(event["in_picklist"], event["waveform_id"], event["pick_time"], event["phase_manuel"]):
    #             if in_picklist is False:
    #                 network, station, location = waveform_id.split(".")
    #                 stream = client.get_waveforms(network=network, station=station, location=location,
    #                                               channel="*", starttime=obspy.UTCDateTime(time) - 30,
    #                                               endtime=obspy.UTCDateTime(time) + 60)
    #                 stream.filter(type="bandpass", freqmin=5, freqmax=40)
    #
    #                 preds = model.annotate(stream,
    #                                        blinding=[250, 250],
    #                                        overlap=1500)
    #
    #                 fig = plt.figure()
    #                 ax1 = fig.add_subplot(211)
    #                 time_array = np.arange(0, stream[0].stats.npts) * stream[0].stats.delta
    #                 time_array = [stream[0].stats.starttime.datetime + datetime.timedelta(seconds=k) for k in time_array]
    #
    #                 for trace in stream:
    #                     ax1.plot(time_array, trace.data)
    #
    #                 ax2 = fig.add_subplot(212, sharex=ax1)
    #                 for trace in preds:
    #                     time_array = np.arange(0, trace.stats.npts) * trace.stats.delta
    #                     time_array = [trace.stats.starttime.datetime + datetime.timedelta(seconds=k) for k in time_array]
    #                     if trace.stats.channel[-1] != "N":
    #                         ax2.plot(time_array, trace.data)
    #
    #                 if phase.lower() in ["p", "pg", "pn"]:
    #                     color = "tab:blue"
    #                 else:
    #                     color = "tab:orange"
    #                 ax2.axvline(pick_time.datetime, color=color)
    #
    #                 plt.show()
