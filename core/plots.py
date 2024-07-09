"""
Collection of functions to plot catalogs created with methods in this project
"""

import glob
import os
import datetime
import yaml
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import obspy

from typing import Union

from obspy.geodetics.base import locations2degrees, degrees2kilometers
from core.functions import load_stations


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
    picks = dict(sorted(picks.items(), key=lambda item: item[1]["distance_km"]))

    return picks


def start_endtime(picks: dict, time_before: float = 15, time_after: float = 25):
    starttime = obspy.UTCDateTime()
    endtime = obspy.UTCDateTime("1970 01 01")
    for key, phases in picks.items():
        if phases.get("P"):
            if starttime > phases["P"]:
                starttime = phases["P"]

        if phases.get("S"):
            if endtime < phases["S"]:
                endtime = phases["S"]

    starttime = starttime - time_before
    endtime = endtime + time_after

    return starttime, endtime


def plot_picks(time: datetime.datetime, plot_position: float, ax, **kwargs):
    ax.plot(
        [time, time],
        [plot_position - 0.4, plot_position + 0.4],
        **kwargs
    )


def find_pick(dataframe, datetime, pick_resiudal=0.5):
    picktimes, phases = [], []
    for index in range(len(dataframe)):
        if abs(obspy.UTCDateTime(dataframe["peak_time"][index]) - obspy.UTCDateTime(datetime)) <= pick_resiudal:
            picktime, phase = dataframe["peak_time"][index], dataframe["phase"][index]
            picktimes.append(obspy.UTCDateTime(picktime))
            phases.append(phase)

    return picktimes, phases


def plot_event(event, client, station_json=None, ax=None, component=None, channel=None,
               with_distance=True, plot_all_picks: bool = False, result_dir: (None, str) = None,
               filter_args: (None, dict) = {"type": "bandpass", "freqmin": 1, "freqmax": 45},
               norm: float = 0.45, title: (None, str) = None):

    # If not station_json, then plot streams with difference between source time and p-pick time
    # If no p_pick, than use relative s-picks???

    # Unpack picks from event
    picks = event_picks(event=event)

    # Add distances to picks
    if station_json:
        picks = add_distances(picks=picks, stations=station_json, event=event)

    # Plot waveforms of each station
    if not ax:
        fig = plt.figure(figsize=(9, 11))
        ax = fig.add_subplot(111)

    if with_distance is False:
        plot_position = 0

    labels = []   # Empty list for labels of each trace
    data_dict = {}

    # Obtain start- and endtime from all picks
    starttime, endtime = start_endtime(picks=picks)

    # Loop over each station in picks
    for key, arrivals in picks.items():
        network_code, station_code, location_code = key.split(".")

        if location_code == "":
            location_code = "*"

        # Read stream
        stream = client.get_waveforms(
            network=network_code,
            station=station_code,
            location=location_code,
            channel="*",
            starttime=starttime,
            endtime=endtime
        )

        if component or channel:
            stream = stream.select(component=component, channel=channel)

        # Filter stream and apply taper afterwards
        if filter_args:
            stream.filter(**filter_args)
            stream.taper(type="cosine", max_percentage=0.1)

        # Find all picks between start- and endtime of stream
        if plot_all_picks is True and result_dir:
            picks_df = pd.read_csv(os.path.join(result_dir, "picks", f"{key}.csv"))
            residual = (stream[0].stats.npts * stream[0].stats.delta) / 2
            all_picktimes, all_phases = find_pick(dataframe=picks_df,
                                                  datetime=stream[0].stats.starttime + residual,
                                                  pick_resiudal=residual)

        # Plot each trace in stream
        for trace in stream:
            time = np.arange(0, trace.stats.npts) * trace.stats.delta
            time = [trace.stats.starttime.datetime + datetime.timedelta(seconds=k) for k in time]

            # Get plot_position from distances
            if with_distance is True:
                plot_position = arrivals["distance_km"]

            # Normalize trace
            if np.max(np.abs(trace.data)) > 0:
                norm_trace = 1 / np.max(np.abs(trace.data)) * norm

            # Plot data from trace
            ax.plot(
                time,
                (trace.data * norm_trace) + plot_position,
                color="k",
                linewidth=0.5
            )

            labels.append(
                f"{trace.stats.network}.{trace.stats.station}."
                f"{trace.stats.location}.{trace.stats.channel}"
            )

            if with_distance is True:
                ax.text(
                    x=0.01,
                    y=plot_position + 0.25,
                    s=labels[-1],
                    fontsize="x-small",
                    transform=ax.get_yaxis_transform()
                )

            # Plot picks in each trace
            if arrivals.get("P"):
                plot_picks(
                    time=arrivals["P"].datetime,
                    plot_position=plot_position,
                    ax=ax,
                    color="r",
                    zorder=2,
                    linewidth=1.5
                )
            if arrivals.get("S"):
                plot_picks(
                    time=arrivals["S"].datetime,
                    plot_position=plot_position,
                    ax=ax,
                    color="b",
                    zorder=2,
                    linewidth=1.5
                )

            # Plot all other picks
            if plot_all_picks is True:
                for pick_datetime, pick_phase in zip(all_picktimes, all_phases):
                    color = "r" if pick_phase == "P" else "b"
                    plot_picks(
                        time=pick_datetime,
                        plot_position=plot_position,
                        ax=ax,
                        color=color,
                        zorder=2,
                        linewidth=1.5
                    )

            # Update data_dict
            data_dict[f"{trace.stats.network}.{trace.stats.station}."
                      f"{trace.stats.location}.{trace.stats.channel}"] = {
                "data": trace.data * norm_trace,
                "time": time,
                "plot_position": plot_position,
                "trace_id": key,
                "P": arrivals.get("P"),
                "S": arrivals.get("S"),
                "with_distance": with_distance
            }

            # Update plot_position
            if with_distance is False:
                plot_position += 1

    # Set x- and y-labels and limits
    ax.set_xlim(starttime.datetime, endtime.datetime)
    ax.set_xlabel(f"Time (UTC) on {starttime.strftime('%Y %m %d')}")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f"{event.origins[0].time}\n{event.origins[0].latitude} | {event.origins[0].longitude} | "
            f"{np.round(event.origins[0].depth / 1e3, 2)} km"
        )

    # Set up y-label
    if with_distance is False:
        ax.set_yticks([])
        ax.set_yticks(ticks=np.arange(0, len(labels)))
        ax.set_yticklabels(labels)
    else:
        ax.set_ylabel("Epicentral distance (km)")

    if not ax:
        plt.show()   # TODO: ax always exists
    else:
        return data_dict


def plot_from_data_dict(data_dict: dict, result_dir: str, ax=None, pick_residual: float = 0.25,
                        plot_all_picks: bool = False, title: (str, None) = None):
    # Plot waveforms of each station
    if not ax:
        fig = plt.figure(figsize=(9, 11))
        ax = fig.add_subplot(111)

    # Set title
    if title:
        ax.set_title(title)

    # Loop over each key in data_dict
    for label, data_dct in data_dict.items():
        # Find picks for station
        pick_df = pd.read_csv(glob.glob(os.path.join(result_dir, "picks", f"{data_dct['trace_id']}.csv"))[0])

        # Plot data
        ax.plot(
            data_dct["time"],
            data_dct["data"] + data_dct["plot_position"],
            color="k",
            linewidth=0.5
        )

        if data_dct["with_distance"] is True:
            ax.text(
                x=0.01,
                y=data_dct["plot_position"] + 0.25,
                s=label,
                fontsize="x-small",
                transform=ax.get_yaxis_transform()
            )

        # Plot picks from different model
        for phase in ["P", "S"]:
            if data_dct.get(phase):
                if plot_all_picks is True:
                    central_time = obspy.UTCDateTime(data_dct["time"][int(len(data_dct["time"]) / 2)])
                    all_picks_residual = (obspy.UTCDateTime(data_dct["time"][-1]) -
                                          obspy.UTCDateTime(data_dct["time"][0])) / 2
                    picktimes, pick_phases = find_pick(dataframe=pick_df,
                                                       datetime=central_time,
                                                       pick_resiudal=all_picks_residual)
                else:
                    picktimes, pick_phases = find_pick(dataframe=pick_df,
                                                       datetime=data_dct[phase],
                                                       pick_resiudal=pick_residual)

                for picktime, pick_phase in zip(picktimes, pick_phases):
                    color = "r" if pick_phase == "P" else "b"
                    if picktime:
                        plot_picks(
                            time=picktime.datetime,
                            plot_position=data_dct["plot_position"],
                            ax=ax,
                            color=color,
                            zorder=2,
                            linewidth=1.5
                        )


def plot_all_traces_with_picks(result_dir: str, client, ax=None,
                               component=None, channel=None,
                               filter_args: (None, dict) = {"type": "bandpass", "freqmin": 1, "freqmax": 45},
                               norm: float = 0.45,
                               starttime: Union[obspy.UTCDateTime, str, None] = None,
                               endtime: Union[obspy.UTCDateTime, str, None] = None):

    if isinstance(starttime, str):
        starttime = obspy.UTCDateTime(starttime)

    if isinstance(endtime, str):
        endtime = obspy.UTCDateTime(endtime)

    labels = []
    plot_position = 0

    showfig = False
    if not ax:
        fig = plt.figure(figsize=(9, 11))
        ax = fig.add_subplot(111)
        showfig = True

    # Read start- and endtime from yml file
    yml_file = glob.glob(os.path.join(result_dir, "*.yml"))[0]
    with open(yml_file, "r") as f:
        parameters = yaml.safe_load(f)

    if starttime and endtime:
        parameters["data"]["starttime"] = starttime
        parameters["data"]["endtime"] = endtime

    if obspy.UTCDateTime(parameters["data"]["endtime"]) - obspy.UTCDateTime(parameters["data"]["starttime"]) > 86400:
        msg = "Files are too long"
        raise IOError(msg)

    # Load stations
    stations = load_stations(station_json=parameters["data"]["stations"])

    # Read all picks from each directory
    for station in tqdm.tqdm(stations["id"]):
        network_code, station_code, location_code = station.split(".")

        # Read stream
        stream = client.get_waveforms(
            network=network_code,
            station=station_code,
            location=location_code,
            channel="*",
            starttime=obspy.UTCDateTime(parameters["data"]["starttime"]),
            endtime=obspy.UTCDateTime(parameters["data"]["endtime"])
        )

        # Read starttime for labeling
        if len(stream) > 0:
            starttime_str = stream[0].stats.starttime.strftime('%Y %m %d')

        # Read picks
        try:
            pick_df = pd.read_csv(os.path.join(result_dir, "picks", f"{station}.csv"))
            # Convert pick peak times to UTCDateTime
            peak_times = pick_df["peak_time"].to_list()
            for index in range(len(peak_times)):
                peak_times[index] = obspy.UTCDateTime(peak_times[index])
            pick_df["peak_time"] = peak_times
        except FileNotFoundError:
            pick_df = None

        # Filter picks between start- and endtime
        if starttime and endtime and pick_df is not None:
            pick_df = pick_df.loc[pick_df['peak_time'] >= starttime]
            pick_df = pick_df.loc[pick_df['peak_time'] <= endtime]

        if component or channel:
            stream = stream.select(component=component, channel=channel)

        # Filter stream and apply taper afterwards
        if filter_args:
            stream.filter(**filter_args)
            stream.taper(type="cosine", max_percentage=0.01)

        # Plot each trace in stream
        for trace in stream:
            time = np.arange(0, trace.stats.npts) * trace.stats.delta
            time = [trace.stats.starttime.datetime + datetime.timedelta(seconds=k) for k in time]

            # Normalize trace
            if np.max(np.abs(trace.data)) > 0:
                norm_trace = 1 / np.max(np.abs(trace.data)) * norm
            else:
                norm_trace = norm

            # Plot data from trace
            ax.plot(
                time,
                (trace.data * norm_trace) + plot_position,
                color="k",
                linewidth=0.5
            )

            labels.append(
                f"{trace.stats.network}.{trace.stats.station}."
                f"{trace.stats.location}.{trace.stats.channel}"
            )

            # Plot all picks in picks_df
            if pick_df is not None:
                for index in pick_df.index.to_list():
                    color = "r" if pick_df["phase"][index] == "P" else "b"
                    plot_picks(
                        time=obspy.UTCDateTime(pick_df["peak_time"][index]).datetime,
                        plot_position=plot_position,
                        ax=ax,
                        color=color,
                        zorder=2,
                        linewidth=1.5
                    )

            # Update plot position
            plot_position += 1

    # Set up y-label
    ax.set_yticks([])
    ax.set_yticks(ticks=np.arange(0, len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Time (UTC) on {starttime_str}")

    if showfig is True:
        plt.show()




if __name__ == "__main__":
    from obspy.clients.filesystem.sds import Client
    import random
    from functions import compare_catalogs

    client = Client("/home/jheuel/data/SDS")
    # client = Client("/data_scc/KABBA_archive_SDS")

    dirname = "/home/jheuel/code/seisbench_catalogues/results/rittershofen_jan2024"
    catalog = obspy.read_events(glob.glob(os.path.join(dirname, "*.xml"))[0])
    station_json = glob.glob(os.path.join(dirname, "*.json"))[0]

    # dirname_stead = "/home/jheuel/code/seisbench_catalogues/results/rittershofen_stead"

    plot_all_traces_with_picks(dirname, client,
                               starttime=obspy.UTCDateTime("20240128 02:45"),
                               endtime=obspy.UTCDateTime("20240128 02:55"))
    for event in catalog.events:
        fig = plot_event(event=event, client=client, station_json=station_json, with_distance=False)
        plt.show()

    # catalog = obspy.read_events("/home/jheuel/code/seisbench_catalogues/results/forge_catalog_diting/forge_catalog_diting.xml")
    # station_json = "../station_json/FORGE.json"
    # catalog2 = obspy.read_events("/home/jheuel/code/seisbench_catalogues/results/forge_catalog_induced_stead/forge_catalog_induced_stead.xml")
    # events1, events2 = compare_catalogs(catalog, catalog2, verbose=True)

    # events = catalog.events[::-1]
    # random.shuffle(events)

    # for event in events2:
    #     fig = plt.figure(figsize=(15, 11))
    #     ax1 = fig.add_subplot(121)
    #     ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
    #     data_dict = plot_event(event=event, client=client, station_json=station_json, component="Z",
    #                            with_distance=True, ax=ax2, title="Induced STEAD")
    #     plot_from_data_dict(data_dict=data_dict,
    #                         result_dir="/home/jheuel/nextcloud/code/seisbench_catalogues/results/forge_catalog_diting",
    #                         ax=ax1, pick_residual=0.5, title="Diting", plot_all_picks=True)
    #     plt.tight_layout()
    #     plt.show()

    # dirs = glob.glob("/home/jheuel/nextcloud/code/seisbench_catalogues/results/steadbasis*")
    # fig = plt.figure()
    # for i, dirname in enumerate(dirs):
    #     if i == 0:
    #         ax = fig.add_subplot(1, len(dirs), 1)
    #     else:
    #         ax = fig.add_subplot(1, len(dirs), i + 1, sharex=ax, sharey=ax)
    #     ax.set_title(dirname.split("/")[-1])
    #
    #     plot_all_traces_with_picks(result_dir=dirname,
    #                                client=client, ax=ax, component="Z")
    # plt.show()
