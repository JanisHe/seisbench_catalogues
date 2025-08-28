"""
Collection of functions to plot catalogs created with methods in this project
"""

import glob
import os
import datetime
import warnings

import yaml
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import obspy

from typing import Union

from core.functions import load_stations, event_picks, add_distances, get_daily_waveforms, count_picks


def start_endtime(picks: dict,
                  time_before: float = 15,
                  time_after: float = 25):
    starttime = obspy.UTCDateTime()
    endtime = obspy.UTCDateTime("1970 01 01")
    for pick_dct in picks.values():
        for phase, time in pick_dct.items():
            if isinstance(time, obspy.UTCDateTime):
                if time < starttime:
                    starttime = time

                if time > endtime:
                    endtime = time

    starttime = starttime - time_before
    endtime = endtime + time_after

    # Last check if starttime is after endtime (happens e.g. when only S phases in picks)
    if starttime > endtime:
        starttime = endtime - time_after - time_before

    return starttime, endtime


def plot_picks(time: datetime.datetime,
               plot_position: float,
               ax,
               **kwargs):
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


def plot_event(event,
               client,
               station_json=None,
               ax=None,
               component=None,
               channel=None,
               with_distance=True,
               plot_all_picks: bool = False,
               result_dir: (None, str) = None,
               filter_args: (None, dict) = {"type": "bandpass", "freqmin": 1, "freqmax": 45},
               norm: float = 0.45,
               title: (None, str) = None,
               show_plot: bool=True):

    if ax:
        show_plot = False

    # If not station_json, then plot streams with difference between source time and p-pick time
    # If no p_pick, than use relative s-picks???

    # Unpack picks from event
    picks = event_picks(event=event)

    # Add distances to picks
    if station_json is not None:
        picks = add_distances(picks=picks, stations=station_json, event=event)

    # Plot waveforms of each station_picks
    if not ax:
        fig = plt.figure(figsize=(9, 11))
        ax = fig.add_subplot(111)

    if with_distance is False:
        plot_position = 0

    labels = []   # Empty list for labels of each trace
    data_dict = {}

    # Obtain start- and endtime from all picks
    starttime, endtime = start_endtime(picks=picks)

    # Loop over each station_picks in picks
    for key, arrivals in picks.items():
        network_code, station_code, location_code = key.split(".")

        if location_code in ["", "None"]:
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

        # # Denoise stream
        # import seisbench.models as sbm # noqa
        # import torch
        # model = sbm.SeisDAE.load("/home/jheuel/code/sb_denoiser/test_no_attention",
        #                          map_location=torch.device('cpu'))
        # stream = model.annotate(stream,
        #                         overlap=2500,
        #                         blinding=[500, 500])

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
            time_array = np.arange(0, trace.stats.npts) * trace.stats.delta
            time_array = [trace.stats.starttime.datetime + datetime.timedelta(seconds=k) for k in time_array]

            # Get plot_position from distances
            if with_distance is True:
                plot_position = arrivals["distance_km"]

            # Normalize trace
            if np.max(np.abs(trace.data)) > 0:
                norm_trace = 1 / np.max(np.abs(trace.data)) * norm

            # Plot data from trace
            ax.plot(
                time_array,
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
            for phase, time in arrivals.items():
                if phase.lower() in ["pg", "p", "pn"]:
                    color = "tab:blue"
                elif phase.lower() in ["sg", "s", "sn"]:
                    color = "tab:orange"
                elif phase.lower() == "distance_km":
                    continue
                else:
                    warnings.warn(f"Phase {phase} is unknown.")
                    color = "green"

                plot_picks(time=time,
                           plot_position=plot_position,
                           ax=ax,
                           color=color,
                           zorder=-1,
                           linewidth=1.5)

            # if arrivals.get("P"):
            #     plot_picks(
            #         time=arrivals["P"].datetime,
            #         plot_position=plot_position,
            #         ax=ax,
            #         color="r",
            #         zorder=2,
            #         linewidth=1.5
            #     )
            # if arrivals.get("S"):
            #     plot_picks(
            #         time=arrivals["S"].datetime,
            #         plot_position=plot_position,
            #         ax=ax,
            #         color="b",
            #         zorder=2,
            #         linewidth=1.5
            #     )

            # Plot all other picks
            if plot_all_picks is True:
                for pick_datetime, pick_phase in zip(all_picktimes, all_phases):
                    color = "tab:blue" if pick_phase == "P" else "tab:orange"
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
                "time": time_array,
                "plot_position": plot_position,
                "trace_id": key,
                "with_distance": with_distance
            }

            # Add phases to data dict
            for phase, time in arrivals.items():
                if isinstance(time, obspy.UTCDateTime):
                    data_dict[f"{trace.stats.network}.{trace.stats.station}."
                      f"{trace.stats.location}.{trace.stats.channel}"][phase] = time

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

    if show_plot is True:
        plt.show()   # TODO: ax always exists
    else:
        return data_dict


def plot_from_data_dict(data_dict: dict,
                        result_dir: str,
                        ax=None,
                        pick_residual: float = 0.25,
                        plot_all_picks: bool = False,
                        title: (str, None) = None):
    ax_default = ax

    # Plot waveforms of each station_picks
    if not ax:
        fig = plt.figure(figsize=(9, 11))
        ax = fig.add_subplot(111)

    # Set title
    if title:
        ax.set_title(title)

    # Loop over each key in data_dict
    for label, data_dct in data_dict.items():
        # Find picks for station_picks
        try:
            pick_df = pd.read_csv(glob.glob(os.path.join(result_dir, "picks", f"{data_dct['trace_id']}.csv"))[0])
        except:
            pick_df = None

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

        # Plot picks from different model (items of picks have obspy.UTCDateTime data format)
        for phase, time in data_dct.items():
            if isinstance(time, obspy.UTCDateTime):
                if plot_all_picks is True and pick_df is not None:
                    central_time = obspy.UTCDateTime(data_dct["time"][int(len(data_dct["time"]) / 2)])
                    all_picks_residual = (obspy.UTCDateTime(data_dct["time"][-1]) -
                                          obspy.UTCDateTime(data_dct["time"][0])) / 2
                    picktimes, pick_phases = find_pick(dataframe=pick_df,
                                                       datetime=central_time,
                                                       pick_resiudal=all_picks_residual)
                elif plot_all_picks is False and pick_df is not None:
                    picktimes, pick_phases = find_pick(dataframe=pick_df,
                                                       datetime=data_dct[phase],
                                                       pick_resiudal=pick_residual)

                if pick_df is not None:
                    for picktime, pick_phase in zip(picktimes, pick_phases):
                        if pick_phase.lower() in ["pg", "p", "pn"]:
                            color = "tab:blue"
                        elif pick_phase.lower() in ["sg", "s", "sn"]:
                            color = "tab:orange"
                        elif pick_phase.lower() == "distance_km":
                            continue
                        else:
                            warnings.warn(f"Phase {phase} is unknown.")
                            color = "green"

                        if picktime:
                            plot_picks(
                                time=picktime.datetime,
                                plot_position=data_dct["plot_position"],
                                ax=ax,
                                color=color,
                                zorder=2,
                                linewidth=1.5
                            )

    if not ax_default:
        return fig
        # for phase in ["P", "S"]:
        #     if data_dct.get(phase):
        #         if plot_all_picks is True:
        #             central_time = obspy.UTCDateTime(data_dct["time"][int(len(data_dct["time"]) / 2)])
        #             all_picks_residual = (obspy.UTCDateTime(data_dct["time"][-1]) -
        #                                   obspy.UTCDateTime(data_dct["time"][0])) / 2
        #             picktimes, pick_phases = find_pick(dataframe=pick_df,
        #                                                datetime=central_time,
        #                                                pick_resiudal=all_picks_residual)
        #         else:
        #             picktimes, pick_phases = find_pick(dataframe=pick_df,
        #                                                datetime=data_dct[phase],
        #                                                pick_resiudal=pick_residual)
        #
        #         for picktime, pick_phase in zip(picktimes, pick_phases):
        #             color = "r" if pick_phase == "P" else "b"
        #             if picktime:
        #                 plot_picks(
        #                     time=picktime.datetime,
        #                     plot_position=data_dct["plot_position"],
        #                     ax=ax,
        #                     color=color,
        #                     zorder=2,
        #                     linewidth=1.5
        #                 )



def plot_all_traces_catalog(catalog: obspy.Catalog,
                            starttime: Union[obspy.UTCDateTime, str, None] = None,
                            endtime: Union[obspy.UTCDateTime, str, None] = None,
                            ax=None,
                            component=None,
                            filter_args: (None, dict) = {"type": "bandpass", "freqmin": 1, "freqmax": 45},
                            norm: float = 0.45,
                            ):

    if isinstance(starttime, str):
        starttime = obspy.UTCDateTime(starttime)

    if isinstance(endtime, str):
        endtime = obspy.UTCDateTime(endtime)

    labels = []
    plot_position = 0
    # picks = picks_from_catalog(catalog=catalog) -> event_picks

    showfig = False
    if not ax:
        fig = plt.figure(figsize=(9, 11))
        ax = fig.add_subplot(111)
        showfig = True

    print()





def plot_all_traces_with_picks(result_dir: str,
                               client,
                               ax=None,
                               component=None,
                               channel=None,
                               filter_args: (None, dict) = {"type": "bandpass", "freqmin": 1, "freqmax": 45},
                               norm: float = 0.45,
                               starttime: Union[obspy.UTCDateTime, str, None] = None,
                               endtime: Union[obspy.UTCDateTime, str, None] = None,
                               plot_probability: bool = False,
                               catalog: Union[None, obspy.Catalog]=None,
                               ignore_stations: list = None):

    yaml_file = glob.glob(os.path.join(result_dir, "*.yml"))[0]

    with open(yaml_file, "r") as f:
        parameters = yaml.safe_load(f)

    if not starttime:
        starttime = obspy.UTCDateTime(parameters["data"]["starttime"])
    if not endtime:
        endtime = obspy.UTCDateTime(parameters["data"]["endtime"])

    if isinstance(starttime, str):
        starttime = obspy.UTCDateTime(starttime)

    if isinstance(endtime, str):
        endtime = obspy.UTCDateTime(endtime)

    labels = []
    plot_position = 0
    num_components = 3
    if component:
        num_components = len(component)

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
    stations = load_stations(glob.glob(os.path.join(dirname, "*.json"))[0])

    # Remove stations if ignore_stations exists
    if ignore_stations:
        drop_idx = []
        for station_ignore in ignore_stations:
            for idx, station_id in enumerate(stations["id"]):
                if station_ignore in station_id:
                    drop_idx.append(idx)
        stations = stations.drop(drop_idx)

    # Count number of P and S arrivals
    count_p = 0
    count_s = 0

    # Read all picks from each directory
    for station in tqdm.tqdm(stations["id"]):
        network_code, station_code, location_code = station.split(".")

        # Read stream
        stream = client.get_waveforms(
            network=network_code,
            station=station_code,
            location=location_code,
            channel="*",
            starttime=starttime,
            endtime=endtime
        )

        # # Denoise stream
        # import seisbench.models as sbm # noqa
        # import torch
        # model = sbm.SeisDAE.load("/home/jheuel/code/sb_denoiser/denoiser_ritt30",
        #                          map_location=torch.device('cpu'))
        # stream = model.annotate(stream)

        # if len(stream) != num_components:
        #     stream = get_daily_waveforms(julday=starttime.julday,
        #                                  year=starttime.year,
        #                                  starttime=starttime,
        #                                  endtime=endtime,
        #                                  sds_path=client.sds_root,
        #                                  station_picks=station_picks,
        #                                  network=network_code,
        #                                  channel_code="*",
        #                                  )

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
        else:  # demean
            stream.detrend(type="demean")

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
                    color = "tab:blue" if pick_df["phase"][index] == "P" else "tab:orange"
                    if pick_df["phase"][index] == "P":
                        count_p += 1
                    else:
                        count_s += 1
                    plot_picks(
                        time=obspy.UTCDateTime(pick_df["peak_time"][index]).datetime,
                        plot_position=plot_position,
                        ax=ax,
                        color=color,
                        zorder=-1,
                        linewidth=1.5
                    )

                    # Add probability in plot
                    if plot_probability is True:
                        ax.text(x=obspy.UTCDateTime(pick_df["peak_time"][index]).datetime,
                                y=plot_position + 0.2,
                                s=np.round(pick_df["peak_value"][index], 2))

            # Update plot position
            plot_position += 1

    # Add vertical lines from origin time if catalog is given
    if catalog:
        for event in catalog:
            ax.axvline(x=event.origins[0].time.datetime,
                       color="r")

    # Set up y-label
    ax.set_yticks([])
    ax.set_yticks(ticks=np.arange(0, len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Time (UTC) on {starttime_str}")
    ax.set_xlim([time[0], time[-1]])

    print("P", count_p)
    print("S", count_s)

    if showfig is True:
        plt.show()




if __name__ == "__main__":
    from obspy.clients.filesystem.sds import Client
    import random
    from functions import compare_catalogs, association_score

    client = Client("/home/jheuel/data/SDS")
    # client = Client("/data_scc/KABBA_archive_SDS")

    dirname = "/home/jheuel/code/seisbench_catalogues/results/albstadt"
    # dirname = "/home/jheuel/work/berichte/2024_seismica/catalogs/rittershoffen_sequence_pisdl"
    # dirname = "/home/jheuel/work/berichte/2024_seismica/catalogs/rittershoffen_012024_final_models13"
    # dirname = "/home/jheuel/code/seisbench_catalogues/results/rittershoffen_sequence_pisdl"
    catalog = obspy.read_events(glob.glob(os.path.join(dirname, "*.xml"))[0])
    # catalog = obspy.read_events(glob.glob(os.path.join(dirname, "*genie.xml"))[0])
    # catalog = obspy.read_events("/home/jheuel/work/ais/catalogs/association_tests/pisdl_harpa_sequence_march.xml")
    # catalog = obspy.read_events("/home/jheuel/work/ais/catalogs/sequence_march2024_subset_small.xml")
    station_json = glob.glob(os.path.join(dirname, "*.json"))[0]

    # dirname_stead = "/home/jheuel/code/seisbench_catalogues/results/rittershofen_stead"

    # plot_all_traces_with_picks(dirname, client, plot_probability=False,
    #                            catalog=catalog,
    #                            starttime=obspy.UTCDateTime("2024-03-02 06:25"),
    #                            endtime=obspy.UTCDateTime("2024-03-02 06:35"),
    #                            norm=0.6,
    #                            component="*")
                               # filter_args={"type": "bandpass", "freqmin": 5, "freqmax": 49})
    for index, event in enumerate(catalog.events):
        print(event.origins[0].time)
        fig = plot_event(event=event,
                         client=client,
                         station_json=station_json,
                         with_distance=False,
                         component="*",
                         show_plot=False)
        # a_score = association_score(event=event, station_json=station_json, client=client, cutoff_radius=1.5)
        # print(a_score)
        # plt.savefig(fname="/home/jheuel/Pictures/pisdl1/{:03d}.png".format(index), dpi=200)
        # plt.close()
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
