import os
import json
import obspy

from core.functions import sort_events
from core.nll_functions import Event2NLL, update_events_from_nll, check_nll_time


def station_json_from_inventory(inventory, station_json="../station_json/stations.json"):
    pass


def nll_wrapper(catalog: obspy.Catalog,
                station_json: str,
                nll_basepath: str = "/home/jheuel/tmp/NLL",
                vel_model: str = "./velocity_models/velocity.nll",
                nll_executable: str = "/work/software/nlloc/7.00.16/src/bin"):
    """

    :param catalog:
    :param station_json:
    :param nll_basepath:
    :param vel_model:
    :param nll_executable:
    :return:
    """
    # TODO: Wenn viele runs gleichzeitig laufen, greifen diese auf den gleichen locs und obs Ordner zu
    #       Das wird nicht funktionieren -> individuelles config file for jeden run
    #       Auch obs files bleiben ggf. erhalten und fließen mit in den Katalog ein
    # Run NonLinLoc
    # Check files in NLL basepath directory
    # nll_time_files_exist = check_nll_time(station_json=station_json,
    #                                       nll_basepath=nll_basepath)
    # if nll_time_files_exist is True:
    #     create_nll_files = False
    # else:
    #     create_nll_files = True

    nll = Event2NLL(catalog=catalog,
                    nll_basepath=nll_basepath,
                    vel_model=vel_model,
                    stations_json=station_json,
                    nll_executable=nll_executable,
                    create_files=True)

    # if nll_time_files_exist is True:
    #     nll.create_obs()
    #     nll.localise()
    # else:
    nll.run_nll()

    # Update each localisation of each event
    # Catalog is also filtered with depth uncertainty
    all_events = update_events_from_nll(station_json=station_json,
                                        nll_basepath=nll_basepath)

    # Export full catalog
    full_catalog = obspy.Catalog(events=sort_events(all_events))

    return full_catalog


def merge_catalogs(catalogs: list[obspy.Catalog | str]) -> obspy.Catalog:
    """

    :param catalogs:
    :return:
    """
    # Empty list for all events
    all_events = []

    # Loop over each catalog and add events to all_events
    for catalog in catalogs:
        if isinstance(catalog, str):
            catalog = obspy.read_events(catalog)
        all_events += catalog.events

    # Sort events by origin dates
    sorted_events = sort_events(events=all_events)

    return obspy.Catalog(events=sorted_events)


if __name__ == "__main__":
    catalog = obspy.read_events("/home/jheuel/nextcloud/code/seisbench_catalogues/results/forge/forge_pyocto.xml")
    station_json = "/home/jheuel/nextcloud/code/seisbench_catalogues/results/forge/forge.json"

    nll_catalog = nll_wrapper(catalog=catalog,
                              station_json=station_json,
                              nll_basepath="/home/jheuel/tmp/NLL",
                              vel_model="/home/jheuel/nextcloud/code/seisbench_catalogues/velocity_models/velocity.nll",
                              nll_executable="/work/software/nlloc/7.00.16/src/bin")
