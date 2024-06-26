import os
import json
import obspy

from core.functions import sort_events
from nll_functions import Event2NLL, update_events_from_nll


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
    # Run NonLinLoc
    nll = Event2NLL(catalog=catalog,
                    nll_basepath=nll_basepath,
                    vel_model=vel_model,
                    stations_json=station_json,
                    nll_executable=nll_executable)
    nll.run_nll()  #delete_dirs=["time", "model"])

    # Update each localisation of each event
    # Catalog is also filtered with depth uncertainty
    all_events = update_events_from_nll(station_json=station_json,
                                        nll_basepath=nll_basepath)

    # Export full catalog
    full_catalog = obspy.Catalog(events=sort_events(all_events))

    return full_catalog


if __name__ == "__main__":
    catalog = obspy.read_events("/home/jheuel/nextcloud/code/seisbench_catalogues/results/forge/forge_pyocto.xml")
    station_json = "/home/jheuel/nextcloud/code/seisbench_catalogues/results/forge/forge.json"

    nll_catalog = nll_wrapper(catalog=catalog,
                              station_json=station_json,
                              nll_basepath="/home/jheuel/tmp/NLL",
                              vel_model="/home/jheuel/nextcloud/code/seisbench_catalogues/velocity_models/velocity.nll",
                              nll_executable="/work/software/nlloc/7.00.16/src/bin")
