import os
import glob
import shutil
import datetime
import pandas as pd

from obspy.core.event import read_events, Catalog

from core.functions import load_stations, area_limits


class Event2NLL:

    def __init__(self, catalog: Catalog, nll_basepath: str, vel_model: str, stations_json: str,
                 nll_executable="/work/software/nlloc/7.00.16/src/bin", create_files=True, **kwargs,):
        self.catalog = catalog
        self.nll_basepath = nll_basepath
        self.vel_model = vel_model
        self.nll_executables = nll_executable
        self.stations_json = stations_json

        self.event_names = [str(event.origins[0].time) for event in catalog]
        self.config_name = "_".join(str(datetime.datetime.now()).split())
        if isinstance(stations_json, pd.DataFrame):
            self.df_stations = stations_json
        else:
            self.df_stations = load_stations(station_json=stations_json)
        self.df_vel_model = None
        self.conf_file = os.path.join(self.nll_basepath, 'conf', f'{self.config_name}.conf')

        # Compute midpoint from all stations in station_json
        limits = area_limits(stations=self.df_stations)
        self.latorig = limits["center"][0]
        self.longorig = limits["center"][1]

        if create_files is True:
            self.create_nll_dirs()
            self.create_obs()
            self.create_config(**kwargs)
        else:
            self.conf_file = glob.glob(os.path.join(self.nll_basepath, 'conf', '*'))[0]

    def create_nll_dirs(self, delete_basepath=True):
        if os.path.isdir(self.nll_basepath) and delete_basepath is True:
            shutil.rmtree(self.nll_basepath)

        for dir_name in ["conf", "model", "obs", "time", "locs"]:
            if os.path.isdir(os.path.join(self.nll_basepath, dir_name)) is False:
                os.makedirs(os.path.join(self.nll_basepath, dir_name))

    def create_obs(self):
        """
        Creates obs file for NLL.
        One file that contains all observations
        """
        s = "{:<5} ? {:<3} ? {:<3} ? {:>25} GAU 0.5 -1 -1 -1 1\n"
        # Create one obs file that contains all events
        with open(os.path.join(self.nll_basepath, "obs", "events.obs"), "w") as f_obs:
            for event in self.catalog.events:
                f_obs.write(f"# Origin time: {event.origins[0].time}\n")
                for pick in event.picks:
                    datetime_str = pick.time.datetime.strftime("%Y%m%d %H%M %S.%f")
                    f_obs.write(s.format(pick.waveform_id.station_code, "Z", pick.phase_hint.upper(),
                                         datetime_str))
                f_obs.write("\n\n")

    def create_config(self, gtfile="P",
                      signature="LOCSIG Janis Heuel - KIT",
                      comment="LOCCOM Test"):
        """
        Creates config file for NLL.
        """
        if gtfile.upper() not in ["P", "S"]:
            msg = "Parameter gtfile must be either 'S' or 'P' and not {}".format(gtfile)
            raise ValueError(msg)

        if signature.split()[0] != "LOCSIG" and len(signature.split()) <= 1:
            msg = "signature must start with 'LOCSIG' and a further signature."
            raise ValueError(msg)

        if comment.split()[0] != "LOCCOM" and len(comment.split()) <= 1:
            msg = "comment must start with 'LOCCOM' and a further comment."
            raise ValueError(msg)

        # Load velocity model
        self.__read_velocity_model()

        # Write config file
        with open(os.path.join(self.nll_basepath, "conf", f"{self.config_name}.conf"), "w") as f_conf:
            f_conf.write("# Generic control file statements\n")
            f_conf.write("CONTROL 2 51904\n")
            # Write earthquake location from phase associator
            #f_conf.write(f"TRANS SIMPLE {self.event.origins[0].latitude} {self.event.origins[0].longitude} 0.0\n")
            f_conf.write(f"TRANS SIMPLE {self.latorig} {self.longorig} 0.0\n")
            f_conf.write("# END of Generic control file statements\n")
            # Write Vel2Grid control file statements
            f_conf.write("# Vel2Grid control file statements\n")
            f_conf.write(f"VGOUT {os.path.join(self.nll_basepath, 'model', self.config_name)}\n")
            f_conf.write("VGTYPE P\n")
            f_conf.write("VGTYPE S\n")
            f_conf.write("VGGRID 2 2000 7000 0.0 0.0 0.0 0.5 0.5 0.5 SLOW_LEN\n")
            # f_conf.write("VGGRID 2 10000 7000 0.0 0.0 0.0 0.5 0.5 0.5 SLOW_LEN\n")  # XXX Probably this needs adjustment
            # Write velocity model
            for ind in range(len(self.df_vel_model)):
                f_conf.write(f"LAYER {self.df_vel_model['depth'][ind]} {self.df_vel_model['vp'][ind]} 0 "
                             f"{self.df_vel_model['vs'][ind]} 0 {self.df_vel_model['density'][ind]} 0\n")
            f_conf.write("# END of Vel2Grid control file statements\n")
            # Write Grid2Time control statements
            f_conf.write("# Grid2Time control file statements\n")
            f_conf.write(f"GTFILES {os.path.join(self.nll_basepath, 'model', self.config_name)} "
                         f"{os.path.join(self.nll_basepath, 'time', self.config_name)} {gtfile.upper()}\n")
            f_conf.write("GTMODE GRID2D ANGLES_YES\n\n")

            # Write all station names and locations from json file to config
            for index in range(len(self.df_stations)):
                f_conf.write(f"GTSRCE {self.df_stations['id'][index].split('.')[1]} LATLON "
                             f"{self.df_stations['latitude'][index]} {self.df_stations['longitude'][index]} 0 0\n")

            f_conf.write("\nGT_PLFD 1.0e-3 0\n")
            f_conf.write("# END of Grid2Time control file statements\n")
            # Write NLL control file
            f_conf.write("# NLLoc control file statements\n")
            f_conf.write(f"{signature}\n")
            f_conf.write(f"{comment}\n")
            f_conf.write(f"LOCFILES {os.path.join(self.nll_basepath, 'obs', '*.obs')} NLLOC_OBS "
                         f"{os.path.join(self.nll_basepath, 'time', self.config_name)} "
                         f"{os.path.join(self.nll_basepath, 'locs', 'loc')}\n\n")
            f_conf.write("LOCPHASEID P\n")
            f_conf.write("LOCPHASEID S\n\n")
            f_conf.write("LOCGRID 6000 6000 1400 -200.0 -250.0 0.0 0.1 0.1 0.1 PROB_DENSITY SAVE\n\n")  # XXX
            f_conf.write("LOCSEARCH OCT 96 48 6 0.005 50000 10000 4 0\n\n")
            f_conf.write("LOCMETH EDT_OT_WT_ML 1.0e6 1 50 -1 -1.78 6 -1.0\n\n")
            f_conf.write("LOCQUAL2ERR 0.01 0.05 0.1 0.2 99999.9\n")
            f_conf.write("LOCANGLES ANGLES_NO 5\n")
            f_conf.write("LOCGAU 0.5 0.0\n")
            f_conf.write("LOCGAU2 0.01 0.05 2.0\n")
            f_conf.write("LOCHYPOUT SAVE_NLLOC_ALL\n\n")
            f_conf.write("# END of NLLoc control file statements\n")

    def __read_velocity_model(self):
        self.df_vel_model = pd.read_csv(self.vel_model)

    def run_vel2grid(self):
        execute = os.path.join(self.nll_executables, 'Vel2Grid')
        os.system(f"{execute} {self.conf_file}")

    def run_grid2time(self):
        execute = os.path.join(self.nll_executables, 'Grid2Time')
        # Run Grid2Time for P and S
        for phase in ["P", "S"]:
            if phase == "S":
                self.create_config(gtfile="S")
            os.system(f"{execute} {self.conf_file}")

    def localise(self):
        execute = os.path.join(self.nll_executables, 'NLLoc')
        os.system(f"{execute} {self.conf_file}")

    def run_nll(self, delete_dirs=None):
        self.run_vel2grid()
        self.run_grid2time()
        self.localise()

        # Delete travel time files in directory time
        if delete_dirs:
            for dir_name in delete_dirs:
                self.delete_dir_content(directory=dir_name)

    def delete_dir_content(self, directory):
        files = glob.glob(os.path.join(self.nll_basepath, directory, "*"))
        for filename in files:
            os.remove(filename)


def nll_object(nll):
    nll.run_nll(delete_dirs=["time", "model"])


def update_events_from_nll(station_json, nll_basepath, depth_filter=10000):
    # Read all .hyp files from NonLinLoc
    hyp_files = glob.glob(os.path.join(nll_basepath, 'locs', 'loc.*.grid0.loc.hyp'))

    # Remove sum file from .hyp files and read each fname as a single event and append to event_lst
    event_lst = []
    for fname in hyp_files:
        if os.path.split(fname)[1].split(".")[1] != "sum":
            nll_event = read_events(pathname_or_url=fname, format="NLLOC_HYP")[0]
            # Apply depth filter, i.e. if depth uncertainty of nll_event is greater than a threshold, the event
            # will not be used further. # Default value (5000 m) is taken from S-EqT Paper
            if depth_filter:
                depth_uncertainty = nll_event.origins[0].depth_errors.uncertainty
                if depth_uncertainty <= depth_filter:
                    event_lst.append(nll_event)
            else:
                event_lst.append(nll_event)

    # Since NonLinLoc only works with station names and not with station and network names, the network needs to
    # be added to each single pick for each event in event_lst
    # Load station_json as pandas dataframe
    if isinstance(station_json, pd.DataFrame):
        station_df = station_json
    else:
        station_df = load_stations(station_json=station_json)

    # Loop over each event and pick and add network and location to waveform_id
    for event in event_lst:
        for pick in event.picks:
            for id in station_df["id"]:
                if id.split(".")[1] == pick.waveform_id.station_code:
                    network, station, location = id.split(".")
                    break
            pick.waveform_id.network_code = network
            pick.waveform_id.location_code = location

    return event_lst
