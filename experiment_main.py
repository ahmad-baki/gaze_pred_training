# MY_CW_MAIN.py
from cw2 import cw_error, experiment
from cw2.cw_data import cw_logging

from main import run_main


class MyExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Skip for Quickguide
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task
        print("Running MyExperiment with config:", config)
        run_main(config)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass

from cw2 import cluster_work

if __name__ == "__main__":
    print("Starting MyExperiment...")
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)

    # # Optional: Add loggers 
    # cw.add_logger(...)

    # RUN!
    cw.run() 