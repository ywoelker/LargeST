from src.base.experiment import BaseExperiment, SparsityExperiment

def get_experiment(experiment_id = None):


    if experiment_id is None or experiment_id == '000':
        return BaseExperiment(
            name='Base Experiment',
            description='This is the base experiment with no specific configuration.',
        )