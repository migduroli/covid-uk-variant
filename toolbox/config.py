from configparser import ConfigParser
from toolbox.model import (SirModelType, PredictionTarget)


class GeneralParams:
    def __init__(self, config: ConfigParser):
        self.total_population = config.getfloat('default', 'population')
        self.tests_per_thousand = config.getfloat('default', 'tests_per_thousand')

        self.vaccination_rates = [
            float(x) for x in config['default']['vaccination_rates'].split(',')
        ]

        self.model_type = SirModelType.Controlled \
            if config['default']['model'] == 'controlled' \
            else SirModelType.Free

        self.prediction_target = PredictionTarget.NewCases \
            if config['default']['target'] == 'new_cases' \
            else PredictionTarget.ActiveCases

        self.scaling_data = config.getboolean('default', 'scaling')

        self.data_file_path = config['default']['input_file']


class TrainParams:
    def __init__(self, config: ConfigParser):
        self.train_end_date = config['default']['train_end_date']
        self.control_delay = config.getfloat('default', 'control_delay')
        self.delay_time = config.getfloat('default', 'delay_time')
        self.inertial_time = config.getfloat('default', 'inertial_time')


class FirstWaveParams:
    def __init__(self, config: ConfigParser):
        self.time = config.getfloat('first_wave', 'time')


class SecondWaveParams:
    def __init__(self, config: ConfigParser):
        self.time = config.getfloat('second_wave', 'time')
        self.maturity = config.getfloat('second_wave', 'maturity')
        self.scale = config.getfloat('second_wave', 'scale')


class ThirdWaveParams:
    def __init__(self, config: ConfigParser):
        self.time = config.getfloat('third_wave', 'time')
        self.maturity = config.getfloat('third_wave', 'maturity')
        self.scale = config.getfloat('third_wave', 'scale')


class FitParams:
    def __init__(self, config):
        self.initial_guess = [
            config.getfloat(section='controlled', option='beta_ini'),
            config.getfloat(section='controlled', option='alpha_ini'),
            [float(x) for x in config['controlled']['maturities_ini'].split(',')],
            [float(x) for x in config['controlled']['scales_ini'].split(',')]
        ]

        self.bounds = [
            [float(x) for x in config['controlled']['lower_bounds'].split(',')],
            [float(x) for x in config['controlled']['upper_bounds'].split(',')]
        ]


class Config:

    def __init__(self, config):
        self.general = GeneralParams(config)
        self.training = TrainParams(config)
        self.fitting = FitParams(config)
        self.first_wave = FirstWaveParams(config)
        self.second_wave = SecondWaveParams(config)
        self.third_wave = ThirdWaveParams(config)
