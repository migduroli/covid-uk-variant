from configparser import ConfigParser
from toolbox.model import (SirModelType, PredictionTarget)

import pandas as pd


class GeneralParams:
    def __init__(self, config: ConfigParser):
        self.total_population = config.getfloat('default', 'population')
        self.tests_per_thousand = config.getfloat('default', 'tests_per_thousand')

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
        model_type = SirModelType.Controlled \
            if config['default']['model'] == 'controlled' \
            else SirModelType.Free

        self._initialise(config, model_type)

    def _initialise(self, config, model_type: SirModelType):
        section = 'controlled' \
                if model_type == SirModelType.Controlled \
                else 'free'

        self.initial_guess = [
            config.getfloat(section=section, option='beta_ini'),
            config.getfloat(section=section, option='alpha_ini'),
        ]

        if model_type == SirModelType.Controlled:
            self.initial_guess += [
                [float(x) for x in config[section]['maturities_ini'].split(',')],
                [float(x) for x in config[section]['scales_ini'].split(',')]
            ]

        self.bounds = [
            [float(x) for x in config[section]['lower_bounds'].split(',')],
            [float(x) for x in config[section]['upper_bounds'].split(',')]
        ]


class VaccinationParams:
    def __init__(self, config: ConfigParser):
        self.population = config.getfloat('default', 'population')
        self.actual_data = config.getboolean('vaccination', 'actual_data')
        self.input_file = config['vaccination']['input_file']
        self.start = config.getfloat('vaccination', 'start')
        self.effectiveness = config.getfloat('vaccination', 'effectiveness')
        self.rates = [
            float(x) for x in config['vaccination']['rates'].split(',')
        ]

        self.average_rate = 0
        self.vaccination = False
        self.total_vaccines = 0

        self.data = pd.read_csv(self.input_file)

    def __str__(self):
        return f"\t|-> vaccination?: {self.vaccination}\n" \
            f"\t|-> actual data: {self.actual_data}\n" \
            f"\t|-> input file: {self.input_file if self.actual_data else None}\n" \
            f"\t|-> start: {None if self.actual_data else self.start}\n" \
            f"\t|-> effectiveness: {self.effectiveness}\n" \
            f"\t|-> average rate: {self.average_rate}"

    def _get_average_rate_per_day(self, time: float):
        return (time > self.start) * self.average_rate

    def _get_actual_rate_per_day(self, time):
        end_data_idx = self.data.shape[0] - 1
        idx = int(time)
        return (time <= end_data_idx) * self.data.iloc[min(idx, end_data_idx)].n + \
               (time > end_data_idx) * self._get_average_rate_per_day(time)

    def get_vaccines_per_day(self, time: float, remaining: float):

        v = self.vaccination * (
            (not self.actual_data) * self._get_average_rate_per_day(time) +
            self.actual_data * self._get_actual_rate_per_day(time)
        )

        v = min(v, remaining)

        return v


class Config:

    def __init__(self, config):
        self.general = GeneralParams(config)
        self.vaccination = VaccinationParams(config)
        self.training = TrainParams(config)
        self.fitting = FitParams(config)
        self.first_wave = FirstWaveParams(config)
        self.second_wave = SecondWaveParams(config)
        self.third_wave = ThirdWaveParams(config)
