from configparser import ConfigParser

from toolbox.utils import (run)
from toolbox.config import (Config)

config = ConfigParser()
config.read('configs/uk.ini')

params = Config(config)

output_file = 'figs/uk-controlled-scaled-new_cases'

two_waves = run(
    params=params,
    export=True,
    file_name=f'{output_file}-2waves.pdf',
    second_wave=True,
    third_wave=False,
    inset_plot=True,
)

three_waves_no_vaccination = run(
    params=params,
    export=True,
    file_name=f'{output_file}-3waves.pdf',
    second_wave=True,
    third_wave=True,
    inset_plot=True,
)

for v in params.general.vaccination_rates:
    file_name = f'{output_file}-3waves-vaccination_{str(v).replace(".", "")}.pdf'
    vaccination_models = run(
        params=params,
        export=True,
        file_name=file_name,
        second_wave=True,
        third_wave=True,
        inset_plot=True,
        vaccination_rate=v,
        vaccination_begins=360,
        vaccination_effectiveness=0.7
    )
