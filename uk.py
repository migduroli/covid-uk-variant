import logging

from configparser import ConfigParser

from toolbox.utils import (run)
from toolbox.config import (Config)

logging.basicConfig(level=logging.INFO)

config = ConfigParser()
config.read('configs/uk.ini')

params = Config(config)

output_file = 'figs/uk-controlled-scaled-new_cases'

# one_wave = run(
#     params=params,
#     export=True,
#     file_name=f'{output_file}-1wave.pdf',
#     second_wave=False,
#     third_wave=False,
#     inset_plot=True,
# )

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
    inset_plot=True
)

for v in params.vaccination.rates:
    file_name = f'{output_file}-3waves-vaccination_{str(v).replace(".", "")}.pdf'

    params.vaccination.average_rate = v
    params.vaccination.vaccination = True

    result = run(
        params=params,
        figure=True,
        export=True,
        file_name=file_name,
        second_wave=True,
        third_wave=True,
        inset_plot=True
    )

    model = result['model']
    dt = result['dt']
    z = result['cases']
