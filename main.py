from toolbox.model import (SirModelType, PredictionTarget)
from toolbox.utils import run

total_population = 60_000_000
vaccination_rates = [0.001, 0.002, 0.003, 0.004]

two_waves = run(
    model_type=SirModelType.Controlled,
    prediction_target=PredictionTarget.NewCases,
    scaling_data=True,
    export=True,
    population=total_population,
    file_name='figs/uk-controlled-scaled-new_cases-2waves.pdf',
    second_wave=True,
    third_wave=False,
    inset_plot=True
)

three_waves_no_vaccination = run(
    model_type=SirModelType.Controlled,
    prediction_target=PredictionTarget.NewCases,
    scaling_data=True,
    export=True,
    population=total_population,
    file_name='figs/uk-controlled-scaled-new_cases-3waves.pdf',
    second_wave=True,
    third_wave=True,
    inset_plot=True
)

for v in vaccination_rates:
    vaccination_models = run(
        model_type=SirModelType.Controlled,
        prediction_target=PredictionTarget.NewCases,
        scaling_data=True,
        export=True,
        population=total_population,
        file_name=f'figs/uk-controlled-scaled-new_cases-3waves-vaccination_{str(v).replace(".","")}.pdf',
        second_wave=True,
        third_wave=True,
        inset_plot=True,
        vaccination_rate=v,
        vaccination_begins=360
    )
