import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from toolbox.config import (Config)
from datetime import (date, timedelta)
from toolbox.date_params import (Months)
from toolbox.model import (SirModels, SirModelType, PredictionTarget, read_data)
from mpl_toolkits.axes_grid.inset_locator import (InsetPosition)

TIME_INITIAL = date(2020, 1, 1)
TIME_END = date(2021, 3, 2)
TIME_ARROW = [
    (TIME_INITIAL + timedelta(days=x)).strftime('%Y-%m-%d')
    for x in range((TIME_END - TIME_INITIAL).days + 1)
]


def setup_matplotlib(
        font_family: str = 'serif',
        font_size: int = 8,
        fig_size: tuple = (5, 3),
        fig_dpi: int = 100
):
    plt.rc('font', family=font_family)
    plt.rc('font', size=font_size)
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = fig_dpi
    # plt.rc('text', usetex=True)


def plot_init(
        model: SirModels,
        dt: np.array,
        simulation,
        x_axis: np.array,
        x_train: np.array,
        y_train: np.array,
        train_idx: int,
        x_total: np.array,
        y_total: np.array,
        total_idx: int,
        x_ticks: np.array,
        time_offset: int,
):
    fig, ax = plt.subplots()
    ax.grid(b=True, which='major', color='#666666', linestyle='--', alpha=0.25, axis='x')
    ax.set_xticks(x_axis[:-1])
    ax.set_xticklabels(x_ticks, rotation=90)
    ax.bar(time_offset + x_train, y_train, color='b', alpha=0.4)
    ax.bar(time_offset + x_total[train_idx:], y_total[train_idx:], color='red', alpha=0.4)
    ax.plot(time_offset + x_total, simulation[:total_idx + 1], '-', color='darkturquoise')
    ax.plot(time_offset + dt[total_idx:], simulation[total_idx:], '--', color='darkturquoise')
    model.add_confidence_intervals(dt=dt, z=simulation, ax=ax, time_offset=time_offset)
    return fig, ax


def setup_limits_and_ticks(axis, model_type: SirModelType, second_wave: bool, third_wave: bool):
    if model_type == SirModelType.Free:
        axis.set_ylim([0, 90_000])
        axis.set_yticks(np.arange(0, 80_000, 25_000))
    if model_type == SirModelType.Controlled and second_wave:
        axis.set_ylim([0, 140_000])
        axis.set_yticks(np.arange(0, 140_000, 30_000))
    if model_type == SirModelType.Controlled and third_wave:
        axis.set_ylim([0, 140_000])
        axis.set_yticks(np.arange(0, 140_000, 30_000))


def add_control_plot(axis, model: SirModels, model_type: SirModelType, dt: np.array, time_offset: int):
    if model_type == SirModelType.Controlled:
        model.add_control(axis, time_offset + dt)


def add_inset_plot(
        axis,
        inset_plot: bool,
        model: SirModels,
        dt: np.array,
        third_wave: bool,
        x_axis: np.array,
        y_total: np.array,
        simulation: np.array,
        train_idx: int,
        total_idx: int,
        time_offset: int,
        x_ticks: np.array,
):
    if inset_plot:
        ax_inset = plt.axes([0, 0, 1, 1])
        ip = InsetPosition(axis, [0.02, 0.55, 0.4, 0.4])
        ax_inset.set_axes_locator(ip)
        ax_inset.yaxis.tick_right()
        ax_inset.patch.set_alpha(0.7)

        limits = None
        color_style = None

        if not third_wave:
            color_style = 'blue'
            limits = [0, int(0.75 * train_idx)]
            ax_inset.set_xticks(x_axis[:8:2])
            ax_inset.set_xticklabels(x_ticks[:8:2], rotation=90)  # plt.show()
        if third_wave:
            color_style = 'red'
            limits = [train_idx, total_idx]
            ax_inset.set_xticks(x_axis[10:-1:2])
            ax_inset.set_xticklabels(x_ticks[10::2], rotation=90)  # plt.show()

        ax_inset.bar(
            time_offset + dt[limits[0]: limits[1]],
            y_total[limits[0]: limits[1]],
            color=color_style, alpha=0.4
        )

        ax_inset.plot(
            time_offset + dt[limits[0]:limits[1]],
            simulation[limits[0]: limits[1]],
            '-', color='darkturquoise'
        )

        model.add_confidence_intervals(
            dt=dt, z=simulation, ax=ax_inset, time_offset=time_offset, limits=limits
        )


def get_axis_data(df: pd.DataFrame, max_idx: int):
    x_origin = pd.to_datetime(df.date.min(), format="%Y-%m-%d")
    x_tick_idx = Months.List.Numbers.index(x_origin.month)
    x_ticks = [f'{m}-2020' for m in Months.List.Names[x_tick_idx:]] + \
              [f'{m}-2021' for m in Months.List.Names[0:max_idx]]

    lengths = np.array(Months.List.Lengths(year=2020)[x_tick_idx:] +
                       Months.List.Lengths(year=2021)[0:max_idx])
    x_axis = [0] + lengths.cumsum().tolist()
    x_offset = x_origin.day
    return x_axis, x_ticks, x_offset


def daily_tests_per_thousand(x, x_ini: date, x_end: date, y_ini: float, y_end: float):
    time_span = (x_end-x_ini).days
    [y0, yf] = [y_ini, y_end]
    slope = (yf-y0) / time_span
    corr = y0 + slope * (x + 11)
    return corr


def scaling_data_with_tests(x, params: Config):
    z = daily_tests_per_thousand(
        x,
        x_ini=date(2020, 1, 1), x_end=date(2021, 1, 1),
        y_ini=0, y_end=params.general.tests_per_thousand
    )
    corr = 1 + (1/z - 1/z[-1])

    return corr


def fit_model(
        params: Config,
        n_peaks: int,
        vaccination_rate: float,
        vaccination_begins: float,
        vaccination_effectiveness: float,
        rename_columns: dict = None,
):

    df = read_data(
        file=params.general.data_file_path,
        rename_columns=rename_columns,
        t_ini=TIME_INITIAL,
        t_end=TIME_END
    )[['date', 'new_cases']]

    y_total = df['new_cases']
    x_total = np.arange(0, len(y_total))

    if params.general.scaling_data:
        y_total *= scaling_data_with_tests(x_total, params=params)

    idx_train_ini = 0
    idx_train_end = df[df['date'] == params.training.train_end_date].index[0]

    x = x_total[idx_train_ini:idx_train_end]
    y = y_total[idx_train_ini:idx_train_end].reset_index(drop=True)

    m = SirModels(
        n_total=params.general.total_population,
        n_infected=y[0],
        model_type=params.general.model_type,
        prediction_target=params.general.prediction_target
    )

    m.initial_guess = params.fitting.initial_guess
    m.bounds = params.fitting.bounds
    m.vaccination_rate = vaccination_rate
    m.t_vaccination = vaccination_begins
    m.vaccination_effectiveness = vaccination_effectiveness

    if params.general.model_type == SirModelType.Controlled:
        m.control_delay = params.training.control_delay
        m.delay_time = params.training.delay_time
        m.inertial_time = params.training.inertial_time
        m.measure_init = [params.first_wave.time]
        m.n_peaks = n_peaks

    m.fit(x, y)
    return {'model': m,
            'x_train': x,
            'y_train': y,
            'data': df,
            'X': x_total,
            'Y': y_total}


def add_wave(m: SirModels, params: Config, n_wave: int):
    wave = params.second_wave
    if n_wave == 3:
        wave = params.third_wave

    m.measure_init = m.measure_init + [wave.time]
    m.maturities = m.maturities + [wave.maturity]
    m.scales = m.scales + [wave.scale]


def make_plot(
        m: SirModels,
        m_type: SirModelType,
        df: pd.DataFrame,
        x_train: np.array,
        y_train: np.array,
        x_total: np.array,
        y_total: np.array,
        second_wave: bool,
        third_wave: bool,
        inset_plot: bool = False
):
    max_months_2021 = 3 if not third_wave else 8

    x_axis, x_ticks, x_offset = get_axis_data(
        df=df,
        max_idx=max_months_2021
    )

    dt = np.arange(0, x_axis[-2])

    z = m.compute_trajectory(time=dt)

    train_idx = x_total.tolist().index(x_train[-1])
    total_idx = dt.tolist().index(x_total[-1])

    setup_matplotlib()

    fig, ax = plot_init(
        model=m,
        dt=dt,
        simulation=z,
        x_axis=x_axis,
        x_train=x_train,
        y_train=y_train,
        train_idx=train_idx,
        x_total=x_total,
        y_total=y_total,
        total_idx=total_idx,
        x_ticks=x_ticks,
        time_offset=x_offset,
    )

    setup_limits_and_ticks(
        axis=ax,
        model_type=m_type,
        second_wave=second_wave,
        third_wave=third_wave
    )

    add_control_plot(
        axis=ax,
        model=m,
        model_type=m_type,
        dt=dt,
        time_offset=x_offset
    )

    add_inset_plot(
        axis=ax,
        inset_plot=inset_plot,
        model=m,
        dt=dt,
        third_wave=third_wave,
        x_axis=x_axis,
        y_total=y_total,
        simulation=z,
        train_idx=train_idx,
        total_idx=total_idx,
        time_offset=x_offset,
        x_ticks=x_ticks,
    )

    return fig, ax


def run(params: Config,
        export: bool = False,
        file_name: str = None,
        second_wave: bool = True,
        third_wave: bool = False,
        inset_plot: bool = False,
        vaccination_rate: float = None,
        vaccination_begins: float = None,
        vaccination_effectiveness: float = None):

    opt = fit_model(
        rename_columns={'newCases': 'new_cases'},
        params=params,
        n_peaks=1,
        vaccination_rate=vaccination_rate,
        vaccination_begins=vaccination_begins,
        vaccination_effectiveness=vaccination_effectiveness,
    )

    model: SirModels = opt['model']
    x_train = opt['x_train']
    y_train = opt['y_train']
    data = opt['data']
    X = opt['X']
    Y = opt['Y']

    if (params.general.model_type == SirModelType.Controlled) and second_wave:
        add_wave(m=model, params=params, n_wave=2)

    if (params.general.model_type == SirModelType.Controlled) and third_wave:
        add_wave(m=model, params=params, n_wave=3)

    fig, ax = make_plot(
        m=model,
        m_type=params.general.model_type,
        df=data,
        x_train=x_train,
        y_train=y_train,
        x_total=X,
        y_total=Y,
        second_wave=second_wave,
        third_wave=third_wave,
        inset_plot=inset_plot
    )

    fig.tight_layout()
    if export:
        fig.savefig(file_name)

    print(model)
    return model
