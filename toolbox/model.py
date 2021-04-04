import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from typing import Union, Any
from scipy.special import erf
from scipy.integrate import odeint
from scipy.optimize import curve_fit


# region Utils:

def flatten_list(l):
    v = []
    [v.append(e) if not isinstance(e, list) else v.extend(e) for e in l]
    return v


def make_list(arg: Union[Any, list]):
    if isinstance(arg, list):
        return arg
    elif isinstance(arg, tuple):
        return [e for e in arg]
    else:
        return [arg]


def read_data(file: str, rename_columns: dict, t_ini, t_end):
    df = pd.read_csv(file)
    df = df.rename(columns=rename_columns)
    df = df[
        (df.date >= t_ini.strftime("%Y-%m-%d")) &
        (df.date <= t_end.strftime("%Y-%m-%d"))].sort_values(
        by='date'
    )
    df = df.reset_index(drop=True)
    return df


def control(t: Union[float, np.array],
            t_init: Union[list, tuple],
            maturity: Union[list, tuple],
            effectiveness: Union[list, tuple],
            t_delay: float,
            t_inertia: float
            ):
    z = np.zeros_like(t)
    tuple_list = zip(make_list(t_init), make_list(maturity), make_list(effectiveness))
    for (t0, l, scale) in tuple_list:
        z = z + scale * 0.5 * ((1 + erf((t - t0) / t_delay)) - (1 + erf((t - (t0 + l)) / t_inertia)))
    return np.where(z > 0, z, 0)

# endregion


# region Aux:

class SirModelType(Enum):
    Free = 1,
    Controlled = 2


class PredictionTarget(Enum):
    ActiveCases = 1
    NewCases = 2

# endregion


# region SIR model

class SirModels:

    # region Cttor:
    def __init__(
            self,
            n_total: float, n_infected: float, n_recovered: float = 0,
            model_type: SirModelType = SirModelType.Free,
            prediction_target: PredictionTarget = PredictionTarget.ActiveCases,
            control_delay: float = 0,
            number_of_peaks: int = 1,
            n_vaccinated: int = 0,
    ):
        self._initialise(n_total, n_infected, n_recovered, model_type, prediction_target, control_delay, number_of_peaks, n_vaccinated)
        self._set_correct_flow()
        self._set_correct_target()

    def __str__(self):
        return f"\n" \
            f"SirModel type: {self.type}\n" \
            f"----------------------------------------------------\n" \
            f"N: {self.total_population}\n" \
            f"beta: {self.beta} +/- {self.beta_err}\n" \
            f"alpha: {self.alpha} +/- {self.alpha_err}\n" \
            f"----------------------------------------------------\n" \
            f"R0: {self.beta/self.alpha} +/- ({(self.beta_err/self.beta + self.alpha_err/self.alpha) * self.beta/self.alpha})\n" \
            f"PM times: {self.measure_init}\n" \
            f"PM scales: {self.scales} +/- {self.scales_err}\n" \
            f"Social delay: {self.delay_time}\n" \
            f"Social inertia: {self.inertial_time}\n" \
            f"----------------------------------------------------\n" \
            f"Vaccination params: \n" \
            f"{self.vaccination_params}\n" \
            f"----------------------------------------------------\n"

    def _initialise(
            self, n_total: float, n_infected: float, n_recovered: float,
            model_type: SirModelType, prediction_target: PredictionTarget, control_delay: float,
            n_peaks: int, n_vaccinated: int
    ):
        self.beta = None
        self.alpha = None
        #
        self.beta_err = None
        self.alpha_err = None
        #
        self.measure_init = None
        #
        self.maturities = None
        self.maturities_err = None
        #
        self.scales = None
        self.scales_err = None
        #
        self.initial_guess = None
        self.bounds = None
        self.delay_time = None
        self.inertial_time = None

        # self.vaccination_rate = None
        # self.vaccination_effectiveness = None
        # self.t_vaccination = None
        self.vaccination_params = None

        self.total_population = n_total
        self.initial_infected = n_infected
        self.control_delay = control_delay
        self.initial_condition = [n_total - n_infected,
                                  n_infected,
                                  n_recovered,
                                  n_vaccinated,
                                  n_vaccinated]
        self.type = model_type
        self.prediction_target = prediction_target
        self.n_peaks = n_peaks

    def _set_correct_flow(self):
        if self.type == SirModelType.Controlled:
            self._flow = self._sir_controlled_flow
        elif self.type == SirModelType.Free:
            self._flow = self._sir_flow

    def _set_correct_target(self):
        if self.prediction_target == PredictionTarget.ActiveCases:
            self._target_variable = self._compute_active_cases
        elif self.prediction_target == PredictionTarget.NewCases:
            self._target_variable = self._compute_new_cases

    # endregion

    # region Flow models:
    def _sir_flow(
            self,
            z: tuple,
            time: Union[float, np.array],
            beta: float,
            alpha: float,
            q: Union[float, np.array] = 0
    ):
        (s, i, r, v, s_v) = z
        vaccination_rate = self.vaccination_params.get_vaccines_per_day(
            time=time, remaining=s
        )

        vaccinated = vaccination_rate * self.total_population
        effectiveness = self.vaccination_params.effectiveness

        def susceptible_flow(x: float):
            return (q - 1) * beta * x * i / self.total_population

        ds = susceptible_flow(s) - vaccinated
        ds_v = susceptible_flow(s_v) + (1 - effectiveness) * vaccinated
        di = (1 - q) * beta * (s + s_v) * i / self.total_population - alpha * i
        dr = alpha * i
        dv = effectiveness * vaccinated
        return np.array([ds, di, dr, dv, ds_v])

    def _sir_controlled_flow(
            self,
            z: tuple,
            time: Union[float, np.array],
            beta: float,
            alpha: float,
            measure_maturity: Union[float, list],
            measure_scale: Union[float, list]
    ):
        q = control(time - self.control_delay,
                    self.measure_init,
                    measure_maturity, measure_scale,
                    self.delay_time, self.inertial_time)
        (ds, di, dr, dv, ds_v) = self._sir_flow(z, time, beta, alpha, q)
        return np.array([ds, di, dr, dv, ds_v])

    # endregion

    # region Simulation:
    def _simulate(
            self,
            time: Union[float, np.array],
            beta: float,
            alpha: float,
            maturities: Union[float, list, tuple] = None,
            scales: Union[float, list, tuple] = None
    ):

        arguments = (beta, alpha)
        if self.type == SirModelType.Controlled:
            arguments = (
                beta,
                alpha,
                maturities,
                scales
            )

        z = odeint(
            self._flow,
            self.initial_condition,
            time,
            args=arguments
        )

        return z
    # endregion

    # region Fit:
    def _compute_active_cases(self, z: np.array):
        return z[:, 1]

    def _compute_new_cases(self, z: np.array):
        return np.concatenate(
            [np.array([self.initial_infected]), np.diff(z[:, 1] + z[:, 2])]
        )

    def _compute_trajectory(
            self,
            time: Union[float, np.array],
            beta: float,
            alpha: float,
            *argv: tuple
    ):
        maturities = flatten_list(argv[0: self.n_peaks])
        scales = flatten_list(argv[self.n_peaks: self.n_peaks+2*self.n_peaks])

        z = self._simulate(
            time,
            beta,
            alpha,
            maturities,
            scales
        )

        return self._target_variable(z)

    def fit(self, x_data, y_data):
        p0 = flatten_list(self.initial_guess)
        p_opt, p_cov = curve_fit(
            f=self._compute_trajectory,
            xdata=x_data,
            ydata=y_data,
            p0=p0,
            bounds=self.bounds
        )

        p_err = np.sqrt(np.diag(p_cov))

        idx_off = 2
        if self.type == SirModelType.Free:
            self.beta, self.alpha = p_opt
            self.beta_err, self.alpha_err = p_err[0: idx_off]

        elif self.type == SirModelType.Controlled:
            self.beta, self.alpha = p_opt[0: idx_off]
            self.beta_err, self.alpha_err = p_err[0: idx_off]

            self.maturities = p_opt[idx_off:idx_off + self.n_peaks].tolist()
            self.maturities_err = p_err[idx_off:idx_off + self.n_peaks].tolist()

            self.scales = p_opt[idx_off + self.n_peaks: idx_off + 2 * self.n_peaks].tolist()
            self.scales_err = p_err[idx_off + self.n_peaks: idx_off + 2 * self.n_peaks].tolist()

        return [p_opt, p_cov]

    # endregion

    # region Prediction
    def compute_trajectory(
            self,
            time: Union[float, np.array],
    ):
        z = self._simulate(
            time=time,
            beta=self.beta,
            alpha=self.alpha,
            maturities=self.maturities,
            scales=self.scales
        )

        s = z[:, 0]
        i = z[:, 1]
        r = z[:, 2]
        v = z[:, 3]
        v_s = z[:, 4]

        print(len(s))
        print(range(len(s)))
        print(r[len(s)-1])
        [
            print(
                f't: {tt:.1f}; '
                f'p: {ss + ii + rr + vv + vv_ss:.0f}/{self.total_population} '
                f'| '
                f's: {ss:.1f}, '
                f'i: {ii:.1f}, '
                f'r: {rr:.1f}, '
                f'v: {vv:.1f}, '
                f'v_s:{vv_ss:.1f}'
            )
            for tt, ss, ii, rr, vv, vv_ss in zip(time, s, i, r, v, v_s)
        ]

        return {
            'new_cases': self._target_variable(z),
            'immunised': z[:, 3],
            'non_immunised': z[:, 4]
        }

    def simulate_sir(self, x_data):
        return \
            self._simulate(x_data, self.beta, self.alpha,
                           self.maturities, self.scales)

    # endregion

    # region Plot

    @staticmethod
    def _add_actual_data(ax, x_data, y_data):
        ax.plot(x_data, y_data, 'o', color='blue', alpha=0.5)

    def _add_prediction(self, ax, x_data):
        ax.plot(
            x_data,
            self._compute_trajectory(x_data, self.beta, self.alpha,
                                     self.maturities, self.scales),
            '-', color='darkturquoise',
            linewidth=1
        )

    def _compute_error_from_trajectory(self, s, i, s_v):
        return np.sqrt(
            np.square(s * i * self.beta_err / self.total_population)
            + np.square(s_v * i * self.beta_err / self.total_population)
            + 2 * np.square(i * self.alpha_err)
        )

    def _get_confidence_intervals(self, dt):
        z = self._simulate(
            time=dt,
            beta=self.beta,
            alpha=self.alpha,
            maturities=self.maturities,
            scales=self.scales
        )

        sigma = self._compute_error_from_trajectory(s=z[:, 0], i=z[:, 1], s_v=z[:, 4])
        return [sigma, -sigma]

    def add_confidence_intervals(self, dt, z, ax, time_offset=None, limits=None):
        conf_interval = self._get_confidence_intervals(dt)

        t0 = time_offset if time_offset else 0
        dt = dt if not limits else dt[limits[0]:limits[1]]
        z = z if not limits else z[limits[0]:limits[1]]
        err_max = conf_interval[0] \
            if not limits else conf_interval[0][limits[0]:limits[1]]
        err_min = conf_interval[1] \
            if not limits else conf_interval[1][limits[0]:limits[1]]

        ax.fill_between(
            t0 + dt, z+err_min, z+err_max, alpha=0.15, color='g'
        )

    def add_control(self, ax, x_data):
        if not (not self.measure_init):
            policy = control(
                t=x_data,
                t_init=self.measure_init,
                maturity=self.maturities,
                effectiveness=self.scales,
                t_delay=self.delay_time,
                t_inertia=self.inertial_time
            )
            ax_r = ax.twinx()
            ax_r.plot(x_data, policy, 'k-', alpha=0.25, linewidth=0)
            ax_r.fill_between(x_data, policy, alpha=0.05, color='k')
            ax_r.set_ylim([0, 1])

    def plot(self, x_data, y_data, x_lims=None, y_lims=None, ax=None, fig=None):
        if (not ax) or (not fig):
            fig, ax = plt.subplots()
        self._add_actual_data(ax, x_data, y_data)
        self._add_prediction(ax, x_data)
        self.add_control(ax, x_data)
        fig.tight_layout()
        if not (not x_lims):
            ax.set_xlim(x_lims)
        if not (not y_lims):
            ax.set_ylim(y_lims)

        return fig, ax

    # endregion

# endregion
