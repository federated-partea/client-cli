import numpy as np
import pandas as pd
from lifelines.utils import check_nans_or_infs, check_for_numeric_dtypes_or_raise, check_low_var, normalize, \
    concordance_index, _get_index


def preprocess(file_path, separator, duration_col, event_col):
    if separator == "spss":
        df = pd.read_sas(file_path, encoding='iso-8859-1').sort_values(by=duration_col)
    else:
        sep = ","
        if separator == "comma":
            sep = ","
        elif separator == "tab":
            sep = "\t"
        elif separator == "space":
            sep = " "

        df = pd.read_csv(file_path, sep=sep).sort_values(by=duration_col).dropna()

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = df.select_dtypes(include=numerics).dropna()

    if duration_col not in df.columns or event_col not in df.columns:
        raise Exception("Duration Column or Event Column not found in the dataset.")

    return df


class CoxPHModel:

    def __init__(self, df, duration_col, event_col):
        """
        :param df: dataframe of the dataset
        :param duration_col: column name in df which describes the duration
        :param event_col: column name in df which describes the event occured

        This method will call the preprocess method, will save all needed information
        and will normalize the values of the covariates.
        """
        if duration_col is None:
            raise TypeError("duration_col cannot be None.")
        self.data = df
        self.n_samples = df.shape[0]
        self.duration_col = duration_col
        self.event_col = event_col
        self.timeline = None
        self.params_ = None
        self.smpc = False

        X, T, E, original_index = self._preprocess_dataframe()

        self.durations = T.copy()
        self.event_observed = E.copy()

        # calculates the mean of the covariates
        self._norm_mean = X.mean(0)
        self._norm_std = None
        # calculates the standard deviation of the covariates

        self.X = X
        self.T = T
        self.E = E

        # just initialize X_norm, X_norm will be overwritten when doing the global_normalization
        self.X_norm = X

    def get_mean(self):
        """

        :return: This method will send the mean and standard deviation to the server.
        """
        return self._norm_mean

    def get_std(self, global_mean):
        """

        :return: This method will send the mean and standard deviation to the server.
        """
        return np.sum(np.power(self.X - global_mean, 2))

    def set_mean(self, mean):
        """

        :return: This method will send the mean and standard deviation to the server.
        """
        self._norm_mean = mean

    def set_std(self, std):
        """

        :return: This method will send the mean and standard deviation to the server.
        """
        self._norm_std = std

    def normalize_local_data(self):
        """

        :param mean: global mean from server
        :param std: global standard deviation from server

        This method will normalize the dataset of the client using the global mean and the global standard deviation.
        """
        self.X_norm = pd.DataFrame(
            normalize(self.X.values, self._norm_mean.values, self._norm_std.values), index=self.X.index,
            columns=self.X.columns
        )

    def _preprocess_dataframe(self):
        """

        :return: X,T,E,original_index
        X : dataframe of the covariates
        T : series of the duration column
        E : series of the event column
        original_index : series of the row names before preprocessing

        This method will preprocess the dataframe and checks the values in df.
        """
        df = self.data.copy()
        sort_by = [self.duration_col, self.event_col] if self.event_col else [self.duration_col]
        df = df.sort_values(by=sort_by)
        original_index = df.index.copy()

        # extract time and event
        # pop will delete that column named duration_col from df
        T = df.pop(self.duration_col)
        E = (
            df.pop(self.event_col)
            if (self.event_col is not None)
            else ()
        )
        # all remaining columns are covariates
        X = df.astype(float)
        T = T.astype(float)

        check_nans_or_infs(E)
        # out of 0 or 1 it will make True or False
        E = E.astype(bool)

        check_low_var(X)
        check_for_numeric_dtypes_or_raise(X)
        check_nans_or_infs(T)
        check_nans_or_infs(X)

        return X, T, E, original_index

    def _local_initialization(self):
        """

        :return: D and aggregated statistics (sum of the covariates)

        This method will do the local initialization for the client.
        It calculates the index subsets D, Di, Ri and the sum of the covariates.
        """
        # sort dataframe by duration_column
        sort_by = [self.duration_col]
        self.data = self.data.sort_values(by=sort_by)

        death_set = {}
        numb_d_set = {}

        # get the distinct event times
        if not self.smpc:
            self.distinct_times = pd.Series(self.data[self.duration_col].unique())
        else:
            self.distinct_times = self.timeline.copy()

        for uniq_time in self.distinct_times:
            Di = self.data[
                (self.data[self.duration_col] == uniq_time) & (self.data[self.event_col] == 1)].index.tolist()
            death_set[uniq_time] = Di
            numb_d_set[str(float(uniq_time))] = int(len(Di))

        self.death_set = death_set
        self.numb_d_set = numb_d_set

        # calculate z-value (zlr) // sum over all distinct_times (sum of the covariates over all individuals in Di))
        zlr = pd.Series()
        for time in self.distinct_times:
            covariates = self.X_norm.loc[self.death_set[time]]
            cov_sum = covariates.sum(axis=0, skipna=True)
            zlr = zlr.add(cov_sum, fill_value=0)

        self.zlr = zlr
        return self.distinct_times, self.zlr, self.numb_d_set, self.X_norm.shape[0]

    def _update_aggregated_statistics_(self, beta):
        """

        :param beta: updated beta from the global server
        :return: 3 aggregated statistics

        This method will calculate the three aggregated statistics with the parameter beta and send them to the server.
        """

        np.set_printoptions(precision=30)

        # test algorithm of lifelines
        i1 = {}
        i2 = {}
        i3 = {}

        X = self.X_norm.values
        T = self.T.values

        n, d = X.shape
        risk_phi = 0
        risk_phi_x = np.zeros((d,))
        risk_phi_x_x = np.zeros((d, d))

        _, counts = np.unique(-T, return_counts=True)
        scores = np.exp(np.dot(X, beta))
        pos = n
        time_index = 0
        for count_of_removal in counts:
            uniq_time = _[time_index]
            slice_ = slice(pos - count_of_removal, pos)
            X_at_t = X[slice_]

            phi_i = scores[slice_, None]
            phi_x_i = phi_i * X_at_t
            phi_x_x_i = np.dot(X_at_t.T, phi_x_i)

            risk_phi = risk_phi + phi_i.sum()
            risk_phi_x = risk_phi_x + phi_x_i.sum(0)
            risk_phi_x_x = risk_phi_x_x + phi_x_x_i
            pos = pos - count_of_removal
            t = str(-uniq_time)
            i1[t] = risk_phi.tolist()
            i2[t] = risk_phi_x.tolist()
            i3[t] = risk_phi_x_x.tolist()

            time_index += 1

        if self.smpc:
            str_distinct_times = [str(float(t)) for t in self.timeline]
            i1 = pd.Series(i1, index=str_distinct_times).fillna(method='ffill').fillna(method='bfill')
            i1.index = i1.index.astype(str)
            i1 = i1.to_dict()

            i2 = pd.Series(i2, index=str_distinct_times).fillna(method='ffill').fillna(method='bfill')
            i2.index = i2.index.astype(str)
            i2 = i2.to_dict()

            i3 = pd.Series(i3, index=str_distinct_times).fillna(method='ffill').fillna(method='bfill')
            i3.index = i3.index.astype(str)
            i3 = i3.to_dict()

        return i1, i2, i3

    def local_concordance_calculation(self):
        try:
            hazards = -self._predict_partial_hazard(self.X, self.params_)
            c_index = float(concordance_index(self.T, hazards, self.E))
        except Exception:
            c_index = None

        return c_index

    def _predict_partial_hazard(self, X, params_) -> pd.Series:
        """

        :return: partial_hazard

        """
        hazard = np.exp(self._predict_log_partial_hazard(X, params_))
        return hazard

    def _predict_log_partial_hazard(self, X, params_) -> pd.Series:
        hazard_names = params_.index

        if isinstance(X, pd.Series) and (
                (X.shape[0] == len(hazard_names) + 2) or (X.shape[0] == len(hazard_names))):
            X = X.to_frame().T
            return self._predict_log_partial_hazard(X, params_)
        elif isinstance(X, pd.Series):
            assert len(hazard_names) == 1, "Series not the correct argument"
            X = X.to_frame().T
            return self._predict_log_partial_hazard(X, params_)

        index = _get_index(X)

        if isinstance(X, pd.DataFrame):
            order = hazard_names
            X = X.reindex(order, axis="columns")
            X = X.astype(float)
            X = X.values

        X = X.astype(float)

        X = normalize(X, self._norm_mean.values, 1)
        log_hazard = pd.Series(np.dot(X, params_), index=index)
        return log_hazard
