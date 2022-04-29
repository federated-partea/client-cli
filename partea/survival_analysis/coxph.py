import warnings
from textwrap import dedent

import numpy as np
import pandas as pd


class _BTree:
    """A simple balanced binary order statistic tree to help compute the concordance.

    When computing the concordance, we know all the values the tree will ever contain. That
    condition simplifies this tree a lot. It means that instead of crazy AVL/red-black shenanigans
    we can simply do the following:

    - Store the final tree in flattened form in an array (so node i's children are 2i+1, 2i+2)
    - Additionally, store the current size of each subtree in another array with the same indices
    - To insert a value, just find its index, increment the size of the subtree at that index and
      propagate
    - To get the rank of an element, you add up a bunch of subtree counts
    """

    def __init__(self, values):
        """
        Parameters
        ----------
        values: list
            List of sorted (ascending), unique values that will be inserted.
        """
        self._tree = self._treeify(values)
        self._counts = np.zeros_like(self._tree, dtype=int)

    @staticmethod
    def _treeify(values):
        """Convert the np.ndarray `values` into a complete balanced tree.

        Assumes `values` is sorted ascending. Returns a list `t` of the same length in which t[i] >
        t[2i+1] and t[i] < t[2i+2] for all i."""
        if len(values) == 1:  # this case causes problems later
            return values
        tree = np.empty_like(values)
        # Tree indices work as follows:
        # 0 is the root
        # 2n+1 is the left child of n
        # 2n+2 is the right child of n
        # So we now rearrange `values` into that format...

        # The first step is to remove the bottom row of leaves, which might not be exactly full
        last_full_row = int(np.log2(len(values) + 1) - 1)
        len_ragged_row = len(values) - (2 ** (last_full_row + 1) - 1)
        if len_ragged_row > 0:
            bottom_row_ix = np.s_[: 2 * len_ragged_row: 2]
            tree[-len_ragged_row:] = values[bottom_row_ix]
            values = np.delete(values, bottom_row_ix)

        # Now `values` is length 2**n - 1, so can be packed efficiently into a tree
        # Last row of nodes is indices 0, 2, ..., 2**n - 2
        # Second-last row is indices 1, 5, ..., 2**n - 3
        # nth-last row is indices (2**n - 1)::(2**(n+1))
        values_start = 0
        values_space = 2
        values_len = 2 ** last_full_row
        while values_start < len(values):
            tree[values_len - 1: 2 * values_len - 1] = values[values_start::values_space]
            values_start += int(values_space / 2)
            values_space *= 2
            values_len = int(values_len / 2)
        return tree

    def insert(self, value):
        """Insert an occurrence of `value` into the btree."""
        i = 0
        n = len(self._tree)
        while i < n:
            cur = self._tree[i]
            self._counts[i] += 1
            if value < cur:
                i = 2 * i + 1
            elif value > cur:
                i = 2 * i + 2
            else:
                return
        raise ValueError("Value %s not contained in tree." "Also, the counts are now messed up." % value)

    def __len__(self):
        return self._counts[0]

    def rank(self, value):
        """Returns the rank and count of the value in the btree."""
        i = 0
        n = len(self._tree)
        rank = 0
        count = 0
        while i < n:
            cur = self._tree[i]
            if value < cur:
                i = 2 * i + 1
                continue
            elif value > cur:
                rank += self._counts[i]
                # subtract off the right tree if exists
                nexti = 2 * i + 2
                if nexti < n:
                    rank -= self._counts[nexti]
                    i = nexti
                    continue
                else:
                    return (rank, count)
            else:  # value == cur
                count = self._counts[i]
                lefti = 2 * i + 1
                if lefti < n:
                    nleft = self._counts[lefti]
                    count -= nleft
                    rank += nleft
                    righti = lefti + 1
                    if righti < n:
                        count -= self._counts[righti]
                return (rank, count)
        return (rank, count)


def _concordance_ratio(num_correct: int, num_tied: int, num_pairs: int) -> float:
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs


def _get_index(X):
    # we need a unique index because these are about to become column names.
    if isinstance(X, pd.DataFrame) and X.index.is_unique:
        index = list(X.index)
    elif isinstance(X, pd.DataFrame) and not X.index.is_unique:
        warnings.warn("DataFrame Index is not unique, defaulting to incrementing index instead.")
        index = list(range(X.shape[0]))
    elif isinstance(X, pd.Series):
        return list(X.index)
    else:
        # If it's not a dataframe or index is not unique, order is up to user
        index = list(range(X.shape[0]))
    return index


def _handle_pairs(truth, pred, first_ix, times_to_compare):
    """
    Handle all pairs that exited at the same time as truth[first_ix].

    Returns
    -------
      (pairs, correct, tied, next_ix)
      new_pairs: The number of new comparisons performed
      new_correct: The number of comparisons correctly predicted
      next_ix: The next index that needs to be handled
    """
    next_ix = first_ix
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare.rank(pred[i])
        correct += rank
        tied += count

    return (pairs, correct, tied, next_ix)


def _concordance_summary_statistics(event_times, predicted_event_times,
                                    event_observed):  # pylint: disable=too-many-locals
    """Find the concordance index in n * log(n) time.

    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    # Here's how this works.
    #
    # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    # would be to iterate over the cases in order of their true event time (from least to greatest),
    # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    # cases that we know should be ranked lower than the case we're looking at currently).
    #
    # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    # value less than x"), then the following algorithm is n log n:
    #
    # Sort the times and predictions by time, increasing
    # n_pairs, n_correct := 0
    # pool := {}
    # for each prediction p:
    #     n_pairs += len(pool)
    #     n_correct += rank(pool, p)
    #     add p to pool
    #
    # There are three complications: tied ground truth values, tied predictions, and censored
    # observations.
    #
    # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    # simultaneously at the end.
    #
    # - To handle tied predictions, which should each count for 0.5, we switch to
    #     n_correct += min_rank(pool, p)
    #     n_tied += count(pool, p)
    #
    # - To handle censored observations, we handle each batch of tied, censored observations just
    # after the batch of observations that died at the same time (since those censored observations
    # are comparable all the observations that died at the same time or previously). However, we do
    # NOT add them to the pool at the end, because they are NOT comparable with any observations
    # that leave the study afterward--whether or not those observations get censored.
    if np.logical_not(event_observed).all():
        return (0, 0, 0)

    died_mask = event_observed.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]
    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = _BTree(np.unique(died_pred))
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = _handle_pairs(censored_truth, censored_pred, censored_ix, times_to_compare)
            censored_ix = next_ix
        elif has_more_died and (not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]):
            pairs, correct, tied, next_ix = _handle_pairs(died_truth, died_pred, died_ix, times_to_compare)
            for pred in died_pred[died_ix:next_ix]:
                times_to_compare.insert(pred)
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs
        num_correct += correct
        num_tied += tied

    return (num_correct, num_tied, num_pairs)


def _preprocess_scoring_data(event_times, predicted_scores, event_observed):
    event_times = np.asarray(event_times, dtype=float)
    predicted_scores = np.asarray(predicted_scores, dtype=float)

    # Allow for (n, 1) or (1, n) arrays
    if event_times.ndim == 2 and (event_times.shape[0] == 1 or event_times.shape[1] == 1):
        # Flatten array
        event_times = event_times.ravel()
    # Allow for (n, 1) or (1, n) arrays
    if predicted_scores.ndim == 2 and (predicted_scores.shape[0] == 1 or predicted_scores.shape[1] == 1):
        # Flatten array
        predicted_scores = predicted_scores.ravel()

    if event_times.shape != predicted_scores.shape:
        raise ValueError("Event times and predictions must have the same shape")
    if event_times.ndim != 1:
        raise ValueError("Event times can only be 1-dimensional: (n,)")

    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError("Observed events must be 1-dimensional of same length as event times")

    # check for NaNs
    for a in [event_times, predicted_scores, event_observed]:
        if np.isnan(a).any():
            raise ValueError("NaNs detected in inputs, please correct or drop.")

    return event_times, predicted_scores, event_observed


def concordance_index(event_times, predicted_scores, event_observed=None) -> float:
    event_times, predicted_scores, event_observed = _preprocess_scoring_data(event_times, predicted_scores,
                                                                             event_observed)
    num_correct, num_tied, num_pairs = _concordance_summary_statistics(event_times, predicted_scores, event_observed)

    return _concordance_ratio(num_correct, num_tied, num_pairs)


def normalize(X, mean=None, std=None):
    if mean is None or std is None:
        mean = X.mean(0)
        std = X.std(0)
    return (X - mean) / std


class ConvergenceWarning(RuntimeWarning):
    pass


def check_low_var(df, prescript="", postscript=""):
    def _low_var(df):
        return df.var(0) < 1e-4

    low_var = _low_var(df)
    if low_var.any():
        cols = str(list(df.columns[low_var]))
        warning_text = (
                "%sColumn(s) %s have very low variance. \
    This may harm convergence. 1) Are you using formula's? Did you mean to add '-1' to the end. "
                "2) Try dropping this redundant column before fitting \
    if convergence fails.%s\n"
                % (prescript, cols, postscript)
        )
        warnings.warn(dedent(warning_text), ConvergenceWarning)


def check_for_numeric_dtypes_or_raise(df):
    nonnumeric_cols = [col for (col, dtype) in df.dtypes.iteritems() if
                       dtype.name == "category" or dtype.kind not in "biuf"]
    if len(nonnumeric_cols) > 0:  # pylint: disable=len-as-condition
        raise TypeError(
            "DataFrame contains nonnumeric columns: %s. Try 1) using pandas.get_dummies to convert "
            "the non-numeric column(s) to numerical data, 2) using it in stratification "
            "`strata=`, or 3) dropping the column(s)."
            % nonnumeric_cols
        )


def check_nans_or_infs(df_or_array):
    if isinstance(df_or_array, (pd.Series, pd.DataFrame)):
        return check_nans_or_infs(df_or_array.values)

    if pd.isnull(df_or_array).any():
        raise TypeError("NaNs were detected in the dataset. Try using pd.isnull to find the problematic values.")

    try:
        infs = np.isinf(df_or_array)
    except TypeError:
        warning_text = (
            """Attempting to convert an unexpected datatype '%s' to float. Suggestion: 1) 
            use `lifelines.utils.datetimes_to_durations` to do conversions or 2)
            manually convert to floats/booleans."""
            % df_or_array.dtype
        )
        warnings.warn(warning_text, UserWarning)
        try:
            infs = np.isinf(df_or_array.astype(float))
        except Exception:
            raise TypeError("Wrong dtype '%s'." % df_or_array.dtype)

    if infs.any():
        raise TypeError("Infs were detected in the dataset. Try using np.isinf to find the problematic values.")


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
