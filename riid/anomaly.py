# Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software.
"""This module contains an algorithm for detecting statistical anomalies from
   sequences of gamma spectra.
"""
import logging
from queue import Queue
from typing import Union

import numpy as np
from scipy.stats import poisson, norm


class PoissonNChannelEventDetector():
    """This class identifies sequences of anomalous spectra.

    This algorithm assumes that the background environment will be consistent over known timescales.
    I.e., this algorithm is only intended for static detector, mobile source scenarios.
    """

    def __init__(self, long_term_duration: int = 120, short_term_duration: int = 1,
                 pre_event_duration: int = 5, max_event_duration: float = 120,
                 post_event_duration: int = 1.5, tolerable_false_alarms_per_day: float = 1.0,
                 anomaly_threshold_update_interval: float = 60):
        """
        Args:
            long_term_duration: duration (in seconds) of the long-term buffer;
                this buffer contains background samples only if the correctness of the algorithm
                and environmental assumptions both hold.  A longer duration is often better with
                the trade-off eventually becoming RAM (to store all of the spectra in the buffer)
                and waiting time vs. converging on actually seeing the desired number of tolerable
                false alarms per day.
            short_term_duration: duration (in seconds) of the short-term buffer;
                this buffer can contain background and/or anomalous samples.
                This is the buffer of "most recent" samples that is compared to the long-term buffer
                in order to determine if an anomaly is present.  The basic question asked by this
                algorithm is, "Is what I see in my short-term significantly different than what I've
                seen in my long-term buffer?  If so, it's an anomaly so I'll add it to the event
                buffer.  If not, it's background so I'll add it to the long-term buffer."
                A short-term duration close to the measurement duration is recommended.
            pre_event_duration: duration (in seconds) specifying the amount of pre-event,
                background samples to include in the event result (but not summed into the event
                spectrum) in order to visualize the run-up to the anomaly.
            max_event_duration: maximum duration (in seconds) of the event buffer.
                If anomalies remain present for more than this specified amount of time, the object
                will produce an event so that something is reported.  A new event will likely then
                be started on the subsequent measurement if the anomaly is still present.
            post_event_duration: duration (in seconds) that determines the number of
                consecutive, insignificant measurements (i.e., measurements which are not considered
                anomalies) that must be observed in order to end an event.
            tolerable_false_alarms_per_day: DESIRED maximum number of allowable false
                positive events per day for all spectrum channels.  The actual observed outcome of
                this parameter primarily depends on the long-term duration (although other
                parameters such as the short-term duration and limit update frequency also play a
                part).  Long-term duration influences the number of actual false alarms per day in
                the following ways:
                    (1) if the long-term buffer is not long enough to incorporate long-term
                        background fluctuations, such fluctuations can be interpreted as anomalies;
                        and
                    (2) if the long-term buffer is too short for the desired false alarm rate, it
                        cannot accurately model the anomaly probability distribution
                        (more specifically, to achieve a very low false alarm rate, this algorithm
                        must witness a sufficiently large number of measurements around the tail of
                        the anomaly probability distribution where the false alarm threshold exists,
                        and measurements in that region do not happen very often, hence you must
                        wait a long time to see enough).
                To give an idea of the challenge here, empirical evidence has shown that to
                accurately achieve 1 false alarm per day requires 6 or more hours of long-term
                duration for a 1024-channel, 3"x3" NaI detector at a sample interval of 0.25s.
                The unfortunate reality of this is that you are now dealing with a timescale where
                the background is likely to have a significantly different mean. So what is one to
                do? The simple answer is leave this parameter at its default, and then accept the
                reality of physics that will inevitably lead you to more false positives due to an
                unobtainable background characterization. In other words, aim high but be realistic.
            anomaly_threshold_update_interval: time (in seconds) between updates to the anomaly
                probability thresholds.  Since this calculation is expensive relative to the other
                computations going on, and because background itself usually does not significantly
                between measurements, this parameter can be used to relax computation.

        """
        self._event_in_progress = False

        self._measurement_duration = None
        self._n_measurement_channels = 0
        self._post_event_duration = post_event_duration

        self._anomaly_threshold_update_interval = anomaly_threshold_update_interval
        self._anomaly_threshold_update_counter = 0
        self._pre_event_duration = pre_event_duration

        self._max_event_duration = max_event_duration
        self._tolerable_false_alarms_per_day = tolerable_false_alarms_per_day

        self._short_term_buffer = Queue()
        self._short_term_duration = short_term_duration

        self._long_term_buffer = Queue()
        self._long_term_duration = long_term_duration

        self._event_buffer = Queue()

        self._reset_stats()

    # region Properties

    @property
    def event_in_progress(self):
        """Whether an event is in progress."""
        return self._event_in_progress

    @property
    def short_term_buffer_length(self):
        """Get the number of measurements in the short term buffer (foreground)."""
        return self._short_term_max_count

    @property
    def background_percent_complete(self):
        """Get the percent fullness as a percent (not decimal)."""
        if self._long_term_max_count == 0:
            return 0
        long_term_buffer_percent_full = self._long_term_count / self._long_term_max_count * 100
        return long_term_buffer_percent_full

    @property
    def long_term_sum_norm(self):
        """Get the normalized long term buffer sum (background)."""
        return self._long_term_sum_norm

    @property
    def long_term_duration(self):
        """Get or set the duration (in seconds) of the long term buffer (background)."""
        return self._long_term_duration

    @long_term_duration.setter
    def long_term_duration(self, value):
        f_value = float(value)
        if f_value <= 0:
            raise ValueError("Long term duration must be greater than zero.")
        self._long_term_duration = f_value
        self.clear_background()

    @property
    def short_term_duration(self):
        """Get or set the duration (in seconds) of the short term buffer."""
        return self._short_term_duration

    @short_term_duration.setter
    def short_term_duration(self, value):
        f_value = float(value)
        if f_value <= 0:
            raise ValueError("Short term duration must be greater than zero.")
        self._short_term_duration = f_value

    @property
    def pre_event_duration(self):
        """Get or set the pre-event duration, e.g. the allowable time before an event
        occurs that is considered a part of the event."""
        return self._pre_event_duration

    @pre_event_duration.setter
    def pre_event_duration(self, value):
        f_value = float(value)
        if f_value <= 0:
            raise ValueError("Pre-event duration must be greater than zero.")
        self._pre_event_duration = f_value

    @property
    def max_event_duration(self):
        """Get or set the maximum event duration."""
        return self._max_event_duration

    @max_event_duration.setter
    def max_event_duration(self, value):
        f_value = float(value)
        if f_value <= 0:
            raise ValueError("Max event duration must be greater than zero.")
        self._max_event_duration = f_value

    @property
    def post_event_duration(self):
        """Get or set the post-event duration, e.g. the allowable time after an event
        occurs that is still considered part of the event."""
        return self._post_event_duration

    @post_event_duration.setter
    def post_event_duration(self, value):
        f_value = float(value)
        if f_value < 0:
            raise ValueError("Post event duration must be nonnegative.")
        self._post_event_duration = f_value

    @property
    def tolerable_false_alarms_per_day(self):
        """Get or set the number of tolerable false alarms per day."""
        return self._tolerable_false_alarms_per_day

    @tolerable_false_alarms_per_day.setter
    def tolerable_false_alarms_per_day(self, value):
        f_value = float(value)
        if f_value <= 0:
            raise ValueError("Tolerable false alarms must be greater than zero.")
        self._tolerable_false_alarms_per_day = f_value

    @property
    def limit_update_frequency(self):
        """Get or set the frequency (in Hz) which which to recompute anomaly thresholds.
        """
        return self._limit_update_frequency

    @limit_update_frequency.setter
    def limit_update_frequency(self, value):
        f_value = float(value)
        if f_value < 0:
            raise ValueError("Limit update frequency must be nonnegative.")
        self._limit_update_frequency = f_value

    # endregion

    def _reset_stats(self):
        """Reset the Event Detector stats to a zeroed, initial state."""
        self._measurement_duration = None

        self._anomalous_count = 0
        self._nonanomalous_count_to_trigger_off = 0
        self._nonanomalous_count = 0
        self._anomaly_threshold_update_counter = 0
        self._anomaly_probas = 0
        self._anomaly_proba = None
        self._anomaly_thresholds = None

        self._short_term_buffer_full = False
        self._short_term_count = 0
        self._short_term_max_count = 0
        self._short_term_sum = None

        self._long_term_buffer_full = False
        self._long_term_count = 0
        self._long_term_max_count = 0
        self._long_term_sum = None
        self._long_term_cps = 0
        self._long_term_sum_norm = 0
        self._long_term_proba_buffer = Queue()
        self._long_term_proba_stats = RunningStats(0, 0, 0)

    def clear_background(self):
        """Clear the short-term buffer, long-term buffer, and stats."""
        with self._short_term_buffer.mutex:
            self._short_term_buffer.queue.clear()
        with self._long_term_buffer.mutex:
            self._long_term_buffer.queue.clear()
        self._reset_stats()

    def add_measurement(self, measurement_id: int, measurement: Union[np.array, int],
                        duration: float, verbose=True):
        """Add the next measurement to the event detector.

        Args:
            measurement_id: unique identifier for the measurement
            measurement: 1D array-like object containing one or more channels of measured data
            duration: float representing the length of time over which the measurement was taken
            verbose: whether to print log messages

        Returns:
            if adding a spectrum concludes an event, a tuple is returned containing:
                - the channel-wise sum of all spectra that were part of the event
                - the channel-wise sum of all spectra in the background buffer,
                    normalized to match the duration of the event
                - the duration (in seconds) of the event
                - list of measurement IDs that comprise the event
            else if adding a spectrum does not conclude event, return None
        """
        if self._measurement_duration != duration:
            if verbose:
                logging.debug("Measurement duration changed")
            self.clear_background()
            self._measurement_duration = duration
            self._n_measurement_channels = len(measurement)
            self._short_term_max_count = self._short_term_duration / duration
            self._long_term_max_count = self._long_term_duration / duration
            self._nonanomalous_count_to_trigger_off = self._post_event_duration / duration
            self._sigma_deviation = \
                (self._tolerable_false_alarms_per_day / (24 * 60 * 60 / duration))
            self._long_term_proba_stats.N = self._long_term_max_count

        # Update short-term buffer
        self._short_term_buffer.put((measurement_id, measurement))
        self._short_term_buffer_full = self._short_term_count >= self._short_term_max_count
        if self._short_term_sum is not None:
            self._short_term_sum += np.array(measurement)
        else:
            self._short_term_sum = np.array(measurement)
        if self._short_term_buffer_full:
            _, oldest_short_term_measurement = self._short_term_buffer.get()
            self._short_term_sum -= oldest_short_term_measurement
        else:
            self._short_term_count += 1

        # Anomaly & event determination
        self._long_term_buffer_full = self._long_term_max_count and \
            self._long_term_count >= self._long_term_max_count
        event_result = None
        if self._long_term_buffer_full:
            # Check if it's time to update the anomaly thresholds
            time_to_update_anomaly_thresholds = not self._anomaly_thresholds or \
                self._anomaly_threshold_update_counter >= \
                self._anomaly_threshold_update_interval
            if time_to_update_anomaly_thresholds:
                # It is known that the anomaly probability distribution is more perfectly
                # modeled with a skewed normal distribution, however the number of measurements
                # required to accurately obtain the skew term is infeasible so we use a
                # gaussian for now.
                self._anomaly_thresholds = norm.ppf(
                    self._sigma_deviation,
                    self._long_term_proba_stats.average,
                    self._long_term_proba_stats.stddev
                )
                self._anomaly_threshold_update_counter = 0
            else:
                self._anomaly_threshold_update_counter += self._measurement_duration
            # Check anomaly thresholds
            self._anomaly_probas = poisson.logpmf(
                self._short_term_sum,
                self._long_term_sum_norm
            )
            self._anomaly_proba = np.sum(self._anomaly_probas)
            measurement_is_anomalous = self._anomaly_proba < self._anomaly_thresholds
            # Now determine if signficant measurements meet on_time / off_time requirements
            if measurement_is_anomalous:
                if not self._event_in_progress:
                    if verbose:
                        logging.debug(f"Event started @ {measurement_id}")
                    self._event_in_progress = True
                    self._nonanomalous_count = 0
                # event duration = anomalous count * measurement duration
                self._anomalous_count += 1
                self._event_buffer.put((measurement_id, measurement))
            elif self._event_in_progress:
                self._nonanomalous_count += 1
                anomaly_gone = self._nonanomalous_count >= self._nonanomalous_count_to_trigger_off
                event_duration = self._anomalous_count * self._measurement_duration
                event_reached_max_duration = event_duration >= self.max_event_duration
                event_over = anomaly_gone or event_reached_max_duration
                if event_over:
                    if verbose:
                        logging.debug(f"Event ended @ {measurement_id}")
                    self._event_in_progress = False
                    # Build the event tuple
                    event_measurement_ids, event_measurements = zip(
                        *[self._event_buffer.get() for _ in range(self._event_buffer.qsize())]
                    )
                    event_measurement = np.array(event_measurements).sum(axis=0)
                    self._anomalous_count = 0
                    long_term_cps = self._long_term_sum / self._long_term_duration
                    event_bg_measurement = long_term_cps * event_duration
                    gross_counts = sum(event_measurement)
                    bg_counts = sum(event_bg_measurement)
                    fg_counts = gross_counts - bg_counts
                    snr = fg_counts / bg_counts
                    is_positive_snr_event = snr > 0
                    if not is_positive_snr_event and verbose:
                        logging.debug(f"Event ending @ {measurement_id} had a SNR <= 0 ({snr:.2f})")
                    event_result = (
                        event_measurement,
                        event_bg_measurement,
                        event_duration,
                        event_measurement_ids
                    )
                else:
                    self._event_buffer.put((measurement_id, measurement))

        # Update long-term buffer
        if not self._event_in_progress:
            self._long_term_buffer.put((measurement_id, measurement))
            if self._long_term_sum is not None:
                self._long_term_sum += np.array(measurement)
            else:
                self._long_term_sum = np.array(measurement)

            if self._long_term_buffer_full:
                _, oldest_long_term_measurement = self._long_term_buffer.get()
                self._long_term_sum -= oldest_long_term_measurement
            else:
                self._long_term_count += 1

            self._long_term_cps = np.divide(self._long_term_sum,
                                            (self._long_term_count * self._measurement_duration))
            self._long_term_sum_norm = (self._long_term_cps * self._short_term_duration).clip(
                1 / self._n_measurement_channels
            )
            # Calculate some stats for thresholding
            long_term_anomaly_probas = poisson.logpmf(
                self._short_term_sum,
                self._long_term_sum_norm
            )
            long_term_anomaly_proba = np.sum(long_term_anomaly_probas)
            # Rolling Welford's algorithm
            self._long_term_proba_buffer.put(long_term_anomaly_proba)
            oldest_long_term_proba = 0
            if self._long_term_buffer_full:
                oldest_long_term_proba = self._long_term_proba_buffer.get()
            self._long_term_proba_stats.update(long_term_anomaly_proba, oldest_long_term_proba)

        return event_result


class RunningStats(object):
    """Helper for calculating stats in a rolling fashion."""

    def __init__(self, window_size, average, variance):
        """
        Args:
            window_size: size of the window across which the probabilities are calculated
            average: average within the window
            variance: variance within the window
        """
        self.N = window_size
        self.average = average
        self.variance = variance
        self.stddev = np.sqrt(variance)

    def update(self, new, old):
        """Update the average, variance, and stddev for the RunningStats.

        Args:
            new: new value to include in window
            old: oldest value in window (this must be externally tracked - queue)
        """
        oldavg = self.average
        newavg = oldavg + (new - old) / self.N
        self.average = newavg
        self.variance += (new - old) * (new - newavg + old - oldavg) / (self.N - 1)
        self.stddev = np.sqrt(self.variance)
