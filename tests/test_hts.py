from io import StringIO
import textwrap
from unittest import TestCase

from iso8601 import parse_date as d
import numpy as np
import pandas as pd

import hts


tenmin_test_timeseries = textwrap.dedent("""\
            2008-02-07 09:40,10.32,
            2008-02-07 09:50,10.42,
            2008-02-07 10:00,10.51,
            2008-02-07 10:10,10.54,
            2008-02-07 10:20,10.71,
            2008-02-07 10:30,10.96,
            2008-02-07 10:40,10.93,
            2008-02-07 10:50,11.10,
            2008-02-07 11:00,11.23,
            2008-02-07 11:10,11.44,
            2008-02-07 11:20,11.41,
            2008-02-07 11:30,11.42,MISS
            2008-02-07 11:40,11.54,
            2008-02-07 11:50,11.68,
            2008-02-07 12:00,11.80,
            2008-02-07 12:10,11.91,
            2008-02-07 12:20,12.16,
            2008-02-07 12:30,12.16,
            2008-02-07 12:40,12.24,
            2008-02-07 12:50,12.13,
            2008-02-07 13:00,12.17,
            2008-02-07 13:10,12.31,
            """)

tenmin_test_timeseries_file_version_2 = textwrap.dedent("""\
            Version=2\r
            Unit=°C\r
            Count=22\r
            Title=A test 10-min time series\r
            Comment=This timeseries is extremely important\r
            Comment=because the comment that describes it\r
            Comment=spans five lines.\r
            Comment=\r
            Comment=These five lines form two paragraphs.\r
            Timezone=EET (UTC+0200)\r
            Time_step=10,0\r
            Nominal_offset=0,0\r
            Actual_offset=0,0\r
            Variable=temperature\r
            Precision=1\r
            \r
            2008-02-07 09:40,10.3,\r
            2008-02-07 09:50,10.4,\r
            2008-02-07 10:00,10.5,\r
            2008-02-07 10:10,10.5,\r
            2008-02-07 10:20,10.7,\r
            2008-02-07 10:30,11.0,\r
            2008-02-07 10:40,10.9,\r
            2008-02-07 10:50,11.1,\r
            2008-02-07 11:00,11.2,\r
            2008-02-07 11:10,11.4,\r
            2008-02-07 11:20,11.4,\r
            2008-02-07 11:30,11.4,MISS\r
            2008-02-07 11:40,11.5,\r
            2008-02-07 11:50,11.7,\r
            2008-02-07 12:00,11.8,\r
            2008-02-07 12:10,11.9,\r
            2008-02-07 12:20,12.2,\r
            2008-02-07 12:30,12.2,\r
            2008-02-07 12:40,12.2,\r
            2008-02-07 12:50,12.1,\r
            2008-02-07 13:00,12.2,\r
            2008-02-07 13:10,12.3,\r
            """)

tenmin_test_timeseries_file_version_3 = textwrap.dedent("""\
            Unit=°C\r
            Count=22\r
            Title=A test 10-min time series\r
            Comment=This timeseries is extremely important\r
            Comment=because the comment that describes it\r
            Comment=spans five lines.\r
            Comment=\r
            Comment=These five lines form two paragraphs.\r
            Timezone=EET (UTC+0200)\r
            Time_step=10,0\r
            Nominal_offset=0,0\r
            Actual_offset=0,0\r
            Variable=temperature\r
            Precision=1\r
            Location=24.678900 38.123450 4326\r
            Altitude=219.22\r
            \r
            2008-02-07 09:40,10.3,\r
            2008-02-07 09:50,10.4,\r
            2008-02-07 10:00,10.5,\r
            2008-02-07 10:10,10.5,\r
            2008-02-07 10:20,10.7,\r
            2008-02-07 10:30,11.0,\r
            2008-02-07 10:40,10.9,\r
            2008-02-07 10:50,11.1,\r
            2008-02-07 11:00,11.2,\r
            2008-02-07 11:10,11.4,\r
            2008-02-07 11:20,11.4,\r
            2008-02-07 11:30,11.4,MISS\r
            2008-02-07 11:40,11.5,\r
            2008-02-07 11:50,11.7,\r
            2008-02-07 12:00,11.8,\r
            2008-02-07 12:10,11.9,\r
            2008-02-07 12:20,12.2,\r
            2008-02-07 12:30,12.2,\r
            2008-02-07 12:40,12.2,\r
            2008-02-07 12:50,12.1,\r
            2008-02-07 13:00,12.2,\r
            2008-02-07 13:10,12.3,\r
            """)

tenmin_test_timeseries_file_version_4 = textwrap.dedent("""\
            Unit=°C\r
            Count=22\r
            Title=A test 10-min time series\r
            Comment=This timeseries is extremely important\r
            Comment=because the comment that describes it\r
            Comment=spans five lines.\r
            Comment=\r
            Comment=These five lines form two paragraphs.\r
            Timezone=EET (UTC+0200)\r
            Time_step=10,0\r
            Timestamp_rounding=0,0\r
            Timestamp_offset=0,0\r
            Variable=temperature\r
            Precision=1\r
            Location=24.678900 38.123450 4326\r
            Altitude=219.22\r
            \r
            2008-02-07 09:40,10.3,\r
            2008-02-07 09:50,10.4,\r
            2008-02-07 10:00,10.5,\r
            2008-02-07 10:10,10.5,\r
            2008-02-07 10:20,10.7,\r
            2008-02-07 10:30,11.0,\r
            2008-02-07 10:40,10.9,\r
            2008-02-07 10:50,11.1,\r
            2008-02-07 11:00,11.2,\r
            2008-02-07 11:10,11.4,\r
            2008-02-07 11:20,11.4,\r
            2008-02-07 11:30,11.4,MISS\r
            2008-02-07 11:40,11.5,\r
            2008-02-07 11:50,11.7,\r
            2008-02-07 12:00,11.8,\r
            2008-02-07 12:10,11.9,\r
            2008-02-07 12:20,12.2,\r
            2008-02-07 12:30,12.2,\r
            2008-02-07 12:40,12.2,\r
            2008-02-07 12:50,12.1,\r
            2008-02-07 13:00,12.2,\r
            2008-02-07 13:10,12.3,\r
            """)


class HtsTestCase(TestCase):

    def setUp(self):
        self.reference_ts = pd.read_csv(
            StringIO(tenmin_test_timeseries), parse_dates=[0],
            usecols=['date', 'value', 'flags'], index_col=0, header=None,
            names=('date', 'value', 'flags'),
            converters={'flags': lambda x: x}).asfreq('10T')
        self.reference_ts.timestamp_rounding = '0,0'
        self.reference_ts.timestamp_offset = '0,0'
        self.reference_ts.unit = '°C'
        self.reference_ts.title = "A test 10-min time series"
        self.reference_ts.precision = 1
        self.reference_ts.time_step = "10,0"
        self.reference_ts.timezone = "EET (UTC+0200)"
        self.reference_ts.variable = "temperature"
        self.reference_ts.comment = ("This timeseries is extremely important\n"
                                     "because the comment that describes it\n"
                                     "spans five lines.\n\n"
                                     "These five lines form two paragraphs.")
        self.reference_ts.location = {'abscissa': 24.6789,
                                      'ordinate': 38.12345, 'srid': 4326,
                                      'altitude': 219.22, 'asrid': None}

    def test_write_empty(self):
        ts = pd.Series()
        s = StringIO()
        hts.write(ts, s)
        self.assertEqual(s.getvalue(), '')

    def test_write(self):
        data = np.array(
            [[d("2005-08-23 18:53"),         93,   ''],
             [d("2005-08-24 19:52"),        108.7, ''],
             [d("2005-08-25 23:59"),         28.3, 'HEARTS SPADES'],
             [d("2005-08-26 00:02"), float('NaN'), ''],
             [d("2005-08-27 00:02"), float('NaN'), 'DIAMONDS']])
        ts = pd.DataFrame(data[:, [1, 2]], index=data[:, 0],
                          columns=('value', 'flags'))
        s = StringIO()
        hts.write(ts, s)
        self.assertEqual(s.getvalue(), textwrap.dedent("""\
            2005-08-23 18:53,93,\r
            2005-08-24 19:52,108.7,\r
            2005-08-25 23:59,28.3,HEARTS SPADES\r
            2005-08-26 00:02,,\r
            2005-08-27 00:02,,DIAMONDS\r
            """))

    def test_write_file(self):
        outstring = StringIO()
        hts.write_file(self.reference_ts, outstring, version=2)
        self.assertEqual(outstring.getvalue(),
                         tenmin_test_timeseries_file_version_2)

        outstring = StringIO()
        hts.write_file(self.reference_ts, outstring, version=3)
        self.assertEqual(outstring.getvalue(),
                         tenmin_test_timeseries_file_version_3)

        outstring = StringIO()
        hts.write_file(self.reference_ts, outstring, version=4)
        self.assertEqual(outstring.getvalue(),
                         tenmin_test_timeseries_file_version_4)

    def test_read_empty(self):
        s = StringIO()
        apd = hts.read(s)
        self.assertEqual(len(apd), 0)

    def test_read(self):
        s = StringIO(tenmin_test_timeseries)
        s.seek(0)
        apd = hts.read(s)
        self.assertEqual(len(apd), 22)
        np.testing.assert_array_equal(
            apd.index, pd.date_range('2008-02-07 09:40',
                                     periods=22, freq='10T'))
        expected = np.array([
            [10.32, ''],
            [10.42, ''],
            [10.51, ''],
            [10.54, ''],
            [10.71, ''],
            [10.96, ''],
            [10.93, ''],
            [11.10, ''],
            [11.23, ''],
            [11.44, ''],
            [11.41, ''],
            [11.42, 'MISS'],
            [11.54, ''],
            [11.68, ''],
            [11.80, ''],
            [11.91, ''],
            [12.16, ''],
            [12.16, ''],
            [12.24, ''],
            [12.13, ''],
            [12.17, ''],
            [12.31, '']], dtype=object)
        np.testing.assert_allclose(apd.values[:, 0].astype(float),
                                   expected[:, 0].astype(float))
        np.testing.assert_array_equal(apd.values[:, 1], expected[:, 1])

    def test_read_file(self):
        s = StringIO(tenmin_test_timeseries_file_version_4)
        s.seek(0)
        apd = hts.read_file(s)

        # Check metadata
        self.assertEqual(apd.unit, '°C')
        self.assertEqual(apd.title, 'A test 10-min time series')
        self.assertEqual(apd.comment, textwrap.dedent('''\
            This timeseries is extremely important
            because the comment that describes it
            spans five lines.

            These five lines form two paragraphs.'''))
        self.assertEqual(apd.timezone, 'EET (UTC+0200)')
        self.assertEqual(apd.time_step, '10,0')
        self.assertEqual(apd.timestamp_rounding, '0,0')
        self.assertEqual(apd.timestamp_offset, '0,0')
        self.assertEqual(apd.variable, 'temperature')
        self.assertEqual(apd.precision, 1)
        self.assertAlmostEqual(apd.location['abscissa'], 24.678900, places=6)
        self.assertAlmostEqual(apd.location['ordinate'], 38.123450, places=6)
        self.assertEqual(apd.location['srid'], 4326)
        self.assertAlmostEqual(apd.location['altitude'], 219.22, places=2)
        self.assertTrue(apd.location['asrid'] is None)

        # Check time series data
        self.assertEqual(len(apd), 22)
        np.testing.assert_array_equal(
            apd.index, pd.date_range('2008-02-07 09:40',
                                     periods=22, freq='10T'))
        expected = np.array([
            [10.3, ''],
            [10.4, ''],
            [10.5, ''],
            [10.5, ''],
            [10.7, ''],
            [11.0, ''],
            [10.9, ''],
            [11.1, ''],
            [11.2, ''],
            [11.4, ''],
            [11.4, ''],
            [11.4, 'MISS'],
            [11.5, ''],
            [11.7, ''],
            [11.8, ''],
            [11.9, ''],
            [12.2, ''],
            [12.2, ''],
            [12.2, ''],
            [12.1, ''],
            [12.2, ''],
            [12.3, '']], dtype=object)
        np.testing.assert_allclose(apd.values[:, 0].astype(float),
                                   expected[:, 0].astype(float))
        np.testing.assert_array_equal(apd.values[:, 1], expected[:, 1])
