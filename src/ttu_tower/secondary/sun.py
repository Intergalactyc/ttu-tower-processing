"""Sun elevation and night/day classification at each slot's center."""
import pandas as pd
from astral import Observer
from astral.sun import elevation as astral_elevation
from astral.sun import sun

from ttu_tower.constants import SITE_ELEVATION, SITE_LATITUDE, SITE_LONGITUDE
from ttu_tower.timegrid import SAMPLES_PER_SLOT

_OBSERVER = Observer(latitude=SITE_LATITUDE, longitude=SITE_LONGITUDE, elevation=SITE_ELEVATION)
_SLOT_HALF_WIDTH = pd.Timedelta(seconds=SAMPLES_PER_SLOT / 50 / 2)


def slot_stats(slots: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """The `slot_stats` table: sun elevation (deg) and night (bool) at each
    slot's center. `slots` needs `slot` and `slot_start` (tz-aware, in
    `timezone`). Sunrise/sunset are cached per UTC calendar date, since they
    don't vary within one.
    """
    centers = slots["slot_start"] + _SLOT_HALF_WIDTH
    utc_dates = centers.dt.tz_convert("UTC").dt.date

    sunrise_by_date: dict = {}
    sunset_by_date: dict = {}
    for date in utc_dates.unique():
        s = sun(_OBSERVER, date=date, tzinfo=timezone)
        sunrise_by_date[date] = s["sunrise"]
        sunset_by_date[date] = s["sunset"]

    sunrise = utc_dates.map(sunrise_by_date)
    sunset = utc_dates.map(sunset_by_date)
    night = ~((sunrise < centers) & (centers < sunset))
    elevation = [astral_elevation(_OBSERVER, dateandtime=c) for c in centers]

    return pd.DataFrame({
        "slot": slots["slot"].to_numpy(), "sun_elevation": elevation, "night": night.to_numpy(),
    })
