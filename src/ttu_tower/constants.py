"""Site facts and file-format constants: booms, heights, tilt tables, header
maps, source units.
"""
from ttu_tower.physics import thermo

BOOMS: list[int] = list(range(1, 11))
HEIGHTS: dict[int, float] = {
    1: 0.9, 2: 2.4, 3: 4.0, 4: 10.1, 5: 16.8,
    6: 47.3, 7: 74.7, 8: 116.5, 9: 158.2, 10: 200.0,
}

SITE_LATITUDE = 33.61055
SITE_LONGITUDE = -102.05056
SITE_ELEVATION = 1014.0  # m above sea level

SOURCE_TIMEZONE = "Etc/GMT+6"  # local standard time, no DST; the raw file names' zone

SAMPLE_HZ = 50
ROWS_PER_FILE = 90_000
DETECTION_WINDOW_MIN = 80
STAGE_B_MARGIN_S = 600

# Kelly & Ennis (Sandia report on this tower): residual tilt relative to the
# datalogger's TRANS channels. TILT_ANGLES in degrees, TILT_AXES in degrees.
TILT_ANGLES: dict[int, float] = {
    1: 2, 2: 1, 3: 2, 4: 0, 5: 5,
    6: 3, 7: 5, 8: 3, 9: 4, 10: 1,
}
TILT_AXES: dict[int, float] = {
    1: 120, 2: 90, 3: 90, 4: 10, 5: 175,
    6: 130, 7: 130, 8: 130, 9: 250, 10: 150,
}

# Raw source units, for the converter (io/convert.py).
SOURCE_UNITS = {
    "p": "inHg",
    "t": "F",
    "ts": "F",
    "rh": "%",
    "u": "mph",
    "v": "mph",
    "w": "mph",
    "ws": "mph",
    "wd": ("degrees", "N", "CW"),
    "propu": "mph",
    "propv": "mph",
    "propw": "mph",
    "propws": "mph",
    "propwd": ("degrees", "N", "CW"),
}

# Raw file column headers, two eras. The "old" era has propeller columns for
# booms 3-10; the "new" era has no propeller columns at all.
SOURCE_HEADERS_OLD = [
    "TSU_1", "TSV_1", "TSW_1", "TST_1", "TT_1", "TRH_1", "TBP_1",
    "TSU_2", "TSV_2", "TSW_2", "TST_2", "TT_2", "TRH_2", "TBP_2",
    "TSU_3", "TSV_3", "TSW_3", "TST_3", "TT_3", "TRH_3", "TBP_3", "TPU_3", "TPV_3", "TPW_3",
    "TSU_4", "TSV_4", "TSW_4", "TST_4", "TT_4", "TRH_4", "TBP_4", "TPU_4", "TPV_4", "TPW_4",
    "TSU_5", "TSV_5", "TSW_5", "TST_5", "TT_5", "TRH_5", "TBP_5", "TPU_5", "TPV_5", "TPW_5",
    "TSU_6", "TSV_6", "TSW_6", "TST_6", "TT_6", "TRH_6", "TBP_6", "TPU_6", "TPV_6", "TPW_6",
    "TSU_7", "TSV_7", "TSW_7", "TST_7", "TT_7", "TRH_7", "TBP_7", "TPU_7", "TPV_7", "TPW_7",
    "TSU_8", "TSV_8", "TSW_8", "TST_8", "TT_8", "TRH_8", "TBP_8", "TPU_8", "TPV_8", "TPW_8",
    "TSU_9", "TSV_9", "TSW_9", "TST_9", "TT_9", "TRH_9", "TBP_9", "TPU_9", "TPV_9", "TPW_9",
    "TSU_10", "TSV_10", "TSW_10", "TST_10", "TT_10", "TRH_10", "TBP_10", "TPU_10", "TPV_10", "TPW_10",
    "TSN-TRANS_1", "TSW-TRANS_1", "TSV-TRANS_1", "TS-WS_1", "TS-WD_1", "TS-AWC_1", "TS-XWC_1",
    "TSN-TRANS_2", "TSW-TRANS_2", "TSV-TRANS_2", "TS-WS_2", "TS-WD_2", "TS-AWC_2", "TS-XWC_2",
    "TSN-TRANS_3", "TSW-TRANS_3", "TSV-TRANS_3", "TS-WS_3", "TS-WD_3", "TS-AWC_3", "TS-XWC_3", "TPN-TRANS_3", "TPW-TRANS_3", "TPV-TRANS_3", "TP-WS_3", "TP-WD_3", "TP-AWC_3", "TP-XWC_3",
    "TSN-TRANS_4", "TSW-TRANS_4", "TSV-TRANS_4", "TS-WS_4", "TS-WD_4", "TS-AWC_4", "TS-XWC_4", "TPN-TRANS_4", "TPW-TRANS_4", "TPV-TRANS_4", "TP-WS_4", "TP-WD_4", "TP-AWC_4", "TP-XWC_4",
    "TSN-TRANS_5", "TSW-TRANS_5", "TSV-TRANS_5", "TS-WS_5", "TS-WD_5", "TS-AWC_5", "TS-XWC_5", "TPN-TRANS_5", "TPW-TRANS_5", "TPV-TRANS_5", "TP-WS_5", "TP-WD_5", "TP-AWC_5", "TP-XWC_5",
    "TSN-TRANS_6", "TSW-TRANS_6", "TSV-TRANS_6", "TS-WS_6", "TS-WD_6", "TS-AWC_6", "TS-XWC_6", "TPN-TRANS_6", "TPW-TRANS_6", "TPV-TRANS_6", "TP-WS_6", "TP-WD_6", "TP-AWC_6", "TP-XWC_6",
    "TSN-TRANS_7", "TSW-TRANS_7", "TSV-TRANS_7", "TS-WS_7", "TS-WD_7", "TS-AWC_7", "TS-XWC_7", "TPN-TRANS_7", "TPW-TRANS_7", "TPV-TRANS_7", "TP-WS_7", "TP-WD_7", "TP-AWC_7", "TP-XWC_7",
    "TSN-TRANS_8", "TSW-TRANS_8", "TSV-TRANS_8", "TS-WS_8", "TS-WD_8", "TS-AWC_8", "TS-XWC_8", "TPN-TRANS_8", "TPW-TRANS_8", "TPV-TRANS_8", "TP-WS_8", "TP-WD_8", "TP-AWC_8", "TP-XWC_8",
    "TSN-TRANS_9", "TSW-TRANS_9", "TSV-TRANS_9", "TS-WS_9", "TS-WD_9", "TS-AWC_9", "TS-XWC_9", "TPN-TRANS_9", "TPW-TRANS_9", "TPV-TRANS_9", "TP-WS_9", "TP-WD_9", "TP-AWC_9", "TP-XWC_9",
    "TSN-TRANS_10", "TSW-TRANS_10", "TSV-TRANS_10", "TS-WS_10", "TS-WD_10", "TS-AWC_10", "TS-XWC_10", "TPN-TRANS_10", "TPW-TRANS_10", "TPV-TRANS_10", "TP-WS_10", "TP-WD_10", "TP-AWC_10", "TP-XWC_10",
]
SOURCE_HEADERS_NEW = [
    "TSU_1", "TSV_1", "TSW_1", "TST_1", "TT_1", "TRH_1", "TBP_1",
    "TSU_2", "TSV_2", "TSW_2", "TST_2", "TT_2", "TRH_2", "TBP_2",
    "TSU_3", "TSV_3", "TSW_3", "TST_3", "TT_3", "TRH_3", "TBP_3",
    "TSU_4", "TSV_4", "TSW_4", "TST_4", "TT_4", "TRH_4", "TBP_4",
    "TSU_5", "TSV_5", "TSW_5", "TST_5", "TT_5", "TRH_5", "TBP_5",
    "TSU_6", "TSV_6", "TSW_6", "TST_6", "TT_6", "TRH_6", "TBP_6",
    "TSU_7", "TSV_7", "TSW_7", "TST_7", "TT_7", "TRH_7", "TBP_7",
    "TSU_8", "TSV_8", "TSW_8", "TST_8", "TT_8", "TRH_8", "TBP_8",
    "TSU_9", "TSV_9", "TSW_9", "TST_9", "TT_9", "TRH_9", "TBP_9",
    "TSU_10", "TSV_10", "TSW_10", "TST_10", "TT_10", "TRH_10", "TBP_10",
    "TSN-TRANS_1", "TSW-TRANS_1", "TSV-TRANS_1", "TS-WS_1", "TS-WD_1",
    "TSN-TRANS_2", "TSW-TRANS_2", "TSV-TRANS_2", "TS-WS_2", "TS-WD_2",
    "TSN-TRANS_3", "TSW-TRANS_3", "TSV-TRANS_3", "TS-WS_3", "TS-WD_3",
    "TSN-TRANS_4", "TSW-TRANS_4", "TSV-TRANS_4", "TS-WS_4", "TS-WD_4",
    "TSN-TRANS_5", "TSW-TRANS_5", "TSV-TRANS_5", "TS-WS_5", "TS-WD_5",
    "TSN-TRANS_6", "TSW-TRANS_6", "TSV-TRANS_6", "TS-WS_6", "TS-WD_6",
    "TSN-TRANS_7", "TSW-TRANS_7", "TSV-TRANS_7", "TS-WS_7", "TS-WD_7",
    "TSN-TRANS_8", "TSW-TRANS_8", "TSV-TRANS_8", "TS-WS_8", "TS-WD_8",
    "TSN-TRANS_9", "TSW-TRANS_9", "TSV-TRANS_9", "TS-WS_9", "TS-WD_9",
    "TSN-TRANS_10", "TSW-TRANS_10", "TSV-TRANS_10", "TS-WS_10", "TS-WD_10",
]

# Header prefix -> converted column name; None drops the column. TSN/TSW/TSV
# -TRANS are the datalogger's stored (north, west, up) sonic components: raw
# u/v are the (north, west) components of the wind's FROM-vector (not
# blows-toward) - primary/stage_a.py converts to true (east, north)
# blows-toward.
HEADER_MAP_OLD = {
    "TSU": None, "TSV": None, "TSW": None,
    "TPU": None, "TPV": None, "TPW": None,
    "TST": "ts", "TT": "t", "TRH": "rh", "TBP": "p",
    "TSN-TRANS": "u", "TSW-TRANS": "v", "TSV-TRANS": "w",
    "TS-WS": None, "TS-WD": None, "TS-AWC": None, "TS-XWC": None,
    "TPN-TRANS": "propu", "TPW-TRANS": "propv", "TPV-TRANS": "propw",
    "TP-WS": None, "TP-WD": None, "TP-AWC": None, "TP-XWC": None,
}
HEADER_MAP_NEW = {
    "TSU": None, "TSV": None, "TSW": None,
    "TST": "ts", "TT": "t", "TRH": "rh", "TBP": "p",
    "TSN-TRANS": "u", "TSW-TRANS": "v", "TSV-TRANS": "w",
    "TS-WS": "ws", "TS-WD": None,
}

# Per-boom reference pressure: ISA pressure at site elevation + boom height (kPa).
P_REF: dict[int, float] = {b: thermo.isa_pressure(SITE_ELEVATION + HEIGHTS[b]) for b in BOOMS}
