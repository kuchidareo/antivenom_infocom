"""CloudLab hardware type catalog.

Snapshot supplied by the user on 2026-07-19.
Availability counts are informational only and may change at any time.
"""

HARDWARE_CATALOG = {
    "utah": {
        "c6525-100g": 5,
        "c6525-25g": 31,
        "c6620": 13,
        "d2950": 0,
        "d6515": 5,
        "d750": 0,
        "d760": 0,
        "d760-hbm": 0,
        "d7615": 0,
        "dell-s4048": 0,
        "dl360": 1,
        "m400": 36,
        "m510": 23,
        "mlnx-sn2410": 0,
        "r660-fw": 0,
        "r720": 0,
        "r760-storage": 0,
        "xl170": 57,
    },
    "mass": {
        "build": 0,
        "build-flax0": 0,
        "fc430": 16,
        "fpga-alveo": 0,
        "fpga-alveo-100g": 0,
        "fpga-dl380-u280": 0,
        "fpga-r740-vck5000": 0,
        "fpga-r760-v100-vck5000": 0,
        "fpga-r760-v70": 0,
        "rs440": 0,
        "rs620": 1,
        "rs630": 8,
    },
    "emulab": {
        "d430": 76,
        "d710": 132,
        "d820": 8,
        "dl360": 0,
        "pc3000": 63,
        "sh-sm200": 0,
        "sxmep": 0,
        "vlpru-b48": 1,
        "x410": 2,
    },
    "clemson": {
        "c4130": 0,
        "c6320": 9,
        "c6420": 0,
        "c8220": 20,
        "c8220x": 0,
        "dss7500": 0,
        "ibm8335": 5,
        "nvidiagh": 1,
        "r650": 5,
        "r6525": 13,
        "r6615": 16,
        "r7525": 0,
    },
    "apt": {
        "c6220": 4,
        "r320": 90,
        "r720": 0,
    },
    "wisconsin": {
        "c220g1": 23,
        "c220g2": 24,
        "c220g5": 5,
        "c240g1": 0,
        "c240g2": 0,
        "c240g5": 2,
        "c4130": 1,
        "d7525": 1,
        "d8545": 0,
        "g893": 1,
        "sm110p": 5,
        "sm220u": 0,
    },
}


def hardware_types(cluster):
    """Return all known hardware types for a cluster."""
    return tuple(HARDWARE_CATALOG[cluster])


def available_hardware_types(cluster, minimum_count=1):
    """Return hardware types whose snapshot count meets the threshold."""
    return tuple(
        hardware_type
        for hardware_type, count in HARDWARE_CATALOG[cluster].items()
        if count >= minimum_count
    )
