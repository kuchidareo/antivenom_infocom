"""
Heterogeneous bare-metal nodes for Antivenom performance experiments.

Default nodes:
  node0: Utah xl170
  node1: Utah c6525-25g

Instructions:
Wait until /local/antivenom/READY exists on every node.

If bootstrap fails, inspect:
  /local/antivenom/FAILED
  /local/antivenom/logs/bootstrap.log
"""

import geni.portal as portal
import geni.rspec.pg as pg


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_NODES = 21

BOOTSTRAP = (
    "/local/repository/"
    "M260717-collaborative-behavioral-modeling/"
    "bootstrap.sh"
)

CATALOG_SNAPSHOT_DATE = "2026-07-19"


# ---------------------------------------------------------------------------
# Embedded CloudLab hardware catalog
#
# The integer values are availability counts from the snapshot date above.
# They are informational only and must not be treated as guarantees.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Portal parameters
# ---------------------------------------------------------------------------

pc = portal.Context()

pc.defineParameter(
    "cluster",
    "Cluster used to validate the selected hardware types",
    portal.ParameterType.STRING,
    "utah",
)

pc.defineParameter(
    "hardware_types",
    (
        "Comma-separated hardware types. "
        "One physical node is created for each entry."
    ),
    portal.ParameterType.STRING,
    "xl170,c6525-25g",
)

params = pc.bindParameters()


# ---------------------------------------------------------------------------
# Normalize parameters
# ---------------------------------------------------------------------------

cluster = params.cluster.strip().lower()

selected_hardware_types = [
    hardware_type.strip()
    for hardware_type in params.hardware_types.split(",")
    if hardware_type.strip()
]


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

if cluster not in HARDWARE_CATALOG:
    pc.reportError(
        portal.ParameterError(
            "Unknown cluster '{}'. Valid clusters: {}".format(
                cluster,
                ", ".join(sorted(HARDWARE_CATALOG.keys())),
            ),
            ["cluster"],
        )
    )


if not selected_hardware_types:
    pc.reportError(
        portal.ParameterError(
            "At least one hardware type must be specified",
            ["hardware_types"],
        )
    )


if len(selected_hardware_types) > MAX_NODES:
    pc.reportError(
        portal.ParameterError(
            "At most {} nodes may be requested".format(MAX_NODES),
            ["hardware_types"],
        )
    )


if cluster in HARDWARE_CATALOG:
    cluster_catalog = HARDWARE_CATALOG[cluster]

    unknown_hardware_types = [
        hardware_type
        for hardware_type in selected_hardware_types
        if hardware_type not in cluster_catalog
    ]

    if unknown_hardware_types:
        pc.reportError(
            portal.ParameterError(
                (
                    "Hardware type(s) not listed for cluster '{}': {}. "
                    "Valid hardware types: {}"
                ).format(
                    cluster,
                    ", ".join(unknown_hardware_types),
                    ", ".join(sorted(cluster_catalog.keys())),
                ),
                ["hardware_types"],
            )
        )


pc.verifyParameters()


# ---------------------------------------------------------------------------
# Request RSpec
# ---------------------------------------------------------------------------

request = pc.makeRequestRSpec()
lan = request.LAN("experiment-lan")


for index, hardware_type in enumerate(selected_hardware_types):
    node_name = "node{}".format(index)
    interface_name = "if{}".format(index)
    private_address = "10.10.1.{}".format(index + 1)

    node = request.RawPC(node_name)
    node.hardware_type = hardware_type

    interface = node.addInterface(interface_name)

    interface.addAddress(
        pg.IPv4Address(
            private_address,
            "255.255.255.0",
        )
    )

    lan.addInterface(interface)

    bootstrap_command = (
        "sudo -n env "
        "ANTIVENOM_CLUSTER='{cluster}' "
        "ANTIVENOM_HARDWARE_TYPE='{hardware_type}' "
        "/bin/bash '{bootstrap}'"
    ).format(
        cluster=cluster,
        hardware_type=hardware_type,
        bootstrap=BOOTSTRAP,
    )

    node.addService(
        pg.Execute(
            shell="sh",
            command=bootstrap_command,
        )
    )


pc.printRequestRSpec(request)
