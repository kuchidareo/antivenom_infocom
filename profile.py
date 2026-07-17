"""
Homogeneous bare-metal nodes for Antivenom perf experiments.

Instructions:
Wait until /local/antivenom/READY exists on every node.
The repository is available at /local/repository.
"""

import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()

pc.defineParameter(
    "node_count",
    "Number of physical nodes",
    portal.ParameterType.INTEGER,
    1,
)

pc.defineParameter(
    "hardware_type",
    "CloudLab hardware type",
    portal.ParameterType.NODETYPE,
    "c6525-100g",
)

params = pc.bindParameters()

if params.node_count < 1 or params.node_count > 21:
    pc.reportError(
        portal.ParameterError(
            "node_count must be between 1 and 21",
            ["node_count"],
        )
    )

pc.verifyParameters()

request = pc.makeRequestRSpec()
lan = request.LAN("experiment-lan")

for index in range(params.node_count):
    node = request.RawPC("node{}".format(index))
    node.hardware_type = params.hardware_type

    interface = node.addInterface()
    lan.addInterface(interface)

    node.addService(
        pg.Execute(
            shell="sh",
            command=(
                "sudo /bin/zsh /local/repository/M260717-collaborative-behavioral-modeling/bootstrap.sh"
            ),
        )
    )

pc.printRequestRSpec(request)
