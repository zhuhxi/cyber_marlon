# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model a toy Capture the flag exercise

See Jupyter notebook toyctf-simulation.ipynb for an example of
game played on this simulation.
"""
from cyberbattle.simulation import model as m
from cyberbattle.simulation.model import NodeID, NodeInfo, VulnerabilityID, VulnerabilityInfo, Identifiers
from typing import Dict, Iterator, cast, Tuple

default_allow_rules = [
    m.FirewallRule("RDP", m.RulePermission.ALLOW),
    m.FirewallRule("SSH", m.RulePermission.ALLOW),
    m.FirewallRule("MySQL", m.RulePermission.ALLOW),
    m.FirewallRule("HTTPS", m.RulePermission.ALLOW),
    m.FirewallRule("HTTP", m.RulePermission.ALLOW)]

# Network nodes involved in the Capture the flag game
nodes = {
    "Website": m.NodeInfo(
        services=[m.ListeningService("HTTPS"),
                  m.ListeningService("MySQL", allowedCredentials=[
                      "ReusedMySqlCred-web"])],
        firewall=m.FirewallConfiguration(incoming=default_allow_rules,
                                         outgoing=default_allow_rules + [
                                             m.FirewallRule("su", m.RulePermission.ALLOW),
                                             m.FirewallRule("sudo", m.RulePermission.ALLOW)]),
        value=100,
        # If can SSH into server then gets FLAG "Shared credentials with
        # database user"
        owned_string="FLAG: Login using insecure SSH user/password",
        vulnerabilities=dict(
            ScanPageContent=m.VulnerabilityInfo(
                description="LeakedGitHubProjectUrl: Website page content shows a link to GitHub "
                            "repo",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.LeakedNodesId(["GitHubProject"]),
                reward_string="WEBSITE page content has a link to github -> Github project discovered!",
                cost=1.0
            ))),


    "GitHubProject": m.NodeInfo(
        services=[m.ListeningService("GIT")],
        value=10,
        vulnerabilities=dict(
            CredScanGitHistory=m.VulnerabilityInfo(
                description="Some secure access token (SAS) leaked in a "
                "reverted git commit",
                type=m.VulnerabilityType.REMOTE,
                precondition=m.Precondition('SasUrlInCommit&GitHub'),
                outcome=m.LeakedCredentials(credentials=[
                    m.CachedCredential(node="AzureStorage",
                                       port="HTTPS",
                                       credential="SASTOKEN1")]),
                rates=m.Rates(probingDetectionRate=0.0,
                              exploitDetectionRate=0.0,
                              successRate=1.0),
                reward_string="CredScan success: Some secure access token (SAS) was leaked in a reverted git commit",
                cost=1.0
            ))),

    "AzureStorage": m.NodeInfo(
        services=[
            m.ListeningService("HTTPS", allowedCredentials=["SASTOKEN1"])],
        value=50,
        vulnerabilities=dict(
            AccessDataWithSASToken=m.VulnerabilityInfo(
                description="Stealing secrets using a publicly shared "
                            "SAS token",
                type=m.VulnerabilityType.REMOTE,
                outcome=m.CustomerData(),
                rates=m.Rates(successRate=1.0),
                reward_string="Stole data using a publicly shared SAS token",
                cost=1.0
            )
        )),


    'client': m.NodeInfo(
        services=[],
        value=0,
        vulnerabilities=dict(
            SearchEdgeHistory=m.VulnerabilityInfo(
                description="Search web history for list of accessed websites",
                type=m.VulnerabilityType.LOCAL,
                outcome=m.LeakedNodesId(["Website"]),
                reward_string="Web browser history revealed website URL of interest",
                cost=1.0
            )),
        agent_installed=True,
        reimagable=False),
}

global_vulnerability_library: Dict[VulnerabilityID, VulnerabilityInfo] = dict([])

# Environment constants
ENV_IDENTIFIERS = m.infer_constants_from_nodes(
    cast(Iterator[Tuple[NodeID, NodeInfo]], list(nodes.items())),
    global_vulnerability_library)

PROPERTIES = ['Linux']

PORTS = ['GIT', 'HTTPS', 'MySQL', 'PING', 'SSH', 'SSH-key', 'su']

LOCAL_VULS = ['SearchEdgeHistory']

REMOTE_VULS = [
    'AccessDataWithSASToken',
    'CredScanGitHistory',
    'ScanPageContent',
]


def generate_env_identifiers(config_list):
    local_vuls_lib_cnt, remote_vuls_lib_cnt, ports_lib_cnt = int(config_list[0]), int(config_list[1]), int(config_list[2])
    assert local_vuls_lib_cnt > 0, 'local vuls lib should be valid'
    assert remote_vuls_lib_cnt > 0, 'remote vuls lib should be valid'
    assert ports_lib_cnt > 0, 'ports lib should be valid'

    local_vuls_lib, remote_vuls_lib, ports_lib = [x for x in LOCAL_VULS], [y for y in REMOTE_VULS], [z for z in PORTS]
    if local_vuls_lib_cnt > len(local_vuls_lib):
        local_vuls_lib = local_vuls_lib + [f'local_vul{i+1}' for i in range(len(local_vuls_lib), local_vuls_lib_cnt)]
    if remote_vuls_lib_cnt > len(remote_vuls_lib):
        remote_vuls_lib = remote_vuls_lib + [f'remote_vul{i+1}' for i in range(len(remote_vuls_lib), remote_vuls_lib_cnt)]
    if ports_lib_cnt > len(ports_lib):
        ports_lib = ports_lib + [f'port{i+1}' for i in range(len(ports_lib), ports_lib_cnt)]

    return Identifiers(properties=PROPERTIES, ports=ports_lib, local_vulnerabilities=local_vuls_lib, remote_vulnerabilities=remote_vuls_lib)


def new_environment(config_list) -> m.Environment:
    return m.Environment(
        network=m.create_network(nodes),
        vulnerability_library=global_vulnerability_library,
        identifiers=generate_env_identifiers(config_list=config_list)
    )
