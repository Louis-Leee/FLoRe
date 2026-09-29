"""ROS 2 DDS configuration helpers."""

import os
import pathlib

# ---------------------------------------------------------------------------
# CycloneDDS (loopback, for local sim2sim)
# ---------------------------------------------------------------------------

_CYCLONEDDS_XML = pathlib.Path.home() / "cyclonedds.xml"

_CYCLONEDDS_XML_CONTENT = """\
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="https://cdds.io/config
            https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="lo" priority="default" multicast="true" />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
      <EnableMulticastLoopback>true</EnableMulticastLoopback>
    </General>
  </Domain>
</CycloneDDS>
"""


def ensure_cyclonedds() -> None:
    """Set env vars, write the XML config, and assert CycloneDDS is the active RMW."""
    # 1. Ensure XML file exists with correct content
    _CYCLONEDDS_XML.write_text(_CYCLONEDDS_XML_CONTENT)

    # 2. Ensure env vars
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", f"file://{_CYCLONEDDS_XML}")

    # 3. Assert we're actually using cyclonedds in rclpy
    from rclpy.utilities import get_rmw_implementation_identifier  # type: ignore[import-untyped]

    rmw = get_rmw_implementation_identifier()
    if rmw != "rmw_cyclonedds_cpp":
        raise RuntimeError(
            f"Expected rmw_cyclonedds_cpp but got {rmw!r}. Install cmd: sudo apt install ros-jazzy-rmw-cyclonedds-cpp"
        )


# ---------------------------------------------------------------------------
# FastDDS (BXI hardware — UDP-only, no shared memory)
# ---------------------------------------------------------------------------

_FASTDDS_XML = pathlib.Path.home() / "fastdds_no_shm.xml"

_FASTDDS_XML_CONTENT = """\
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
    <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
        <transport_descriptors>
            <transport_descriptor>
                <transport_id>udp_transport</transport_id>
                <type>UDPv4</type>
            </transport_descriptor>
        </transport_descriptors>
        <participant profile_name="participant_profile" is_default_profile="true">
            <rtps>
                <userTransports>
                    <transport_id>udp_transport</transport_id>
                </userTransports>
                <useBuiltinTransports>false</useBuiltinTransports>
            </rtps>
        </participant>
    </profiles>
</dds>
"""


def ensure_fastdds(domain_id: int = 33) -> None:
    """Configure ROS 2 for BXI-compatible FastDDS transport.

    * Sets ``ROS_DOMAIN_ID`` (default 33, required by BXI hardware nodes).
    * Sets ``RMW_IMPLEMENTATION=rmw_fastrtps_cpp``.
    * Writes a FastDDS XML profile that disables shared-memory transport
      (needed because ``hardware_elf3`` runs as root while inference runs
      as the ``bxi`` user — SHM permissions mismatch).
    """
    os.environ.setdefault("ROS_DOMAIN_ID", str(domain_id))
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    _FASTDDS_XML.write_text(_FASTDDS_XML_CONTENT)
    os.environ.setdefault("FASTRTPS_DEFAULT_PROFILES_FILE", str(_FASTDDS_XML))

    from rclpy.utilities import get_rmw_implementation_identifier  # type: ignore[import-untyped]

    rmw = get_rmw_implementation_identifier()
    if rmw != "rmw_fastrtps_cpp":
        raise RuntimeError(
            f"Expected rmw_fastrtps_cpp but got {rmw!r}. Install cmd: sudo apt install ros-jazzy-rmw-fastrtps-cpp"
        )


# ---------------------------------------------------------------------------
# Unified entry point — reads HOLOSOMA_DDS env var
# ---------------------------------------------------------------------------

_DDS_SETUP = {"FAST": ensure_fastdds, "CYCLONE": ensure_cyclonedds}


def ensure_dds_configuration() -> None:
    """Read ``HOLOSOMA_DDS`` env var (``FAST`` or ``CYCLONE``) and configure accordingly.

    Defaults to ``FAST`` if unset.
    """
    dds = os.environ.get("HOLOSOMA_DDS", "CYCLONE").upper()
    if dds not in _DDS_SETUP:
        raise ValueError(f"HOLOSOMA_DDS must be FAST or CYCLONE, got {dds!r}")
    _DDS_SETUP[dds]()
