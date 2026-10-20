"""Checked-in gRPC stubs for the FLoRe policy service.

Regenerate the stubs from ``flore_policy.proto`` with
``python -m flore.s1_policy.serving.compile_protos``. They are packaged so imports
never need to write into the source tree or an installed wheel.
"""

from . import flore_policy_pb2, flore_policy_pb2_grpc

__all__ = ["flore_policy_pb2", "flore_policy_pb2_grpc"]
