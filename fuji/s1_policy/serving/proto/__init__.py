"""Checked-in gRPC stubs for the FUJI policy service.

Regenerate the stubs from ``fuji_policy.proto`` with
``python -m fuji.s1_policy.serving.compile_protos``. They are packaged so imports
never need to write into the source tree or an installed wheel.
"""

from . import fuji_policy_pb2, fuji_policy_pb2_grpc

__all__ = ["fuji_policy_pb2", "fuji_policy_pb2_grpc"]
