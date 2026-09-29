"""Regenerate the checked-in Python gRPC stubs from ``fuji_policy.proto``.

The absolute ``import fuji_policy_pb2`` that grpc emits into the ``_pb2_grpc``
module is rewritten to a package-relative import so the stubs import cleanly
under ``fuji.s1_policy.serving.proto``. Runtime imports never call this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROTO_NAME = "fuji_policy"
_PROTO_DIR = Path(__file__).resolve().parent / "proto"


def compile_protos(proto_dir: Path | None = None) -> None:
    """Compile ``fuji_policy.proto`` into ``*_pb2.py`` / ``*_pb2_grpc.py``.

    Idempotent and intended for development/CI. Raises ``RuntimeError`` if protoc
    fails. Does nothing (with a warning) if the ``.proto`` is missing.
    """
    proto_dir = proto_dir or _PROTO_DIR
    proto_file = proto_dir / f"{_PROTO_NAME}.proto"
    pb2_grpc_file = proto_dir / f"{_PROTO_NAME}_pb2_grpc.py"

    if not proto_file.exists():
        logger.warning("Proto file not found at %s, skipping compilation", proto_file)
        return

    import grpc_tools
    from grpc_tools import protoc as grpc_protoc

    well_known_protos = str(Path(grpc_tools.__path__[0]) / "_proto")

    exit_code = grpc_protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto_dir}",
            f"-I{well_known_protos}",
            f"--python_out={proto_dir}",
            f"--grpc_python_out={proto_dir}",
            str(proto_file),
        ]
    )
    if exit_code != 0:
        raise RuntimeError(f"protoc failed with exit code {exit_code}")

    # grpc_tools emits ``import fuji_policy_pb2`` (absolute); rewrite it to a
    # package-relative import so the generated grpc module resolves its sibling
    # pb2 module under fuji.s1_policy.serving.proto.
    grpc_text = pb2_grpc_file.read_text()
    if f"from . import {_PROTO_NAME}_pb2" not in grpc_text:
        fixed_text = grpc_text.replace(
            f"import {_PROTO_NAME}_pb2",
            f"from . import {_PROTO_NAME}_pb2",
        )
        if grpc_text != fixed_text:
            pb2_grpc_file.write_text(fixed_text)

    logger.debug("FUJI gRPC proto stubs compiled successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    compile_protos()
    logger.info("Compiled %s stubs into %s", _PROTO_NAME, _PROTO_DIR)
