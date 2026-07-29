"""Generic checkpoint-backed MultiTask DiT WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from .inference_session import InferenceSession
from .protocol import PROTOCOL_VERSION, pack_message, unpack_message


class PolicyServerWrapper:
    """Environment-independent policy inference wrapper."""

    def __init__(
        self,
        *,
        policy: Any,
        stats_tensors: dict[str, dict[str, torch.Tensor]],
        device: str | torch.device,
        default_replan_interval: int | None = None,
    ) -> None:
        self.policy = policy
        self.config = policy.config
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
        self.session = InferenceSession(
            policy=policy,
            stats_tensors=stats_tensors,
            device=self.device,
            default_replan_interval=default_replan_interval,
        )
        image_keys = list(self.config.image_features)
        self.action_dim = int(self.config.action_feature.shape[0])
        self.metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "policy": "multitask_dit_policy",
            "image_keys": image_keys,
            "image_shapes": {key: list(self.config.input_features[key].shape) for key in image_keys},
            "state_key": "observation.state",
            "state_dim": int(self.config.robot_state_feature.shape[0]),
            "action_dim": self.action_dim,
            "n_obs_steps": int(self.config.n_obs_steps),
            "model_action_chunk_size": int(self.config.n_action_steps),
            "horizon": int(self.config.horizon),
            "default_replan_interval": self.session.default_replan_interval,
            "pid": os.getpid(),
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str,
        default_replan_interval: int | None = None,
    ) -> PolicyServerWrapper:
        from multitask_dit_policy.model.model import MultiTaskDiTPolicy

        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        policy = MultiTaskDiTPolicy.load(checkpoint_path)
        stats_path = checkpoint_path / "dataset_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing training statistics: {stats_path}")
        with stats_path.open() as file:
            stats_json = json.load(file)

        device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        stats_tensors = {
            key: {
                name: torch.tensor(value, device=device_obj, dtype=torch.float32)
                for name, value in stat.items()
            }
            for key, stat in stats_json.items()
        }
        return cls(
            policy=policy,
            stats_tensors=stats_tensors,
            device=device_obj,
            default_replan_interval=default_replan_interval,
        )

    def reset(
        self,
        *,
        task: str,
        episode_id: str | None,
        episode_seed: int | None,
        replan_interval: int | None,
    ) -> dict[str, Any]:
        self.session.reset(
            task=task,
            episode_id=episode_id,
            episode_seed=episode_seed,
            replan_interval=replan_interval,
        )
        return {
            "type": "reset_ack",
            "ok": True,
            "task": self.session.task,
            "episode_seed": self.session.episode_seed,
            "replan_interval": self.session.replan_interval,
        }

    def step(self, observation: dict[str, Any], step: int | None = None) -> dict[str, Any]:
        action, inference_metadata = self.session.step(observation, step)
        if action.size != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got shape {action.shape}.")
        return {"type": "action", "action": action, "metadata": inference_metadata}

    def memory_stats(self, *, reset_peak: bool = False) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {"device": str(self.device), "pid": os.getpid()}
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        mib = 1024**2
        return {
            "device": str(self.device),
            "pid": os.getpid(),
            "allocated_mb": torch.cuda.memory_allocated(self.device) / mib,
            "reserved_mb": torch.cuda.memory_reserved(self.device) / mib,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / mib,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / mib,
        }


async def _serve_client(websocket: Any, wrapper: PolicyServerWrapper, lock: asyncio.Lock) -> None:
    if lock.locked():
        await websocket.send(
            pack_message({"error": "ServerBusy", "message": "Only one active policy session is supported."})
        )
        await websocket.close()
        return

    async with lock:
        await websocket.send(pack_message({"type": "handshake", "metadata": wrapper.metadata}))
        async for message in websocket:
            request_id = None
            episode_id = None
            try:
                request = unpack_message(message)
                request_id = request.get("request_id")
                episode_id = request.get("episode_id")
                request_type = request.get("type", "step")
                if request_type == "reset":
                    response = wrapper.reset(
                        task=request.get("task", ""),
                        episode_id=episode_id,
                        episode_seed=request.get("episode_seed"),
                        replan_interval=request.get("replan_interval"),
                    )
                elif request_type == "step":
                    if episode_id != wrapper.session.episode_id:
                        raise ValueError(
                            f"episode_id {episode_id!r} does not match active "
                            f"episode {wrapper.session.episode_id!r}; call reset first"
                        )
                    response = wrapper.step(request["observation"], request.get("step"))
                elif request_type == "health":
                    response = {"type": "health", "ok": True, "metadata": wrapper.metadata}
                elif request_type == "memory":
                    response = {
                        "type": "memory",
                        "memory": wrapper.memory_stats(reset_peak=bool(request.get("reset_peak", False))),
                    }
                else:
                    raise ValueError(f"Unknown request type: {request_type!r}")
                response.update({"request_id": request_id, "episode_id": episode_id})
            except Exception as exc:
                logging.exception("Policy request failed")
                response = {
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "request_id": request_id,
                    "episode_id": episode_id,
                }
            await websocket.send(pack_message(response))


async def run_server(wrapper: PolicyServerWrapper, host: str, port: int) -> None:
    try:
        import websockets
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Policy serving requires websockets. Run `uv sync`.") from exc

    lock = asyncio.Lock()
    async with websockets.serve(
        lambda websocket: _serve_client(websocket, wrapper, lock),
        host,
        port,
        max_size=None,
        compression=None,
    ):
        logging.info("Policy server listening on %s:%s metadata=%s", host, port, wrapper.metadata)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-replan-interval", type=int)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    if args.pid_file is not None:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(f"{os.getpid()}\n")
    try:
        wrapper = PolicyServerWrapper.from_checkpoint(
            args.checkpoint,
            args.device,
            args.default_replan_interval,
        )
        asyncio.run(run_server(wrapper, args.host, args.port))
    finally:
        if args.pid_file is not None:
            args.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
