import asyncio
import logging
from typing import Sequence

from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack.server.db import async_session
from gpustack.server.worker_reachability import (
    evaluate_worker_reachability,
    fetch_active_control_sessions_by_worker_id,
)
from gpustack.utils.network import is_url_reachable
from gpustack import envs

logger = logging.getLogger(__name__)


class WorkerSyncer:
    """
    WorkerSyncer syncs worker status periodically.
    1. Performs worker reachability checks.
    2. Performs readiness checks based on heartbeats.
    """

    def __init__(self, interval=15, worker_unreachable_timeout=20):
        self._interval = interval
        self._worker_unreachable_timeout = worker_unreachable_timeout

        logger.debug(
            f"WorkerSyncer initialized with unreachable check mode: {envs.WORKER_UNREACHABLE_CHECK_MODE}"
        )

    async def start(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._sync_workers_states()
            except Exception as e:
                logger.error(f"Failed to sync workers: {e}")

    async def _sync_workers_states(self):
        """
        Sync workers' states by checking their reachability and readiness.
        """
        async with async_session() as session:
            all_workers = await Worker.all(session)
            worker_ids = [worker.id for worker in all_workers if worker.id is not None]
            active_control_sessions = await fetch_active_control_sessions_by_worker_id(
                session,
                worker_ids,
            )

        if not all_workers:
            return

        original_worker_states = {
            worker.id: (worker.unreachable, worker.state, worker.state_message)
            for worker in all_workers
        }

        should_check_unreachable = self._should_check_unreachable(len(all_workers))
        tasks = []
        for worker in all_workers:
            if worker.state.is_provisioning:
                continue
            worker_id = worker.id
            active_control_session = (
                active_control_sessions.get(worker_id)
                if worker_id is not None
                else None
            )
            tasks.append(
                self._set_worker_unreachable(
                    worker,
                    active_control_session,
                    should_check_unreachable=should_check_unreachable,
                )
            )
        await asyncio.gather(*tasks)

        state_changed_workers = self.filter_state_change_workers(
            list(all_workers),
            original_worker_states,
        )

        should_update_workers = []
        state_to_worker_name = {
            WorkerStateEnum.NOT_READY: [],
            WorkerStateEnum.UNREACHABLE: [],
            WorkerStateEnum.READY: [],
            WorkerStateEnum.MAINTENANCE: [],
        }
        for worker in state_changed_workers:
            if worker and worker.state in state_to_worker_name:
                should_update_workers.append(worker)
                state_to_worker_name[worker.state].append(worker.name)

        async with async_session() as session:
            for worker in should_update_workers:
                # reload from DB and update states only
                if worker.id is None:
                    continue
                to_update_worker = await Worker.one_by_id(session, worker.id)
                if to_update_worker:
                    to_update_worker.unreachable = worker.unreachable
                    to_update_worker.state = worker.state
                    to_update_worker.state_message = worker.state_message
                    session.add(to_update_worker)
            await session.commit()

        for state, worker_names in state_to_worker_name.items():
            if worker_names:
                logger.info(f"Marked worker {', '.join(worker_names)} as {state}")

    def _should_check_unreachable(self, worker_count: int) -> bool:
        """
        Determine if unreachable check should be performed based on mode and worker count.

        Args:
            worker_count: Total number of workers

        Returns:
            True if unreachable check should be performed, False otherwise
        """
        mode = envs.WORKER_UNREACHABLE_CHECK_MODE
        auto_threshold = 50  # Auto-disable threshold for worker count

        if mode == "disabled":
            return False
        elif mode == "enabled":
            return True
        elif mode == "auto":
            if worker_count > auto_threshold:
                return False
            return True
        else:
            logger.warning(
                f"Invalid WORKER_UNREACHABLE_CHECK_MODE: {mode}, defaulting to 'auto'"
            )
            # Default to auto behavior
            return worker_count <= auto_threshold

    async def _set_worker_unreachable(
        self,
        worker: Worker,
        active_control_session=None,
        should_check_unreachable: bool = True,
    ):
        decision = evaluate_worker_reachability(worker, active_control_session)
        if not decision.reverse_probe_required:
            logger.info(
                "Worker %s reverse probe %s",
                worker.name,
                decision.reverse_probe_reason,
            )
            return

        if not should_check_unreachable:
            logger.info(
                "Worker %s reverse probe skipped: disabled_by_mode (%s)",
                worker.name,
                decision.reverse_probe_reason,
            )
            return

        reachable = await self.is_worker_reachable(worker)
        worker.unreachable = not reachable

        probe_state = "applied" if reachable else "failed"
        if decision.transport_timed_out and decision.transport_message:
            worker.state = WorkerStateEnum.NOT_READY
            worker.state_message = (
                f"{decision.transport_message} Reverse probe {probe_state} for this worker."
            )

        logger.info(
            "Worker %s reverse probe %s (%s)",
            worker.name,
            probe_state,
            decision.reverse_probe_reason,
        )

    @staticmethod
    def filter_state_change_workers(
        workers: Sequence[Worker],
        original_worker_states: dict[int | None, tuple[bool, WorkerStateEnum, str | None]]
        | None = None,
    ) -> list[Worker]:
        """
        Filter workers whose state has changed.

        Args:
            workers: List of workers to check

        Returns:
            List of workers whose state has changed
        """
        state_changed_workers = []
        for worker in workers:
            if original_worker_states is not None:
                (
                    original_worker_unreachable,
                    original_worker_state,
                    original_worker_state_message,
                ) = original_worker_states.get(
                    worker.id,
                    (worker.unreachable, worker.state, worker.state_message),
                )
            else:
                original_worker_unreachable = worker.unreachable
                original_worker_state = worker.state
                original_worker_state_message = worker.state_message

            worker.compute_state()

            if (
                worker.unreachable != original_worker_unreachable
                or worker.state != original_worker_state
                or worker.state_message != original_worker_state_message
            ):
                state_changed_workers.append(worker)
        return state_changed_workers

    async def is_worker_reachable(
        self,
        worker: Worker,
    ) -> bool:
        address = worker.advertise_address or worker.ip
        healthz_url = f"http://{address}:{worker.port}/healthz"
        reachable = await is_url_reachable(
            healthz_url,
            self._worker_unreachable_timeout,
        )
        return reachable
