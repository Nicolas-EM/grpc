# Copyright 2026 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression test: _AioCall._maybe_set_client_call_tracer_on_call() (added by
this same observability CL) takes the process-global
grpc._observability._plugin_lock synchronously, on the event loop thread, on
*every* aio RPC -- even when no observability plugin is registered. See
_create_grpc_call() -> _maybe_set_client_call_tracer_on_call() in
src/python/grpcio/grpc/_cython/_cygrpc/aio/call.pyx.pxi.

If another thread holds that same lock while doing blocking work (a metrics
exporter flushing to a collector, for example), the entire asyncio event loop
freezes off-CPU until the lock is released -- not just the coroutine making
the call. Externally this surfaces as a single-worker aio service going
completely unresponsive for tens of seconds and getting killed by a worker
watchdog, invisible to a CPU profiler because the thread isn't spinning, it's
parked in a futex/mutex wait.

This test drives a real grpc.aio server and channel through the compiled
extension (not a simulation) to prove the stall is a genuine property of this
call path, not just of the pure-Python get_plugin() helper in isolation.
"""

import asyncio
import contextlib
import logging
import threading
import time
import unittest

from grpc import _observability

from tests_aio.observability import _test_server
from tests_aio.unit._test_base import AioTestBase

logger = logging.getLogger(__name__)

_LOCK_HELD_SECONDS = 1.0


class ClientCallTracerLockStallTest(AioTestBase):
    async def setUp(self):
        self._server, self._port = await _test_server.start_server()

    async def tearDown(self):
        await self._server.stop(None)

    async def test_exporter_holding_plugin_lock_freezes_event_loop(self):
        """A real unary-unary RPC must not stall while an unrelated thread
        holds _observability._plugin_lock, and the event loop must keep
        servicing other coroutines while the RPC is in flight.

        Both assertions fail against the code introduced by this CL: the RPC
        blocks for the full lock-hold duration, and no other coroutine -- not
        even a plain asyncio.sleep() heartbeat -- gets to run in the
        meantime, because the call-creation path holds the GIL-equivalent
        event loop thread hostage waiting on a threading.RLock.
        """
        exporter_started = threading.Event()

        def exporter_thread():
            # Stands in for a background metrics-exporter thread (e.g. an
            # OTLP HTTP export) that must hold the plugin lock while it does
            # blocking I/O.
            with _observability._plugin_lock:
                exporter_started.set()
                time.sleep(_LOCK_HELD_SECONDS)

        thread = threading.Thread(target=exporter_thread, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5.0)
        self.assertTrue(
            exporter_started.wait(timeout=5.0),
            "exporter thread never acquired the plugin lock",
        )

        heartbeat_ticks = 0

        async def heartbeat():
            nonlocal heartbeat_ticks
            while True:
                await asyncio.sleep(0.01)
                heartbeat_ticks += 1

        heartbeat_task = asyncio.get_event_loop().create_task(heartbeat())

        start = time.monotonic()
        # A real aio unary-unary call through the compiled extension. This
        # triggers _AioCall._create_grpc_call() ->
        # _maybe_set_client_call_tracer_on_call() -> get_plugin(), which
        # blocks on _plugin_lock above even though no plugin is registered.
        await _test_server.unary_unary_call(port=self._port)
        elapsed = time.monotonic() - start

        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

        logger.info(
            "RPC took %.3fs while exporter held the lock for %.3fs; "
            "heartbeat ticked %d times",
            elapsed,
            _LOCK_HELD_SECONDS,
            heartbeat_ticks,
        )

        # The RPC should not have to wait on an unrelated exporter thread at
        # all. Regression: it blocks for ~_LOCK_HELD_SECONDS.
        self.assertLess(
            elapsed,
            _LOCK_HELD_SECONDS * 0.5,
            "RPC creation blocked on the observability plugin lock held by "
            "an unrelated thread -- the event loop stalled instead of "
            "making progress",
        )

        # Even if the RPC itself is allowed to wait, the event loop should
        # keep servicing other coroutines in the meantime. Regression: zero
        # ticks, because the wait happens synchronously on the loop thread.
        self.assertGreater(
            heartbeat_ticks,
            0,
            "event loop produced no heartbeat ticks while the RPC was in "
            "flight -- the whole loop froze, not just the RPC",
        )


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main(verbosity=2)
