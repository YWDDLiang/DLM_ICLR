"""Retained crystal DLM implementation; see docs/reproduction.md for the public workflow."""

from collections import deque
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import time


def isolated_results(
    tasks, *, worker_target, worker_arguments, task_timeout=180.0, startup_timeout=120.0, admitted=None
):
    if (
        not worker_arguments
        or not math.isfinite(task_timeout)
        or task_timeout <= 0
        or not math.isfinite(startup_timeout)
        or startup_timeout <= 0
    ):
        raise ValueError("nonempty workers and positive finite process deadlines are required")
    context = mp.get_context("spawn")
    pending = deque(tasks)
    slots, owned_slots = [], []

    def stop(slot):
        if slot.get("stopped"):
            return
        process = slot["process"]
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        slot["connection"].close()
        slot["stopped"] = True

    def start(arguments):
        parent, child = context.Pipe()
        process = context.Process(target=worker_target, args=(child, *arguments))
        process.start()
        child.close()
        slot = {
            "arguments": arguments,
            "process": process,
            "connection": parent,
            "ready": False,
            "active": None,
            "started": time.monotonic(),
        }
        owned_slots.append(slot)
        return slot

    def failure(message):
        return {"status": "worker_error", "error": message, "value": None}

    try:
        for arguments in worker_arguments[: len(pending)]:
            slots.append(start(arguments))
        while pending or any(slot["active"] is not None for slot in slots):
            ready = set(wait([slot["connection"] for slot in slots], timeout=0.1))
            replacements = []
            for index, slot in enumerate(slots):
                restart, completed, has_result = False, None, False
                if slot["connection"] in ready:
                    try:
                        message = slot["connection"].recv()
                        if message.get("ready"):
                            slot["ready"] = True
                        elif "result" in message and slot["active"] is not None:
                            completed, has_result = message["result"], True
                            restart = (
                                isinstance(completed, dict) and completed.get("status") == "worker_error"
                            )
                    except (EOFError, OSError):
                        if slot["active"] is None:
                            raise RuntimeError("scientific worker failed during initialization")
                        completed, has_result = failure("worker exited before its active result"), True
                        restart = True
                if not has_result:
                    elapsed = time.monotonic() - slot["started"]
                    if not slot["ready"] and elapsed > startup_timeout:
                        raise RuntimeError("scientific worker initialization timed out")
                    if slot["active"] is not None and admitted is not None and not admitted(*slot["active"]):
                        completed, has_result = {"status": "cancelled_irrelevant", "value": None}, True
                        restart = True
                    elif slot["active"] is not None and elapsed > task_timeout:
                        completed, has_result = failure(f"active task exceeded {task_timeout:g}s"), True
                        restart = True
                    elif not slot["process"].is_alive():
                        if slot["active"] is None:
                            raise RuntimeError("scientific worker exited while idle")
                        completed, has_result = failure("worker process terminated unexpectedly"), True
                        restart = True
                if has_result:
                    key, payload = slot["active"]
                    slot["active"] = None
                    yield key, payload, completed
                if restart:
                    stop(slot)
                    replacements.append((index, start(slot["arguments"]) if pending else None))
                elif slot["ready"] and slot["active"] is None:
                    while pending:
                        task = pending.popleft()
                        if admitted is None or admitted(*task):
                            slot["active"] = task
                            slot["connection"].send(task[1])
                            slot["started"] = time.monotonic()
                            break
            for index, replacement in reversed(replacements):
                if replacement is None:
                    slots.pop(index)
                else:
                    slots[index] = replacement
    finally:
        for slot in owned_slots:
            stop(slot)
