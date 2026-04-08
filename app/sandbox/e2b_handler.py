# filepath: app/sandbox/e2b_handler.py

from __future__ import annotations

import asyncio
import atexit
from typing import Optional, Any, Callable

from e2b_code_interpreter import Sandbox

from app.utils.logger import logger
from config.settings import settings


class E2BSandbox:
    """
    Manages a persistent E2B Code Interpreter sandbox session.
    Includes robust retry logic to handle the delay in internal Jupyter port availability.
    """

    def __init__(self, plan_id: str, timeout: int = 3600):
        self.plan_id = plan_id
        self._sandbox: Optional[Sandbox] = None
        self.timeout = timeout
        self._is_ready = False
        self._lock = asyncio.Lock()

        logger.info(f"[{self.plan_id}] Initializing E2BSandbox with {timeout}s timeout.")

    def _create_sandbox_sync(self) -> Sandbox:
        """Synchronous sandbox creation via E2B SDK."""
        template = settings.E2B_TEMPLATE_ID or "base"
        api_key = settings.E2B_API_KEY

        logger.info(f"[{self.plan_id}] Requesting sandbox from E2B (template: {template})...")

        # Dynamically get the create method
        create_fn: Callable[..., Any] = getattr(Sandbox, "create")

        sandbox = create_fn(
            template=template,
            api_key=api_key,
            timeout=self.timeout,
        )

        sandbox_id = getattr(sandbox, "sandbox_id", "unknown")
        logger.info(f"[{self.plan_id}] Sandbox VM created with ID: {sandbox_id}")

        return sandbox

    async def _check_ready(self) -> bool:
        """
        Check if the sandbox's internal Jupyter server is actually listening.
        Retries specifically for 'port not open' / 502 errors.
        """
        if not self._sandbox:
            return False

        max_retries = 10
        for i in range(max_retries):
            try:
                # Attempt a trivial execution to verify the internal port is open
                await asyncio.to_thread(self._sandbox.run_code, "1+1", timeout=5)
                logger.info(f"[{self.plan_id}] ✓ Sandbox kernel is responsive.")
                return True
            except Exception as e:
                error_str = str(e)
                # Check for the specific "port not open" error common at startup
                if "502" in error_str or "port is not open" in error_str:
                    wait_time = 1.5 + (i * 0.5) # Increasing wait time
                    logger.warning(
                        f"[{self.plan_id}] Sandbox port not ready yet (Attempt {i+1}/{max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    # If it's a different error (auth, network, etc.), fail immediately
                    logger.error(f"[{self.plan_id}] Health check encountered terminal error: {e}")
                    return False
        
        logger.error(f"[{self.plan_id}] Sandbox failed to become ready after {max_retries} attempts.")
        return False

    async def start(self) -> bool:
        """Start sandbox and verify readiness."""
        async with self._lock:
            if self._sandbox and self._is_ready:
                logger.info(f"[{self.plan_id}] Sandbox already running and ready.")
                return True

            if self._sandbox:
                self._cleanup_sandbox_internal()

            try:
                logger.info(f"[{self.plan_id}] === Starting E2B Sandbox ===")

                # 1. Create the VM
                loop = asyncio.get_event_loop()
                self._sandbox = await loop.run_in_executor(None, self._create_sandbox_sync)

                if not self._sandbox:
                    raise RuntimeError("Sandbox creation returned None")

                # 2. Wait for internal services to boot (Retry Logic)
                self._is_ready = await self._check_ready()

                if self._is_ready:
                    logger.info(f"[{self.plan_id}] ✓ E2B Sandbox fully initialized and ready!")
                else:
                    raise RuntimeError("Sandbox failed health check after creation.")

                return True

            except Exception as e:
                logger.error(f"[{self.plan_id}] Failed to start sandbox: {e}")
                self._cleanup_sandbox_internal()
                return False

    async def ensure_running(self) -> bool:
        """Ensure sandbox is alive and responsive before a tool call."""
        if not self._sandbox:
            return await self.start()

        if not self._is_ready:
            self._is_ready = await self._check_ready()
            if self._is_ready:
                return True

        try:
            # Quick ping
            await asyncio.to_thread(self._sandbox.run_code, "1+1", timeout=5)
            self._is_ready = True
            return True
        except Exception:
            logger.warning(f"[{self.plan_id}] Sandbox lost responsiveness. Restarting...")
            self._cleanup_sandbox_internal()
            return await self.start()

    def _cleanup_sandbox_internal(self) -> None:
        """Internal cleanup logic."""
        self._is_ready = False

        if self._sandbox:
            try:
                sandbox_id = getattr(self._sandbox, "sandbox_id", "unknown")
                logger.info(f"[{self.plan_id}] Killing sandbox {sandbox_id}...")
                self._sandbox.kill()
            except Exception as e:
                logger.error(f"[{self.plan_id}] Error during sandbox kill: {e}")
            finally:
                self._sandbox = None

    def close(self) -> None:
        """Public method to close the sandbox."""
        self._cleanup_sandbox_internal()
        logger.info(f"[{self.plan_id}] Sandbox closed.")

    @property
    def sandbox(self) -> Optional[Sandbox]:
        return self._sandbox

    @property
    def is_running(self) -> bool:
        return self._sandbox is not None and self._is_ready


class GlobalSandboxManager:
    """Singleton manager for the global server lifecycle."""

    _instance: Optional["GlobalSandboxManager"] = None

    def __new__(cls) -> "GlobalSandboxManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._sandbox_handler: Optional[E2BSandbox] = None
        self._initialized = True
        atexit.register(self.shutdown)

    async def initialize(self, plan_id: str = "global", timeout: int = 3600) -> bool:
        """Initialize the global sandbox."""
        if self._sandbox_handler and self._sandbox_handler.is_running:
            return True

        self._sandbox_handler = E2BSandbox(plan_id=plan_id, timeout=timeout)
        return await self._sandbox_handler.start()

    async def get_sandbox(self) -> Optional[Sandbox]:
        """Get the active sandbox, ensuring it's running."""
        if not self._sandbox_handler:
            return None

        success = await self._sandbox_handler.ensure_running()
        return self._sandbox_handler.sandbox if success else None

    def shutdown(self) -> None:
        """Kill the sandbox on exit."""
        if self._sandbox_handler:
            self._sandbox_handler.close()
            self._sandbox_handler = None

global_sandbox_manager = GlobalSandboxManager()