import asyncio
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from core.orchestrator import AgenticOrchestrator
from interfaces.chat_interface import ChatInterface
from interfaces.ghost_overlay import GhostOverlay

# Setup logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOGS_PATH / 'orchestrator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")

    # Check Ollama availability
    from llm.ollama_client import OllamaClient
    llm_client = OllamaClient()

    if not llm_client.is_model_available():
        logger.error("Ollama model %s not available!", settings.OLLAMA_MODEL)
        logger.info("Please install with: ollama pull %s", settings.OLLAMA_MODEL)
        return False

    logger.info("[OK] Ollama model %s is available", settings.OLLAMA_MODEL)

    # Check browser driver
    try:
        from controllers.browser_controller import BrowserController
        browser = BrowserController()
        if browser.driver is None:
            logger.error("Browser driver not available!")
            logger.info("Please install Chrome or Firefox browser")
            return False
        browser.close()
        logger.info("[OK] Browser driver is available")
    except Exception as e:
        logger.error("Browser setup failed: %s", e)
        return False

    return True

async def main():
    """Main application entry point"""
    logger.info("Starting Agentic Orchestrator...")

    # Check prerequisites
    if not await check_prerequisites():
        logger.error("Prerequisites check failed. Please fix the issues above.")
        return

    # Initialize orchestrator
    try:
        orchestrator = AgenticOrchestrator()
        logger.info("Orchestrator initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize orchestrator: %s", e)
        return

    # Initialize chat interface
    try:
        chat_interface = ChatInterface(orchestrator)
        logger.info("Chat interface initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize chat interface: %s", e)
        return

    # Initialize GHOST overlay
    try:
        ghost_overlay = GhostOverlay(orchestrator)
        logger.info("👻 GHOST overlay initialized successfully")
        logger.info("Say 'GHOST' to activate voice mode from anywhere!")
    except Exception as e:
        logger.error("Failed to initialize GHOST overlay: %s", e)
        # Continue without GHOST if it fails
        ghost_overlay = None

    # Start the server
    try:
        logger.info("Starting server on %s:%s", settings.HOST, settings.PORT)
        logger.info("Access the chat interface at: http://%s:%s", settings.HOST, settings.PORT)
        logger.info("API documentation at: http://%s:%s/docs", settings.HOST, settings.PORT)

        config = uvicorn.Config(
            chat_interface.app,
            host=settings.HOST,
            port=settings.PORT,
            log_level="info" if settings.DEBUG else "warning"
        )
        server = uvicorn.Server(config)
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Server error: %s", e)
    finally:
        # Cleanup
        try:
            if ghost_overlay:
                ghost_overlay.close()
            orchestrator.close()
            logger.info("Cleanup completed")
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
