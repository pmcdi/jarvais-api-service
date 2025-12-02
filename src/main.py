import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .config import settings
from .storage import storage_manager
from .routers import upload, visualization, analyzers, trainers, health, dashboard

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' if settings.production else '%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting setup (only for production)
if settings.production:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info(f"Starting Jarvais Highcharts Service in {'production' if settings.production else 'development'} mode")
    logger.info(f"Storage backend: {storage_manager.get_storage_type()}")
    
    # Ensure upload directory exists
    os.makedirs(settings.upload_folder, exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Jarvais Highcharts Service")


# Initialize FastAPI app
app = FastAPI(
    title=settings.title,
    description=settings.description,
    version=settings.version,
    lifespan=lifespan
)

# Add rate limiting (production only)
if settings.production and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware (production only)
if settings.production and settings.trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts
    )

# Helper function to apply rate limiting conditionally
def apply_rate_limit(rate_limit_str: str):
    """Apply rate limiting decorator conditionally."""
    def decorator(func):
        if settings.production and limiter:
            return limiter.limit(rate_limit_str)(func)
        return func
    return decorator

# Include routers
app.include_router(upload.router)
app.include_router(visualization.router)
app.include_router(analyzers.router)
app.include_router(trainers.router)
app.include_router(dashboard.router)
app.include_router(health.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level
    )