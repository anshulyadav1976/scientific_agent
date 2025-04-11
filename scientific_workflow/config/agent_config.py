import os
from portia import Config, StorageClass, LogLevel

def create_agent_config() -> Config:
    """Creates the configuration for the Portia agent."""
    # Use disk storage to persist runs across server restarts
    storage_dir_path = os.path.abspath("./portia_storage") # Store runs in a folder within scientific_workflow
    return Config.from_default(
        storage_class=StorageClass.DISK,
        storage_dir=storage_dir_path,
        default_log_level=LogLevel.INFO # Adjust log level as needed (DEBUG is very verbose)
    ) 