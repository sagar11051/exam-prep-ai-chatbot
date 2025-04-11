import streamlit.watcher.local_sources_watcher as local_sources_watcher
import types

# Define a patched version of extract_paths that skips torch.classes
def safe_extract_paths(module):
    try:
        if isinstance(module.__path__, types.SimpleNamespace):
            return []
        return list(module.__path__)
    except Exception:
        return []

# Replace the original lambda with the safe version
local_sources_watcher.extract_paths = safe_extract_paths
