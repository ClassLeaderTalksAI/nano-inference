"""Create ignored local demo credentials once; never print the key values."""

import os
import secrets
from pathlib import Path

from nano_inference.tenants import TenantConfig

root = Path(__file__).resolve().parents[1]
config = TenantConfig.load(root / "configs/tenants.example.json")
path = root / ".local/tenants.env"
path.parent.mkdir(parents=True, exist_ok=True)
try:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    print(f"Keeping existing credentials: {path}")
else:
    with os.fdopen(descriptor, "w") as stream:
        for tenant in config.tenants:
            stream.write(f"export {tenant.api_key_env}={secrets.token_hex(32)}\n")
    print(f"Created local credentials: {path}")
print("Load with: source .local/tenants.env")
