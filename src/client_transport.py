import time
from typing import Any, Dict, Optional, Tuple

import msgpack
import requests


class EdgeClient:
    """
    Reusable transport client for Edge API.
    Uses persistent HTTP session + msgpack.
    """

    def __init__(self, edge_url: str, timeout_s: float = 20.0):
        self.edge_url = edge_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        url = f"{self.edge_url}/health"
        r = self.session.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def infer_msgpack(
        self,
        sample_id: Any,
        image_u8_bytes: bytes,
        shape_hw3=(32, 32, 3),
        policy: Optional[Dict[str, Any]] = None,
        accept_msgpack: bool = False,
    ) -> Tuple[Dict[str, Any], float, int, str]:
        """
        Sends one inference request via msgpack.
        Returns: (edge_response_dict, client_rtt_ms, status_code, content_type)
        """
        url = f"{self.edge_url}/infer"
        payload: Dict[str, Any] = {
            "sample_id": sample_id,
            "image_u8": image_u8_bytes,       # msgpack "bin"
            "shape": list(shape_hw3),
        }
        if policy is not None:
            payload["policy"] = policy

        body = msgpack.packb(payload, use_bin_type=True)

        headers = {"Content-Type": "application/msgpack"}
        if accept_msgpack:
            headers["Accept"] = "application/msgpack"

        t0 = time.perf_counter()
        r = self.session.post(url, data=body, headers=headers, timeout=self.timeout_s)
        t1 = time.perf_counter()

        client_rtt_ms = (t1 - t0) * 1000.0
        ct = (r.headers.get("Content-Type") or "").lower()

        # Decode response (msgpack or JSON)
        if "application/msgpack" in ct or "application/x-msgpack" in ct:
            resp = msgpack.unpackb(r.content, raw=False)
        else:
            # fallback
            resp = r.json()

        return resp, client_rtt_ms, r.status_code, ct
