import json
from typing import Dict, Any, Optional
from models.base_data import BaseTrackingData

class DataPacker:
    def __init__(self):
        self._frame_data: Dict[str, Any] = {}

    def add(self, data_obj: Optional[BaseTrackingData]) -> None:
        """Accumulate an individual tracking model into the current frame packet."""
        if data_obj and data_obj.is_tracking:
            self._frame_data[data_obj.system_id] = data_obj.to_dict()

    def add_frame_image(self, b64_string: str) -> None:
        """Attach the base64 encoded camera frame to the payload."""
        if b64_string:
            self._frame_data["frame_base64"] = b64_string

    def pack_single(self, data_obj: BaseTrackingData) -> bytes:
        """Directly serialise an individual tracking model to bytes."""
        payload = {data_obj.system_id: data_obj.to_dict()}
        return json.dumps(payload).encode("utf-8")

    def pack_frame(self) -> Optional[bytes]:
        """
        Serialise all accumulated tracking components into a single byte packet
        and reset the buffer for the next frame.
        """
        if not self._frame_data:
            return None
        payload = self._frame_data
        self._frame_data = {}
        return json.dumps(payload).encode("utf-8")

    def clear(self) -> None:
        self._frame_data.clear()