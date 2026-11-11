"""BLE advertisement data structures."""

from __future__ import annotations

import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from .enums import WakeCause

# SHT40 temperature/humidity readings, bit-packed into 3 bytes of the v1 dynamic
# block. Firmware writes them as a 24-bit little-endian word (21 bits used):
#   v = (rh_deci & 0x3FF) | ((t_deci + 400) << 10)
# See sensor_sht40.cpp write_sht40_msd() / opendisplay_sensor_sht40.c.
SHT40_MSD_LENGTH = 3
SHT40_DEFAULT_MSD_START = 7  # firmware default when msd_data_start_byte is 0 or 0xFF
SHT40_MAX_MSD_START = 8  # a 3-byte block must fit the 11-byte dynamic area
_SHT40_RH_MASK = 0x3FF
_SHT40_TEMP_MASK = 0x7FF
_SHT40_TEMP_SHIFT = 10
_SHT40_TEMP_BIAS = 400  # temperature is stored as (deci-degrees + 400) to avoid a sign bit
_SHT40_MAX_RH_DECI = 1000  # 100.0 %RH
_SHT40_MAX_TEMP_UNITS = 1650  # (125.0 C * 10) + bias, the SHT40's upper limit
_SHT40_UNWRITTEN = 0x000000  # slot the firmware has never written
_SHT40_INVALID = 0xFFFFFF  # firmware's explicit read-failure sentinel

# MsdWakeInfo block (3 bytes at PowerOption.wake_info_msd_slot - 1):
#   [0] bits 0-3 WakeCause, bits 4-7 reserved
#   [1] seconds until sleep (0-254, 0xFF = not sleeping / unknown)
#   [2] planned sleep duration code (see decode_sleep_duration)
# Provisional until opendisplay-protocol settles the encoding.
WAKE_INFO_LENGTH = 3
WAKE_INFO_CAUSE_MASK = 0x0F
_SLEEP_FINE_STEP_S = 15
_SLEEP_FINE_MAX_CODE = 120  # 120 x 15 s = 30 min
_SLEEP_COARSE_STEP_S = 480  # 8 min
_SLEEP_MAX_ENCODED_S = _SLEEP_FINE_MAX_CODE * _SLEEP_FINE_STEP_S + (0xFE - _SLEEP_FINE_MAX_CODE) * _SLEEP_COARSE_STEP_S
# A device unseen for longer than this and then advertising again has slept and
# woken. Battery tags sleep for minutes; continuously advertising devices
# never go this quiet unless they rebooted or dropped off the air.
WAKE_GAP_SECONDS = 60.0


@dataclass
class AdvertisementData:
    """Parsed BLE advertisement manufacturer data.

    Supports both legacy and current firmware advertisement layouts.

    Legacy format (11 bytes, manufacturer ID already stripped by Bleak):
    - [0-6]: Fixed protocol bytes
    - [7-8]: Battery voltage in millivolts (little-endian uint16)
    - [9]: Chip temperature in Celsius (signed int8)
    - [10]: Loop counter (uint8, increments each advertisement)

    Current format (14 bytes, firmware 1.0+):
    - [0-10]: dynamic return data bytes
    - [11]: Temperature encoded as (temp_c + 40) * 2 (0.5C resolution)
    - [12]: Battery voltage (10mV units), low byte
    - [13]: Status byte:
      bit0=battery voltage high bit, bit1=reboot flag, bit2=connection requested,
      bits4-7=loop counter (4-bit)

    Note: Bleak provides manufacturer data as {0x2446: bytes([...])},
    so the 2-byte manufacturer ID is not included in this data.
    This parser also accepts payloads where the manufacturer ID is included
    (13-byte legacy or 16-byte current payload) and strips it automatically.

    Attributes:
        battery_mv: Battery voltage in millivolts
        temperature_c: MCU chip temperature in Celsius. This is the microcontroller's
            own temperature, *not* an attached sensor -- for a board with an SHT40,
            read ambient temperature and humidity via :meth:`sht40_reading` instead.
        loop_counter: Incrementing counter for each advertisement
        format_version: Parsed advertisement format ("legacy" or "v1")
        reboot_flag: Reboot flag from status byte (v1 only)
        connection_requested: Connection-request flag from status byte (v1 only)
        dynamic_data: Dynamic return data block (v1 only)
    """

    battery_mv: int
    temperature_c: float
    loop_counter: int
    format_version: str = "legacy"
    reboot_flag: bool | None = None
    connection_requested: bool | None = None
    dynamic_data: bytes = field(default_factory=bytes)
    raw_data: bytes = field(default_factory=bytes)

    def button_event(self, byte_index: int) -> ButtonEventData | None:
        """Decode one dynamic return byte as button data (v1 only).

        Args:
            byte_index: Index in dynamic return block (0-10)

        Returns:
            Parsed button data, or None if this is not a v1 advertisement.

        Raises:
            IndexError: If byte_index is outside 0-10
        """
        if self.format_version != "v1":
            return None
        if byte_index < 0 or byte_index >= len(self.dynamic_data):
            raise IndexError(f"button byte index out of range: {byte_index}")
        return decode_button_event(self.dynamic_data[byte_index], byte_index)

    def is_pressed(self, byte_index: int) -> bool | None:
        """Return pressed state for one dynamic byte (v1 only)."""
        event = self.button_event(byte_index)
        if event is None:
            return None
        return event.pressed

    @property
    def button_events(self) -> list[ButtonEventData]:
        """Decode all dynamic return bytes as button event data (v1 only).

        Every byte is decoded, including those owned by touch controllers and
        sensors, which produce valid-looking but meaningless button reports.
        Callers should keep only the indices their config assigns to buttons
        (see ``BinaryInputs.published_button_byte_index``).
        """
        if self.format_version != "v1":
            return []
        return [decode_button_event(raw, i) for i, raw in enumerate(self.dynamic_data)]

    def touch_event(self, start_byte: int) -> TouchEventData | None:
        """Parse a 5-byte touch block from dynamic_data at the given offset (v1 only).

        Args:
            start_byte: Offset within the 11-byte dynamic return block (0–6).

        Returns:
            Parsed touch data, or None if not a v1 advertisement or block out of range.
        """
        if self.format_version != "v1":
            return None
        # Valid start offsets are 0-6 within the 11-byte block; a negative value
        # would index from the end and return garbage.
        if not 0 <= start_byte <= 6:
            return None
        if start_byte + 5 > len(self.dynamic_data):
            return None
        byte0 = self.dynamic_data[start_byte]
        x = struct.unpack_from("<H", self.dynamic_data, start_byte + 1)[0]
        y = struct.unpack_from("<H", self.dynamic_data, start_byte + 3)[0]
        return TouchEventData(
            start_byte=start_byte,
            contact_count=byte0 & 0x0F,
            track_id=(byte0 >> 4) & 0x0F,
            x=x,
            y=y,
        )

    def wake_cause(self, byte_index: int) -> WakeCause | None:
        """Decode the wake-cause byte at ``byte_index`` of dynamic_data (v1 only).

        Pass ``PowerOption.published_wake_info_byte_index`` from the device
        config. Returns None for legacy advertisements or an out-of-range index.
        """
        info = self.wake_info(byte_index)
        return None if info is None else info.cause

    def wake_info(self, byte_index: int) -> WakeInfo | None:
        """Decode the 3-byte wake-info block starting at ``byte_index`` (v1 only).

        Returns None for legacy advertisements or when the block does not fit.
        """
        if self.format_version != "v1":
            return None
        if not 0 <= byte_index <= len(self.dynamic_data) - WAKE_INFO_LENGTH:
            return None
        return decode_wake_info(self.dynamic_data[byte_index : byte_index + WAKE_INFO_LENGTH])

    def sht40_reading(self, start_byte: int = SHT40_DEFAULT_MSD_START) -> Sht40Reading | None:
        """Decode a 3-byte SHT40 block from dynamic_data at the given offset (v1 only).

        Args:
            start_byte: Offset within the 11-byte dynamic return block (0-8). Pass
                ``SensorData.sht40_msd_start_byte`` from the device config; the
                default matches the firmware's own default slot.

        Returns:
            Parsed temperature and humidity, or None if this is not a v1
            advertisement, the block does not fit, or the sensor has no valid
            reading (not yet read, or the firmware's read failed).
        """
        if self.format_version != "v1":
            return None
        if not 0 <= start_byte <= SHT40_MAX_MSD_START:
            return None
        if start_byte + SHT40_MSD_LENGTH > len(self.dynamic_data):
            return None
        return decode_sht40(self.dynamic_data[start_byte : start_byte + SHT40_MSD_LENGTH], start_byte)


@dataclass(frozen=True)
class ButtonEventData:
    """Decoded button data stored in one v1 dynamic byte."""

    byte_index: int
    raw: int
    button_id: int
    press_count: int
    pressed: bool


@dataclass(frozen=True)
class ButtonChangeEvent:
    """State transition emitted by AdvertisementTracker."""

    address: str
    byte_index: int
    event_type: str
    button_id: int
    pressed: bool
    press_count: int
    previous_press_count: int
    raw: int
    previous_raw: int
    timestamp: float


@dataclass(frozen=True)
class TouchEventData:
    """Decoded touch data from a 5-byte block in v1 dynamic return data."""

    start_byte: int
    contact_count: int
    track_id: int
    x: int
    y: int

    @property
    def event_type(self) -> str:
        """Return event type string for the current contact state."""
        if 1 <= self.contact_count <= 5:
            return "touch_down"
        if self.contact_count == 6:
            return "touch_up"
        return "touch_idle"

    @property
    def is_touching(self) -> bool:
        """Return True when one or more contacts are active."""
        return 1 <= self.contact_count <= 5


@dataclass(frozen=True)
class TouchChangeEvent:
    """Touch state transition emitted by TouchTracker."""

    address: str
    instance: int
    event_type: str
    x: int
    y: int
    contact_count: int
    track_id: int
    timestamp: float


@dataclass(frozen=True)
class Sht40Reading:
    """Decoded SHT40 measurement from a 3-byte block in v1 dynamic return data."""

    start_byte: int
    temperature_c: float
    humidity_percent: float
    raw: bytes


def decode_button_event(raw: int, byte_index: int) -> ButtonEventData:
    """Decode one dynamic return byte into button fields."""
    return ButtonEventData(
        byte_index=byte_index,
        raw=raw,
        button_id=raw & 0x07,
        press_count=(raw >> 3) & 0x0F,
        pressed=bool((raw >> 7) & 0x01),
    )


def decode_sht40(raw: bytes, start_byte: int = SHT40_DEFAULT_MSD_START) -> Sht40Reading | None:
    """Decode a 3-byte SHT40 block into temperature and humidity.

    Two byte patterns are not measurements and decode to None:

    - ``FF FF FF`` -- the firmware's explicit read-failure sentinel. It decodes to
      164.7 C / 102.3 %RH, which the range checks below reject.
    - ``00 00 00`` -- a slot the firmware has never written. It decodes to exactly
      -40.0 C / 0.0 %RH, the simultaneous floor of *both* ranges, so the range
      checks alone would let it through as a plausible-looking reading. Reporting
      it would write a hard -40 C into consumers' long-term statistics, so it is
      rejected outright; a real reading sitting on both floors at once is not
      physically meaningful.

    Args:
        raw: Exactly 3 bytes, as written by the firmware into the dynamic block.
        start_byte: Offset the block was read from, recorded on the result.

    Returns:
        Parsed reading, or None when the block holds no valid measurement.

    Raises:
        ValueError: If raw is not exactly 3 bytes.
    """
    if len(raw) != SHT40_MSD_LENGTH:
        raise ValueError(f"SHT40 block must be {SHT40_MSD_LENGTH} bytes, got {len(raw)}")

    packed = int.from_bytes(raw, "little")
    if packed in (_SHT40_UNWRITTEN, _SHT40_INVALID):
        return None

    rh_deci = packed & _SHT40_RH_MASK
    temp_units = (packed >> _SHT40_TEMP_SHIFT) & _SHT40_TEMP_MASK
    if rh_deci > _SHT40_MAX_RH_DECI or temp_units > _SHT40_MAX_TEMP_UNITS:
        return None

    return Sht40Reading(
        start_byte=start_byte,
        temperature_c=(temp_units - _SHT40_TEMP_BIAS) / 10.0,
        humidity_percent=rh_deci / 10.0,
        raw=bytes(raw),
    )


def decode_wake_cause(raw: int) -> WakeCause:
    """Decode an MsdWakeInfo cause byte to its WakeCause (reserved values -> OTHER)."""
    try:
        return WakeCause(raw & WAKE_INFO_CAUSE_MASK)
    except ValueError:
        return WakeCause.OTHER


def decode_sleep_duration(code: int) -> int | None:
    """Decode the 1-byte sleep-duration code to seconds (None = no timed sleep).

    0 = no timed sleep after this window; 1-120 = code x 15 s (15 s - 30 min);
    121-254 = 30 min + (code - 120) x 8 min (up to ~18.4 h); 255 = longer than
    encodable. Provisional encoding, see the firmware's
    docs/PROPOSAL_WAKE_REASON_MSD.md.
    """
    if code == 0:
        return None
    if code <= _SLEEP_FINE_MAX_CODE:
        return code * _SLEEP_FINE_STEP_S
    if code < 0xFF:
        return _SLEEP_FINE_MAX_CODE * _SLEEP_FINE_STEP_S + (code - _SLEEP_FINE_MAX_CODE) * _SLEEP_COARSE_STEP_S
    return _SLEEP_MAX_ENCODED_S


def encode_sleep_duration(seconds: int) -> int:
    """Encode seconds as the sleep-duration code, rounding up (inverse of decode)."""
    if seconds <= 0:
        return 0
    fine_max = _SLEEP_FINE_MAX_CODE * _SLEEP_FINE_STEP_S
    if seconds <= fine_max:
        return -(-seconds // _SLEEP_FINE_STEP_S)
    code = _SLEEP_FINE_MAX_CODE + -(-(seconds - fine_max) // _SLEEP_COARSE_STEP_S)
    return min(code, 0xFF)


@dataclass(frozen=True)
class WakeInfo:
    """The 3-byte MsdWakeInfo block.

    Attributes:
        cause: why the device last booted/woke (constant for the wake window).
        seconds_until_sleep: seconds until the device plans to sleep, counted
            from when this advertisement was sent; None = not going to sleep
            (wired, deep sleep disabled) or unknown. 254 means "254 or more".
        sleep_seconds: planned length of the coming sleep, None if none.
    """

    cause: WakeCause
    seconds_until_sleep: int | None
    sleep_seconds: int | None


def decode_wake_info(raw: bytes) -> WakeInfo:
    """Decode a 3-byte MsdWakeInfo block (cause, seconds-until-sleep, sleep code)."""
    if len(raw) < WAKE_INFO_LENGTH:
        raise ValueError(f"wake info needs {WAKE_INFO_LENGTH} bytes, got {len(raw)}")
    return WakeInfo(
        cause=decode_wake_cause(raw[0]),
        seconds_until_sleep=None if raw[1] == 0xFF else raw[1],
        sleep_seconds=decode_sleep_duration(raw[2]),
    )


# Level-1 causes, inferred from the reboot flag when the device does not publish
# a wake-info block: "scheduled" = timer or RTC wake (flag clear at wake),
# "restart" = power-on, reset, crash or any other reboot (flag set).
WAKE_CAUSE_SCHEDULED = "scheduled"
WAKE_CAUSE_RESTART = "restart"


@dataclass
class WakeChangeEvent:
    """A newly observed device wake (see :class:`WakeTracker`).

    ``cause`` is a lower-case WakeCause name ("rtc", "power_on", ...) when the
    device publishes a wake-info block, else "scheduled" / "restart".
    """

    address: str
    cause: str
    previous_cause: str | None
    reason: str  # "gap" | "cause_changed" | "rebooted"
    timestamp: float
    info: WakeInfo | None = None


class WakeTracker:
    """Detect new wakes and keep the current wake cause per device.

    Two levels:

    - ``byte_index`` set (level 2): the cause comes from the device's wake-info
      block at that dynamic-data offset.
    - ``byte_index`` None (level 1, any firmware): the cause is inferred from
      the reboot flag as it reads on the FIRST advertisement of a wake, since
      the firmware clears the flag on the first connection inside the window.
      A mid-window False -> True edge is a restart.

    A new wake is reported when the device is seen again after
    ``gap_seconds`` of silence, when the published cause changes (level 2), or
    on a reboot-flag rise. The first advertisement for an address only primes
    the tracker. Best effort: a wake whose whole window the scanner missed is
    not reported.
    """

    def __init__(self, byte_index: int | None = None, gap_seconds: float = WAKE_GAP_SECONDS) -> None:
        """Track the wake-info block at ``byte_index``, or the reboot flag if None."""
        self._byte_index = byte_index
        self._gap_seconds = gap_seconds
        self._last: dict[str, tuple[str, bool | None, float]] = {}

    @property
    def byte_index(self) -> int | None:
        """The wake-info offset in use, None in reboot-flag (level 1) mode."""
        return self._byte_index

    def current(self, address: str) -> str | None:
        """The cause of the device's current/last wake, None before any advert."""
        last = self._last.get(address)
        return None if last is None else last[0]

    def reset(self, address: str | None = None) -> None:
        """Forget state for one device or all devices."""
        if address is None:
            self._last.clear()
        else:
            self._last.pop(address, None)

    def update(
        self,
        address: str,
        advertisement: AdvertisementData,
        timestamp: float | None = None,
    ) -> WakeChangeEvent | None:
        """Process one advertisement; return the wake it reveals, if any."""
        if advertisement.format_version != "v1":
            return None
        info: WakeInfo | None = None
        if self._byte_index is not None:
            info = advertisement.wake_info(self._byte_index)
            if info is None:
                return None
        flag = advertisement.reboot_flag
        now = timestamp if timestamp is not None else time.time()
        previous = self._last.get(address)

        def flag_cause() -> str:
            return WAKE_CAUSE_RESTART if flag else WAKE_CAUSE_SCHEDULED

        published = None if info is None else info.cause.name.lower()
        if previous is None:
            self._last[address] = (published or flag_cause(), flag, now)
            return None
        prev_cause, prev_flag, prev_seen = previous

        reason: str | None = None
        if now - prev_seen > self._gap_seconds:
            reason = "gap"
            cause = published or flag_cause()
        elif published is not None and published != prev_cause:
            reason = "cause_changed"
            cause = published
        elif flag and prev_flag is False:
            reason = "rebooted"
            cause = published or WAKE_CAUSE_RESTART
        else:
            cause = prev_cause
        self._last[address] = (cause, flag, now)
        if reason is None:
            return None
        return WakeChangeEvent(
            address=address,
            cause=cause,
            previous_cause=prev_cause,
            reason=reason,
            timestamp=now,
            info=info,
        )


class AdvertisementTracker:
    """Track per-device v1 advertisements and emit button transitions.

    This is best-effort only: BLE advertisements can be dropped.

    The 11-byte dynamic block is shared: buttons own the bytes their config
    packets claim, and touch controllers and sensors own the rest. Decoding a
    byte that belongs to something else yields a valid-looking button report,
    so a tracker watching every byte emits phantom transitions whenever a
    touch coordinate or a sensor reading changes.

    Pass ``byte_indices`` to watch only the bytes that are really buttons --
    ``BinaryInputs.published_button_byte_index`` for each configured input.
    Omitting it keeps the historical behaviour of watching all 11 bytes.
    """

    def __init__(self, byte_indices: Iterable[int] | None = None) -> None:
        """Initialize the tracker, optionally restricted to button bytes."""
        self._byte_indices = None if byte_indices is None else frozenset(byte_indices)
        self._last_by_address: dict[str, list[ButtonEventData]] = {}

    def _watched(self, events: list[ButtonEventData]) -> list[ButtonEventData]:
        """Drop dynamic bytes that no configured button reports into."""
        if self._byte_indices is None:
            return events
        return [event for event in events if event.byte_index in self._byte_indices]

    def reset(self, address: str | None = None) -> None:
        """Reset tracker state for one device or all devices."""
        if address is None:
            self._last_by_address.clear()
        else:
            self._last_by_address.pop(address, None)

    def update(
        self,
        address: str,
        advertisement: AdvertisementData,
        timestamp: float | None = None,
    ) -> list[ButtonChangeEvent]:
        """Process one advertisement and return detected transitions."""
        if advertisement.format_version != "v1":
            self._last_by_address.pop(address, None)
            return []

        current = self._watched(advertisement.button_events)
        previous = self._last_by_address.get(address)
        self._last_by_address[address] = current

        if previous is None or len(previous) != len(current):
            return []

        now = timestamp if timestamp is not None else time.time()
        events: list[ButtonChangeEvent] = []

        for prev, curr in zip(previous, current, strict=False):
            if prev.raw == curr.raw:
                continue

            if prev.button_id != curr.button_id:
                events.append(
                    ButtonChangeEvent(
                        address=address,
                        byte_index=curr.byte_index,
                        event_type="button_slot_changed",
                        button_id=curr.button_id,
                        pressed=curr.pressed,
                        press_count=curr.press_count,
                        previous_press_count=prev.press_count,
                        raw=curr.raw,
                        previous_raw=prev.raw,
                        timestamp=now,
                    )
                )
                if not prev.pressed and not curr.pressed:
                    events.append(
                        ButtonChangeEvent(
                            address=address,
                            byte_index=curr.byte_index,
                            event_type="button_down",
                            button_id=curr.button_id,
                            pressed=True,
                            press_count=curr.press_count,
                            previous_press_count=prev.press_count,
                            raw=curr.raw,
                            previous_raw=prev.raw,
                            timestamp=now,
                        )
                    )
                    events.append(
                        ButtonChangeEvent(
                            address=address,
                            byte_index=curr.byte_index,
                            event_type="button_up",
                            button_id=curr.button_id,
                            pressed=False,
                            press_count=curr.press_count,
                            previous_press_count=prev.press_count,
                            raw=curr.raw,
                            previous_raw=prev.raw,
                            timestamp=now,
                        )
                    )
                continue

            missed_short_press = curr.press_count > prev.press_count and not prev.pressed and not curr.pressed
            if missed_short_press:
                events.append(
                    ButtonChangeEvent(
                        address=address,
                        byte_index=curr.byte_index,
                        event_type="button_down",
                        button_id=curr.button_id,
                        pressed=True,
                        press_count=curr.press_count,
                        previous_press_count=prev.press_count,
                        raw=curr.raw,
                        previous_raw=prev.raw,
                        timestamp=now,
                    )
                )
                events.append(
                    ButtonChangeEvent(
                        address=address,
                        byte_index=curr.byte_index,
                        event_type="button_up",
                        button_id=curr.button_id,
                        pressed=False,
                        press_count=curr.press_count,
                        previous_press_count=prev.press_count,
                        raw=curr.raw,
                        previous_raw=prev.raw,
                        timestamp=now,
                    )
                )

            if prev.pressed != curr.pressed:
                events.append(
                    ButtonChangeEvent(
                        address=address,
                        byte_index=curr.byte_index,
                        event_type="button_down" if curr.pressed else "button_up",
                        button_id=curr.button_id,
                        pressed=curr.pressed,
                        press_count=curr.press_count,
                        previous_press_count=prev.press_count,
                        raw=curr.raw,
                        previous_raw=prev.raw,
                        timestamp=now,
                    )
                )

            if prev.press_count != curr.press_count:
                events.append(
                    ButtonChangeEvent(
                        address=address,
                        byte_index=curr.byte_index,
                        event_type="press_count_changed",
                        button_id=curr.button_id,
                        pressed=curr.pressed,
                        press_count=curr.press_count,
                        previous_press_count=prev.press_count,
                        raw=curr.raw,
                        previous_raw=prev.raw,
                        timestamp=now,
                    )
                )

        return events


class TouchTracker:
    """Track per-device v1 touch state and emit touch transitions.

    One instance per touch controller; pass the controller's instance_number
    and touch_data_start_byte from GlobalConfig.touch_controllers[i].
    """

    def __init__(self, instance: int, start_byte: int) -> None:
        """Initialize tracker for one touch controller."""
        self._instance = instance
        self._start_byte = start_byte
        self._last_by_address: dict[str, TouchEventData] = {}

    def reset(self, address: str | None = None) -> None:
        """Reset tracker state for one device or all devices."""
        if address is None:
            self._last_by_address.clear()
        else:
            self._last_by_address.pop(address, None)

    def update(
        self,
        address: str,
        advertisement: AdvertisementData,
        timestamp: float | None = None,
    ) -> list[TouchChangeEvent]:
        """Process one advertisement and return detected touch transitions."""
        current = advertisement.touch_event(self._start_byte)
        if current is None:
            return []

        previous = self._last_by_address.get(address)
        self._last_by_address[address] = current

        if previous is None:
            return []

        if not previous.is_touching and not current.is_touching:
            return []

        now = timestamp if timestamp is not None else time.time()

        if not previous.is_touching and current.is_touching:
            event_type = "touch_down"
        elif previous.is_touching and not current.is_touching:
            event_type = "touch_up"
        elif previous.x != current.x or previous.y != current.y:
            event_type = "touch_move"
        else:
            return []

        return [
            TouchChangeEvent(
                address=address,
                instance=self._instance,
                event_type=event_type,
                x=current.x,
                y=current.y,
                contact_count=current.contact_count,
                track_id=current.track_id,
                timestamp=now,
            )
        ]


LEGACY_LENGTH = 11
V1_LENGTH = 14
MANUFACTURER_ID_LE = b"\x46\x24"
LEGACY_PREFIX = b"\x02\x36\x00\x6c\x00\xc3\x01"


def _strip_manufacturer_id(data: bytes) -> bytes:
    """Strip manufacturer ID prefix if present."""
    if len(data) in (13, 16) and data[:2] == MANUFACTURER_ID_LE:
        return data[2:]
    return data


def _parse_legacy(data: bytes) -> AdvertisementData:
    """Parse legacy 11-byte advertisement data."""
    battery_mv = struct.unpack("<H", data[7:9])[0]  # uint16, little-endian
    temperature_c = float(struct.unpack("b", data[9:10])[0])  # int8, signed
    loop_counter = data[10]  # uint8

    return AdvertisementData(
        battery_mv=battery_mv,
        temperature_c=temperature_c,
        loop_counter=loop_counter,
        format_version="legacy",
        raw_data=data[:LEGACY_LENGTH],
    )


def _parse_v1(data: bytes) -> AdvertisementData:
    """Parse v1 14-byte advertisement data (firmware 1.0+)."""
    dynamic_data = data[0:11]
    temperature_c = (data[11] / 2.0) - 40.0
    battery_10mv = data[12] | ((data[13] & 0x01) << 8)
    battery_mv = battery_10mv * 10
    reboot_flag = bool(data[13] & 0x02)
    connection_requested = bool(data[13] & 0x04)
    loop_counter = (data[13] >> 4) & 0x0F

    return AdvertisementData(
        battery_mv=battery_mv,
        temperature_c=temperature_c,
        loop_counter=loop_counter,
        format_version="v1",
        reboot_flag=reboot_flag,
        connection_requested=connection_requested,
        dynamic_data=dynamic_data,
        raw_data=data[:V1_LENGTH],
    )


def parse_advertisement(data: bytes) -> AdvertisementData:
    """Parse BLE advertisement manufacturer data.

    Note: The manufacturer ID (0x2446) is already stripped by Bleak
    and provided as the dictionary key in advertisement_data.manufacturer_data,
    but this parser also accepts payloads where the manufacturer ID is present.

    Args:
        data: Raw manufacturer data in legacy (11 bytes) or v1 (14 bytes) format.

    Returns:
        AdvertisementData with parsed values

    Raises:
        ValueError: If data is too short or has an unsupported format
    """
    payload = _strip_manufacturer_id(data)

    if len(payload) < LEGACY_LENGTH:
        raise ValueError(
            f"Advertisement data too short: {len(payload)} bytes "
            f"(need {LEGACY_LENGTH} for legacy or {V1_LENGTH} for v1)"
        )

    if len(payload) >= V1_LENGTH:
        return _parse_v1(payload)

    if len(payload) >= LEGACY_LENGTH:
        if payload[:7] != LEGACY_PREFIX:
            raise ValueError(f"Unsupported legacy advertisement signature; expected {LEGACY_PREFIX.hex()} at bytes 0-6")
        return _parse_legacy(payload)

    raise ValueError(f"Unsupported advertisement format ({len(payload)} bytes)")
