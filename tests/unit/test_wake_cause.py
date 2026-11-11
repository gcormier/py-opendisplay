"""Wake-cause byte (MsdWakeInfo) decode, config slot, and new-wake tracking."""

import pytest

from opendisplay import WakeCause, WakeTracker, decode_wake_cause, parse_advertisement
from opendisplay.models.config import PowerOption


def _adv(wake_byte: int, *, index: int = 6, reboot_flag: bool = False):
    dynamic = bytearray(11)
    dynamic[index] = wake_byte
    status = 0x02 if reboot_flag else 0x00
    return parse_advertisement(bytes(dynamic) + bytes([124, 0x90, status]))


def _power(reserved: bytes) -> PowerOption:
    data = bytearray(30)
    data[26:30] = reserved
    return PowerOption.from_bytes(bytes(data))


class TestDecode:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0, WakeCause.UNKNOWN),
            (1, WakeCause.POWER_ON),
            (3, WakeCause.RTC),
            (4, WakeCause.TIMER),
            (5, WakeCause.BUTTON),
            (9, WakeCause.BROWNOUT),
            (15, WakeCause.OTHER),
            (12, WakeCause.OTHER),  # reserved value
            (0xF3, WakeCause.RTC),  # upper nibble reserved for future flags
        ],
    )
    def test_decode(self, raw, expected):
        assert decode_wake_cause(raw) is expected

    def test_advertisement_wake_cause(self):
        assert _adv(3).wake_cause(6) is WakeCause.RTC
        assert _adv(3).wake_cause(11) is None
        assert _adv(3).wake_cause(-1) is None

    def test_legacy_advertisement_has_none(self):
        legacy = parse_advertisement(bytes([0x02, 0x36, 0x00, 0x6C, 0x00, 0xC3, 0x01, 0x55, 0x0F, 0x16, 0x4D]))
        assert legacy.wake_cause(0) is None


class TestConfigSlot:
    def test_zero_is_off(self):
        """Configs from before the field existed (reserved = 0) publish nothing."""
        assert _power(b"\x00\x00\x00\x00").published_wake_info_byte_index is None

    def test_one_based(self):
        assert _power(b"\x01\x00\x00\x00").published_wake_info_byte_index == 0
        assert _power(b"\x07\x00\x00\x00").published_wake_info_byte_index == 6
        assert _power(b"\x0b\x00\x00\x00").published_wake_info_byte_index == 10

    def test_out_of_range_is_off(self):
        assert _power(b"\x0c\x00\x00\x00").published_wake_info_byte_index is None
        assert _power(b"\xff\x00\x00\x00").published_wake_info_byte_index is None


class TestWakeTracker:
    def test_first_advertisement_only_primes(self):
        tracker = WakeTracker(6)
        assert tracker.update("AA", _adv(3), timestamp=0.0) is None

    def test_same_window_no_event(self):
        tracker = WakeTracker(6)
        tracker.update("AA", _adv(3), timestamp=0.0)
        assert tracker.update("AA", _adv(3), timestamp=10.0) is None

    def test_repeat_cause_after_gap_is_new_wake(self):
        tracker = WakeTracker(6, gap_seconds=60)
        tracker.update("AA", _adv(3), timestamp=0.0)
        event = tracker.update("AA", _adv(3), timestamp=900.0)
        assert event is not None
        assert event.cause is WakeCause.RTC
        assert event.previous_cause is WakeCause.RTC
        assert event.reason == "gap"

    def test_cause_change_without_gap(self):
        tracker = WakeTracker(6)
        tracker.update("AA", _adv(3), timestamp=0.0)
        event = tracker.update("AA", _adv(1, reboot_flag=True), timestamp=5.0)
        assert event is not None
        assert event.cause is WakeCause.POWER_ON
        assert event.reason == "cause_changed"

    def test_reboot_edge_same_cause(self):
        """An always-on device reset twice in a row: same cause, reboot flag rises."""
        tracker = WakeTracker(6)
        tracker.update("AA", _adv(1, reboot_flag=False), timestamp=0.0)
        event = tracker.update("AA", _adv(1, reboot_flag=True), timestamp=5.0)
        assert event is not None
        assert event.reason == "rebooted"
        # Flag staying high is not another wake.
        assert tracker.update("AA", _adv(1, reboot_flag=True), timestamp=6.0) is None

    def test_per_address_and_reset(self):
        tracker = WakeTracker(6)
        tracker.update("AA", _adv(3), timestamp=0.0)
        assert tracker.update("BB", _adv(4), timestamp=1.0) is None
        tracker.reset("AA")
        assert tracker.update("AA", _adv(1), timestamp=2.0) is None
