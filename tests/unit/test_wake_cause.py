"""Wake-info block (cause + sleep timing) decode, config slot, and wake tracking."""

import pytest

from opendisplay import (
    WakeCause,
    WakeTracker,
    decode_sleep_duration,
    decode_wake_cause,
    decode_wake_info,
    encode_sleep_duration,
    parse_advertisement,
)
from opendisplay.models.config import PowerOption

INDEX = 6


def _adv(
    cause: int | None = None,
    *,
    until: int = 0xFF,
    sleep_code: int = 0,
    reboot_flag: bool = False,
):
    dynamic = bytearray(11)
    if cause is not None:
        dynamic[INDEX : INDEX + 3] = bytes([cause, until, sleep_code])
    status = 0x02 if reboot_flag else 0x00
    return parse_advertisement(bytes(dynamic) + bytes([124, 0x90, status]))


def _power(slot: int) -> PowerOption:
    data = bytearray(30)
    data[26] = slot
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
            (0xF3, WakeCause.RTC),  # upper nibble reserved
        ],
    )
    def test_cause(self, raw, expected):
        assert decode_wake_cause(raw) is expected

    @pytest.mark.parametrize(
        ("code", "seconds"),
        [
            (0, None),
            (1, 15),
            (8, 120),
            (60, 900),
            (120, 1800),
            (121, 2280),
            (254, 1800 + 134 * 480),
        ],
    )
    def test_sleep_duration(self, code, seconds):
        assert decode_sleep_duration(code) == seconds

    @pytest.mark.parametrize("seconds", [15, 120, 900, 1800, 3600, 65535])
    def test_sleep_duration_round_trip_rounds_up(self, seconds):
        decoded = decode_sleep_duration(encode_sleep_duration(seconds))
        assert decoded is not None
        assert decoded >= seconds
        assert decoded - seconds < (15 if seconds <= 1800 else 480)

    def test_encode_zero_is_no_sleep(self):
        assert encode_sleep_duration(0) == 0

    def test_block(self):
        info = decode_wake_info(bytes([3, 17, 60]))
        assert info.cause is WakeCause.RTC
        assert info.seconds_until_sleep == 17
        assert info.sleep_seconds == 900

    def test_not_sleeping(self):
        info = decode_wake_info(bytes([1, 0xFF, 0]))
        assert info.seconds_until_sleep is None
        assert info.sleep_seconds is None

    def test_advertisement_block_must_fit(self):
        adv = _adv(3, until=10, sleep_code=8)
        assert adv.wake_info(INDEX).seconds_until_sleep == 10
        assert adv.wake_cause(INDEX) is WakeCause.RTC
        assert adv.wake_info(9) is None  # 9 + 3 > 11
        assert adv.wake_info(-1) is None

    def test_legacy_advertisement_has_none(self):
        legacy = parse_advertisement(bytes([0x02, 0x36, 0x00, 0x6C, 0x00, 0xC3, 0x01, 0x55, 0x0F, 0x16, 0x4D]))
        assert legacy.wake_info(0) is None


class TestConfigSlot:
    def test_zero_is_off(self):
        """Configs from before the field existed (reserved = 0) publish nothing."""
        assert _power(0).published_wake_info_byte_index is None

    def test_one_based(self):
        assert _power(1).published_wake_info_byte_index == 0
        assert _power(7).published_wake_info_byte_index == 6
        assert _power(9).published_wake_info_byte_index == 8

    def test_block_must_fit(self):
        assert _power(10).published_wake_info_byte_index is None
        assert _power(0xFF).published_wake_info_byte_index is None


class TestWakeTrackerLevel2:
    def test_first_advertisement_only_primes(self):
        tracker = WakeTracker(INDEX)
        assert tracker.update("AA", _adv(3), timestamp=0.0) is None
        assert tracker.current("AA") == "rtc"

    def test_same_window_no_event(self):
        tracker = WakeTracker(INDEX)
        tracker.update("AA", _adv(3, until=20), timestamp=0.0)
        assert tracker.update("AA", _adv(3, until=10), timestamp=10.0) is None

    def test_repeat_cause_after_gap_is_new_wake(self):
        tracker = WakeTracker(INDEX, gap_seconds=60)
        tracker.update("AA", _adv(3), timestamp=0.0)
        event = tracker.update("AA", _adv(3, until=19, sleep_code=60), timestamp=900.0)
        assert event is not None
        assert (event.cause, event.previous_cause, event.reason) == ("rtc", "rtc", "gap")
        assert event.info.sleep_seconds == 900

    def test_cause_change_without_gap(self):
        tracker = WakeTracker(INDEX)
        tracker.update("AA", _adv(3), timestamp=0.0)
        event = tracker.update("AA", _adv(1, reboot_flag=True), timestamp=5.0)
        assert (event.cause, event.reason) == ("power_on", "cause_changed")

    def test_reboot_edge_same_cause(self):
        tracker = WakeTracker(INDEX)
        tracker.update("AA", _adv(1, reboot_flag=False), timestamp=0.0)
        event = tracker.update("AA", _adv(1, reboot_flag=True), timestamp=5.0)
        assert event.reason == "rebooted"
        assert tracker.update("AA", _adv(1, reboot_flag=True), timestamp=6.0) is None

    def test_touch_bridges_a_connection(self):
        """Silence while the host is connected is not a sleep."""
        tracker = WakeTracker(INDEX, gap_seconds=60)
        tracker.update("AA", _adv(3), timestamp=0.0)
        tracker.touch("AA", timestamp=50.0)
        tracker.touch("AA", timestamp=100.0)
        assert tracker.update("AA", _adv(3), timestamp=130.0) is None

    def test_per_address_and_reset(self):
        tracker = WakeTracker(INDEX)
        tracker.update("AA", _adv(3), timestamp=0.0)
        assert tracker.update("BB", _adv(4), timestamp=1.0) is None
        tracker.reset("AA")
        assert tracker.current("AA") is None
        assert tracker.update("AA", _adv(1), timestamp=2.0) is None


class TestWakeTrackerLevel1:
    """No wake-info block: the cause comes from the reboot flag."""

    def test_prime_from_flag(self):
        tracker = WakeTracker()
        tracker.update("AA", _adv(reboot_flag=False), timestamp=0.0)
        assert tracker.current("AA") == "scheduled"

    def test_scheduled_wake_after_gap(self):
        tracker = WakeTracker(gap_seconds=60)
        tracker.update("AA", _adv(reboot_flag=False), timestamp=0.0)
        event = tracker.update("AA", _adv(reboot_flag=False), timestamp=900.0)
        assert (event.cause, event.reason) == ("scheduled", "gap")
        assert event.info is None

    def test_restart_after_gap(self):
        tracker = WakeTracker(gap_seconds=60)
        tracker.update("AA", _adv(reboot_flag=False), timestamp=0.0)
        event = tracker.update("AA", _adv(reboot_flag=True), timestamp=900.0)
        assert event.cause == "restart"

    def test_flag_cleared_by_connection_keeps_cause(self):
        """The firmware clears the flag on the first connection inside the window."""
        tracker = WakeTracker(gap_seconds=60)
        tracker.update("AA", _adv(reboot_flag=False), timestamp=0.0)
        tracker.update("AA", _adv(reboot_flag=True), timestamp=900.0)
        assert tracker.update("AA", _adv(reboot_flag=False), timestamp=905.0) is None
        assert tracker.current("AA") == "restart"

    def test_mid_window_flag_rise_is_restart(self):
        tracker = WakeTracker()
        tracker.update("AA", _adv(reboot_flag=False), timestamp=0.0)
        event = tracker.update("AA", _adv(reboot_flag=True), timestamp=5.0)
        assert (event.cause, event.reason) == ("restart", "rebooted")
