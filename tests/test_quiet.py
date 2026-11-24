from headroom.runner.quiet import QuietReading, quiet_check
from headroom.schemas.common import measured, unavailable


def reading(util=3.0, temp=55.0, ac=1, plan="Balanced"):
    return QuietReading(
        gpu_util_pct=measured(util, "%") if util is not None else unavailable("no NVML"),
        gpu_temp_c=measured(temp, "C") if temp is not None else unavailable("no NVML"),
        on_ac=measured(ac, "bool"),
        power_plan=plan,
    )


class Script:
    def __init__(self, *readings):
        self.readings, self.sleeps = list(readings), []

    def read(self):
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]

    def sleep(self, s):
        self.sleeps.append(s)


def test_quiet_machine_passes_first_time(quiet_settings):
    s = Script(reading())
    q = quiet_check(quiet_settings, s.read, s.sleep)
    assert q.passed and q.attempts == 1 and q.waited_s == 0 and s.sleeps == []
    assert q.gpu_util_pct.value == 3.0 and q.power_plan == "Balanced"


def test_hot_gpu_waits_then_passes(quiet_settings):
    s = Script(reading(temp=80), reading(temp=72), reading(temp=65))
    q = quiet_check(quiet_settings, s.read, s.sleep)
    assert q.passed and q.attempts == 3 and q.waited_s == 20 and s.sleeps == [10, 10]
    assert q.gpu_temp_c.value == 65


def test_never_quiet_is_flagged_after_max_attempts(quiet_settings):
    s = Script(reading(util=40))
    q = quiet_check(quiet_settings, s.read, s.sleep)
    assert not q.passed and q.attempts == quiet_settings.max_attempts
    assert q.failures == ("GPU util 40% >= 10%",)


def test_power_problems_do_not_wait_because_waiting_cannot_fix_them(quiet_settings):
    s = Script(reading(ac=0, plan="Power saver"))
    q = quiet_check(quiet_settings, s.read, s.sleep)
    assert not q.passed and q.attempts == 1 and s.sleeps == []
    assert set(q.failures) == {"not on AC power", "power plan 'Power saver' != 'Balanced'"}


def test_missing_gpu_sensor_is_recorded_as_unchecked(quiet_settings):
    q = quiet_check(quiet_settings, Script(reading(util=None, temp=None)).read, lambda s: None)
    assert q.passed and q.unchecked == ("gpu_util", "gpu_temp")
    assert q.gpu_util_pct.status == "unavailable"
