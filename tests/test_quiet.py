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


class Clock:
    """Fake clock: each reading takes read_s, each pause advances by the pause."""

    def __init__(self, script, read_s=1.0):
        self.t, self.script, self.read_s, self.sleeps = 0.0, script, read_s, []

    def now(self):
        return self.t

    def read(self):
        self.t += self.read_s
        return self.script.read()

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def adaptive(quiet_settings, **kw):
    return quiet_settings.model_copy(update={"cooldown": "adaptive", "poll_s": 1.0, "max_wait_s": 50.0, **kw})


def test_adaptive_cooldown_starts_as_soon_as_the_check_passes(quiet_settings):
    c = Clock(Script(reading(temp=72), reading(temp=71), reading(temp=69)))
    q = quiet_check(adaptive(quiet_settings), c.read, c.sleep, clock=c.now)
    assert q.passed and q.attempts == 3
    assert c.sleeps == [1.0, 1.0]  # polls every second instead of a fixed 10 s pause
    assert q.waited_s == 2.0 and q.elapsed_s == 5.0  # 3 readings of 1 s + 2 pauses: the actual wait


def test_adaptive_cooldown_gives_up_at_max_wait_and_flags_the_run(quiet_settings):
    c = Clock(Script(reading(util=40)))
    q = quiet_check(adaptive(quiet_settings, max_wait_s=10.0), c.read, c.sleep, clock=c.now)
    assert not q.passed and q.elapsed_s <= 10.0 and q.attempts == 5


def test_adaptive_uses_the_same_thresholds(quiet_settings):
    for temp, ok in ((69.9, True), (70.0, False)):
        c = Clock(Script(reading(temp=temp)))
        q = quiet_check(adaptive(quiet_settings, max_wait_s=0.5), c.read, c.sleep, clock=c.now)
        assert q.passed is ok


def test_missing_gpu_sensor_is_recorded_as_unchecked(quiet_settings):
    q = quiet_check(quiet_settings, Script(reading(util=None, temp=None)).read, lambda s: None)
    assert q.passed and q.unchecked == ("gpu_util", "gpu_temp")
    assert q.gpu_util_pct.status == "unavailable"
