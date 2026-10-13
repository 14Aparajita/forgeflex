from dataclasses import dataclass

@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    samples: int = 1440
    interval_seconds: int = 60
    noise_std_kw: float = 1.5

    def validate(self) -> None:
        if self.samples < 10:
            raise ValueError("samples must be >= 10")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if self.noise_std_kw < 0:
            raise ValueError("noise_std_kw must be >= 0")
