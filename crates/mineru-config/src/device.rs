//! The compute device MinerU runs models on.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// The compute device model inference targets.
///
/// Mirrors the Python `device-mode` string but as a closed enum so downstream
/// code never has to match on a bare string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU inference.
    Cpu,
    /// A CUDA GPU, identified by its device index (`cuda:0`, `cuda:1`, ...).
    Cuda(usize),
    /// Apple Metal Performance Shaders (Apple Silicon GPU).
    Mps,
}

impl Default for Device {
    /// The default device.
    ///
    /// For now this is always [`Device::Cpu`].
    // TODO: auto-detect — probe for an available CUDA GPU or Apple MPS backend
    // once feature detection exists and prefer it over CPU.
    fn default() -> Self {
        Device::Cpu
    }
}

impl FromStr for Device {
    type Err = Error;

    /// Parses strings like `"cpu"`, `"cuda"`, `"cuda:0"`, and `"mps"`.
    ///
    /// A bare `"cuda"` is treated as `cuda:0`. Parsing is case-insensitive.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.trim().to_ascii_lowercase();
        match lower.as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" | "gpu" => Ok(Device::Cuda(0)),
            "mps" => Ok(Device::Mps),
            other => match other.strip_prefix("cuda:") {
                Some(idx) => idx
                    .parse::<usize>()
                    .map(Device::Cuda)
                    .map_err(|_| Error::InvalidDevice(s.to_string())),
                None => Err(Error::InvalidDevice(s.to_string())),
            },
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{idx}"),
            Device::Mps => f.write_str("mps"),
        }
    }
}

impl Serialize for Device {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_variants() {
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("mps".parse::<Device>().unwrap(), Device::Mps);
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda(0));
        assert_eq!("cuda:0".parse::<Device>().unwrap(), Device::Cuda(0));
        assert_eq!("cuda:1".parse::<Device>().unwrap(), Device::Cuda(1));
    }

    #[test]
    fn parsing_is_case_and_space_insensitive() {
        assert_eq!(" CPU ".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("CUDA:2".parse::<Device>().unwrap(), Device::Cuda(2));
    }

    #[test]
    fn rejects_garbage() {
        assert!("tpu".parse::<Device>().is_err());
        assert!("cuda:notanumber".parse::<Device>().is_err());
    }

    #[test]
    fn default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn display_roundtrips() {
        for d in [Device::Cpu, Device::Cuda(0), Device::Cuda(3), Device::Mps] {
            assert_eq!(d.to_string().parse::<Device>().unwrap(), d);
        }
    }

    #[test]
    fn serde_uses_string_form() {
        let json = serde_json::to_string(&Device::Cuda(1)).unwrap();
        assert_eq!(json, "\"cuda:1\"");
        assert_eq!(serde_json::from_str::<Device>(&json).unwrap(), Device::Cuda(1));
    }
}
