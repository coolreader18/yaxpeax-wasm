use std::fmt;

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum NumType {
    I32 = 0x7f,
    I64 = 0x7e,
    F32 = 0x7d,
    F64 = 0x7c,
}

impl NumType {
    #[inline]
    pub fn parse(n: u8) -> Option<Self> {
        use NumType::*;
        let ret = match n {
            0x7f => I32,
            0x7e => I64,
            0x7d => F32,
            0x7c => F64,
            _ => return None,
        };
        Some(ret)
    }
}

impl fmt::Display for NumType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            NumType::I32 => "i32",
            NumType::I64 => "i64",
            NumType::F32 => "f32",
            NumType::F64 => "f64",
        })
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum RefType {
    Funcref = 0x70,
    Externref = 0x6f,
}

impl RefType {
    #[inline]
    pub fn parse(n: u8) -> Option<Self> {
        use RefType::*;
        let ret = match n {
            0x70 => Funcref,
            0x6f => Externref,
            _ => return None,
        };
        Some(ret)
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RefType::Funcref => "funcref",
            RefType::Externref => "externref",
        })
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum ValType {
    Num(NumType),
    Ref(RefType),
}

impl ValType {
    #[inline]
    pub fn parse(n: u8) -> Option<Self> {
        let ret = if let Some(x) = NumType::parse(n) {
            Self::Num(x)
        } else if let Some(x) = RefType::parse(n) {
            Self::Ref(x)
        } else {
            return None;
        };
        Some(ret)
    }
}

impl fmt::Display for ValType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValType::Num(n) => n.fmt(f),
            ValType::Ref(r) => r.fmt(f),
        }
    }
}
