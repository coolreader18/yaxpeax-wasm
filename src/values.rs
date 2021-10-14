mod leb128;

pub use leb128::{read as read_leb128, read_bits as read_leb128_bits};
use yaxpeax_arch::Reader;

use crate::DecodeError;

macro_rules! readfloat {
    ($f:ident, $float:ident, $int:ident) => {
        pub fn $f(words: &mut impl Reader<u32, u8>) -> Result<$float, DecodeError> {
            let mut bytes = [0; core::mem::size_of::<$float>()];
            words.next_n(&mut bytes)?;
            Ok($float::from_bits($int::from_le_bytes(bytes)))
        }
    };
}

readfloat!(read_f64, f64, u64);
readfloat!(read_f32, f32, u32);
