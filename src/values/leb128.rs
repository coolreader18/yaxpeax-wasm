// This is modified from wasabi_leb128 by Daniel Lehmann: https://github.com/danleh/wasabi_leb128/
//
// MIT License
//
// Copyright (c) 2019 Daniel Lehmann
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use num_traits::{AsPrimitive, PrimInt};
use yaxpeax_arch::Reader;

use crate::DecodeError;

#[inline]
const fn max_bytes(bits: usize) -> usize {
    // See https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c
    const fn int_div_ceil(x: usize, y: usize) -> usize {
        1 + ((x - 1) / y)
    }

    // ceil( bits(T) / 7 non-continuation bits per LEB128 byte )
    int_div_ceil(bits, 7)
}

fn is_signed<T: PrimInt>() -> bool {
    !T::min_value().is_zero()
}

const CONTINUATION_BIT: u8 = 0x80;

#[inline]
fn continuation_bit(byte: u8) -> bool {
    byte & CONTINUATION_BIT == CONTINUATION_BIT
}

/// Mask off the continuation bit from the byte (= extract only the last 7, meaningful LEB128 bits).
#[inline]
fn non_continuation_bits(byte: u8) -> u8 {
    byte & !CONTINUATION_BIT
}

const SIGN_BIT: u8 = 0x40;

#[inline]
fn sign_bit(byte: u8) -> bool {
    byte & SIGN_BIT == SIGN_BIT
}

pub fn read<T: PrimInt + 'static, R: Reader<u32, u8>>(
    words: &mut R,
) -> Result<(T, usize), DecodeError>
where
    u8: AsPrimitive<T>,
{
    // TODO Should be const, not let because it only depends on T, but Rust doesn't allow it (yet).
    // Rust 1.37: "error[E0401]: can't use generic parameters from outer function".
    let bits: usize = std::mem::size_of::<T>() * 8;
    read_bits(words, bits)
}

pub fn read_bits<T: PrimInt + 'static, R: Reader<u32, u8>>(
    words: &mut R,
    bits: usize,
) -> Result<(T, usize), DecodeError>
where
    u8: AsPrimitive<T>,
{
    let max_bytes = max_bytes(bits);

    let mut value = T::zero();
    let mut shift: usize = 0;
    let mut bytes_read = 0;
    let mut current_byte = CONTINUATION_BIT;

    while continuation_bit(current_byte) {
        current_byte = words.next()?;
        bytes_read += 1;

        if bytes_read > max_bytes {
            return Err(DecodeError::Leb128TooLong);
        }

        let is_last_byte = bytes_read == max_bytes;
        if is_last_byte {
            // The last LEB128 byte has the following structure:
            // -------------------------
            // | c | u ... | s | v ... |
            // -------------------------
            // Where:
            // - c = continuation bit.
            // - u = undefined or "extra bits", which cannot be represented in the target type.
            // - s = sign bit (only if target type is signed).
            // - v = the remaining "value bits".
            // We need to check that:
            // - For signed types: all u bits are equal to the sign bit s. (The byte must be
            //   properly sign-extended.)
            // - For unsigned types: all u bits are 0. (There is no sign bit s.)

            // TODO This should be const (depends on T only), but doesn't work yet, see above.
            let value_bit_count: usize = bits
                    // Bits in the LEB128 bytes so far.
                    - ((max_bytes - 1) * 7)
                    // For signed values, we also check the sign bit, so there is one less value bit.
                    - if is_signed::<T>() { 1 } else { 0 };
            // Extract the extra bits and the sign bit (for signed values) from the input byte.
            let extra_bits_mask = non_continuation_bits(0xffu8 << value_bit_count);
            let extra_bits = current_byte & extra_bits_mask;

            let extra_bits_valid = if is_signed::<T>() {
                // For signed types: The extra bits *plus* the sign bit must either be all 0
                // (non-negative value) or all 1 (negative value, properly sign-extended).
                extra_bits == 0 || extra_bits == extra_bits_mask
            } else {
                // For unsigned types: extra bits must be 0.
                extra_bits == 0
            };

            if !extra_bits_valid {
                return Err(DecodeError::Leb128TooLong);
            }
        }

        // Prepend the extracted bits to value.
        // The following shift left cannot overflow (= shift amount larger than target type,
        // which would be an error in Rust), because the previous condition implies it already:
        //     bytes_read <= max_bytes(T)      // condition that is ensured above
        // <=> bytes_read <= ceil(bits(T) / 7) // substitute definition of max_bytes
        // <=> bytes_read < bits(T) / 7 + 1    // forall x: ceil(x) < x + 1, here x = bits(T) / 7
        // <=> shift / 7 + 1 < bits(T) / 7 + 1 // express bytes_read in terms of shift
        // <=> shift < bits(T)                 // qed.
        let new_bits: T = non_continuation_bits(current_byte).as_().shl(shift);
        value = value.bitor(new_bits);

        shift += 7;
    }

    // Sign-extend value if:
    // - type is signed
    if is_signed::<T>()
            // - value is negative (= sign bit of last LEB128 byte was set)
            && sign_bit(current_byte)
            // - shift amount does not overflow bit-width of target type
            //   (disallowed in Rust, will panic in debug mode).
            && shift < bits
    {
        let sign_extend = (!T::zero()).shl(shift);
        value = value.bitor(sign_extend);
    }

    Ok((value, bytes_read))
}
