#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use core::fmt;

use instructions::Instruction;
use yaxpeax_arch::{annotation, ReadError, Reader};

mod instructions;
mod types;
mod values;

pub struct Wasm32;
impl yaxpeax_arch::Arch for Wasm32 {
    type Word = u8;
    type Address = u32;
    type Instruction = Instruction;
    type DecodeError = DecodeError;
    type Decoder = Decoder;
    type Operand = Operand;
}

#[derive(Debug, PartialEq)]
pub enum DecodeError {
    Leb128TooLong,
    ReadError(ReadError),
    BadType,
    NonzeroMemIdx,
    BadOpcode,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(yaxpeax_arch::DecodeError::description(self))
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}

impl From<ReadError> for DecodeError {
    fn from(err: ReadError) -> Self {
        Self::ReadError(err)
    }
}

impl yaxpeax_arch::DecodeError for DecodeError {
    fn data_exhausted(&self) -> bool {
        matches!(self, DecodeError::ReadError(ReadError::ExhaustedInput))
    }

    fn bad_opcode(&self) -> bool {
        matches!(self, DecodeError::BadOpcode)
    }

    fn bad_operand(&self) -> bool {
        matches!(
            self,
            DecodeError::BadType | DecodeError::Leb128TooLong | DecodeError::NonzeroMemIdx
        )
    }

    fn description(&self) -> &'static str {
        match self {
            DecodeError::Leb128TooLong => "bad leb128 number",
            DecodeError::ReadError(e) => match e {
                ReadError::ExhaustedInput => "exhausted input",
                ReadError::IOError(s) => s,
            },
            DecodeError::BadType => "bad value type",
            DecodeError::NonzeroMemIdx => "memory index was not 0",
            DecodeError::BadOpcode => "bad opcode",
        }
    }
}

#[derive(Default)]
pub struct Decoder;

impl yaxpeax_arch::Decoder<Wasm32> for Decoder {
    fn decode<T: Reader<u32, u8>>(&self, words: &mut T) -> Result<Instruction, DecodeError> {
        instructions::read_instr(words, &mut annotation::NullSink)
    }

    fn decode_into<T>(&self, inst: &mut Instruction, words: &mut T) -> Result<(), DecodeError>
    where
        T: Reader<u32, u8>,
    {
        *inst = self.decode(words)?;
        Ok(())
    }
}

impl annotation::AnnotatingDecoder<Wasm32> for Decoder {
    type FieldDescription = instructions::FieldDescription;

    fn decode_with_annotation<
        T: Reader<u32, u8>,
        S: annotation::DescriptionSink<Self::FieldDescription>,
    >(
        &self,
        inst: &mut Instruction,
        words: &mut T,
        sink: &mut S,
    ) -> Result<(), DecodeError> {
        *inst = instructions::read_instr(words, sink)?;
        Ok(())
    }
}

pub struct Operand;
