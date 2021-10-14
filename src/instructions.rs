use std::fmt;

use itertools::Itertools;
use num_traits::ToPrimitive;
use yaxpeax_arch::annotation::{self, DescriptionSink};
use yaxpeax_arch::{AddressDiff, LengthedInstruction, Reader};

use crate::types::{NumType, RefType, ValType};
use crate::values;
use crate::DecodeError;

#[derive(Debug)]
pub enum Blocktype {
    Empty,
    Val(ValType),
    Index(u32),
}

impl fmt::Display for Blocktype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Blocktype::Empty => f.write_str("()"),
            Blocktype::Val(v) => v.fmt(f),
            Blocktype::Index(typ) => write!(f, "type {}", typ),
        }
    }
}

macro_rules! record {
    ($sink:expr, $words:expr, $startoff:expr, $endoff:expr, $kind:expr) => {{
        let offbits = $words.offset() as u32 * 8;
        $sink.record(
            offbits - $startoff,
            offbits - $endoff,
            FieldDescription {
                id: offbits,
                kind: {
                    use FieldDescriptionKind::*;
                    $kind
                },
            },
        )
    }};
}

impl Blocktype {
    fn read(
        words: &mut impl Reader<u32, u8>,
        sink: &mut impl DescriptionSink<FieldDescription>,
    ) -> Result<Self, DecodeError> {
        let (int, nbytes) = values::read_leb128_bits::<i64, _>(words, 33)?;
        let b = int.abs().to_u8();
        let ret = if let Some(0x40) = b {
            Blocktype::Empty
        } else if let Some(val) = b.and_then(|i| ValType::parse(i)) {
            Blocktype::Val(val)
        } else {
            let idx = int.to_u32().ok_or(DecodeError::BadType)?;
            Blocktype::Index(idx)
        };
        record!(sink, words, nbytes as u32 * 8, 1, Blocktype);
        Ok(ret)
    }
}

pub type TypeIdx = u32;
pub type FuncIdx = u32;
pub type TableIdx = u32;
// pub type MemIdx = u32;
pub type GlobalIdx = u32;
pub type ElemIdx = u32;
pub type DataIdx = u32;
pub type LocalIdx = u32;
pub type LabelIdx = u32;

#[derive(Debug)]
pub struct MemArg {
    align: u32,
    offset: u32,
}

fn read_memarg(
    words: &mut impl Reader<u32, u8>,
    sink: &mut impl DescriptionSink<FieldDescription>,
) -> Result<MemArg, DecodeError> {
    let (align, n1) = values::read_leb128(words)?;
    let (offset, n2) = values::read_leb128(words)?;
    record!(sink, words, (n1 + n2) as u32 * 8, 1, MemArg);
    Ok(MemArg { align, offset })
}

#[derive(Debug)]
pub enum InstrKind {
    Invalid,

    Unreachable,
    Nop,
    Block(Blocktype),
    Loop(Blocktype),
    If(Blocktype),
    Else,
    End,
    Br(LabelIdx),
    BrIf(LabelIdx),
    BrTable(Box<[LabelIdx]>, LabelIdx),
    Return,
    Call(FuncIdx),
    CallIndirect(TableIdx, TypeIdx),

    RefNull(RefType),
    RefIsNull,
    RefFunc(FuncIdx),

    Drop,
    Select,
    SelectN(Box<[ValType]>),

    LocalGet(LocalIdx),
    LocalSet(LocalIdx),
    LocalTee(LocalIdx),
    GlobalGet(GlobalIdx),
    GlobalSet(GlobalIdx),

    TableGet(TableIdx),
    TableSet(TableIdx),
    TableInit(TableIdx, ElemIdx),
    ElemDrop(ElemIdx),
    TableCopy(TableIdx, TableIdx),
    TableGrow(TableIdx),
    TableSize(TableIdx),
    TableFill(TableIdx),

    I32Load(MemArg),
    I64Load(MemArg),
    F32Load(MemArg),
    F64Load(MemArg),
    I32Load8S(MemArg),
    I32Load8U(MemArg),
    I32Load16S(MemArg),
    I32Load16U(MemArg),
    I64Load8S(MemArg),
    I64Load8U(MemArg),
    I64Load16S(MemArg),
    I64Load16U(MemArg),
    I64Load32S(MemArg),
    I64Load32U(MemArg),
    I32Store(MemArg),
    I64Store(MemArg),
    F32Store(MemArg),
    F64Store(MemArg),
    I32Store8(MemArg),
    I32Store16(MemArg),
    I64Store8(MemArg),
    I64Store16(MemArg),
    I64Store32(MemArg),
    MemorySize,
    MemoryGrow,
    MemoryInit(DataIdx),
    DataDrop(DataIdx),
    MemoryCopy,
    MemoryFill,

    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),

    I32Eqz,
    I32Eq,
    I32Ne,
    I32LtS,
    I32LtU,
    I32GtS,
    I32GtU,
    I32LeS,
    I32LeU,
    I32GeS,
    I32GeU,

    I64Eqz,
    I64Eq,
    I64Ne,
    I64LtS,
    I64LtU,
    I64GtS,
    I64GtU,
    I64LeS,
    I64LeU,
    I64GeS,
    I64GeU,

    F32Eq,
    F32Ne,
    F32Lt,
    F32Gt,
    F32Le,
    F32Ge,

    F64Eq,
    F64Ne,
    F64Lt,
    F64Gt,
    F64Le,
    F64Ge,

    I32Clz,
    I32Ctz,
    I32Popcnt,
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32RemS,
    I32RemU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    I32Rotl,
    I32Rotr,

    I64Clz,
    I64Ctz,
    I64Popcnt,
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64RemS,
    I64RemU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    I64Rotl,
    I64Rotr,

    F32Abs,
    F32Neg,
    F32Ceil,
    F32Floor,
    F32Trunc,
    F32Nearest,
    F32Sqrt,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Min,
    F32Max,
    F32Copysign,

    F64Abs,
    F64Neg,
    F64Ceil,
    F64Floor,
    F64Trunc,
    F64Nearest,
    F64Sqrt,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Min,
    F64Max,
    F64Copysign,

    I32WrapI64,
    I32TruncF32S,
    I32TruncF32U,
    I32TruncF64S,
    I32TruncF64U,
    I64ExtendI32S,
    I64ExtendI32U,
    I64TruncF32S,
    I64TruncF32U,
    I64TruncF64S,
    I64TruncF64U,
    F32ConvertI32S,
    F32ConvertI32U,
    F32ConvertI64S,
    F32ConvertI64U,
    F32DemoteF64,
    F64ConvertI32S,
    F64ConvertI32U,
    F64ConvertI64S,
    F64ConvertI64U,
    F64PromoteF32,
    I32ReinterpretF32,
    I64ReinterpretF64,
    F32ReinterpretI32,
    F64ReinterpretI64,

    I32Extend8S,
    I32Extend16S,
    I64Extend8S,
    I64Extend16S,
    I64Extend32S,

    I32TruncSatF32S,
    I32TruncSatF32U,
    I32TruncSatF64S,
    I32TruncSatF64U,
    I64TruncSatF32S,
    I64TruncSatF32U,
    I64TruncSatF64S,
    I64TruncSatF64U,
}

impl Default for InstrKind {
    fn default() -> Self {
        Self::Invalid
    }
}

impl fmt::Display for InstrKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InstrKind::*;
        let s = match self {
            Invalid => "invalid",
            Unreachable => "unreachable",
            Nop => "nop",
            Block(b) => return write!(f, "block {}", b),
            Loop(b) => return write!(f, "loop {}", b),
            If(b) => return write!(f, "if {}", b),
            Else => "else",
            End => "end",
            Br(l) => return write!(f, "br {}", l),
            BrIf(l) => return write!(f, "br_if {}", l),
            BrTable(ls, l) => return write!(f, "br_table {} {}", ls.iter().format(" "), l),
            Return => "return",
            Call(i) => return write!(f, "call ${}", i),
            CallIndirect(tbl, ty) => return write!(f, "call_indirect {}, type {}", tbl, ty),
            RefNull(r) => return write!(f, "ref.null {}", r),
            RefIsNull => "ref.is_null",
            RefFunc(i) => return write!(f, "ref.func ${}", i),
            Drop => "drop",
            Select => "select",
            SelectN(_) => todo!(),
            LocalGet(_) => todo!(),
            LocalSet(_) => todo!(),
            LocalTee(_) => todo!(),
            GlobalGet(_) => todo!(),
            GlobalSet(_) => todo!(),
            TableGet(_) => todo!(),
            TableSet(_) => todo!(),
            TableInit(_, _) => todo!(),
            ElemDrop(_) => todo!(),
            TableCopy(_, _) => todo!(),
            TableGrow(_) => todo!(),
            TableSize(_) => todo!(),
            TableFill(_) => todo!(),
            I32Load(_) => todo!(),
            I64Load(_) => todo!(),
            F32Load(_) => todo!(),
            F64Load(_) => todo!(),
            I32Load8S(_) => todo!(),
            I32Load8U(_) => todo!(),
            I32Load16S(_) => todo!(),
            I32Load16U(_) => todo!(),
            I64Load8S(_) => todo!(),
            I64Load8U(_) => todo!(),
            I64Load16S(_) => todo!(),
            I64Load16U(_) => todo!(),
            I64Load32S(_) => todo!(),
            I64Load32U(_) => todo!(),
            I32Store(_) => todo!(),
            I64Store(_) => todo!(),
            F32Store(_) => todo!(),
            F64Store(_) => todo!(),
            I32Store8(_) => todo!(),
            I32Store16(_) => todo!(),
            I64Store8(_) => todo!(),
            I64Store16(_) => todo!(),
            I64Store32(_) => todo!(),
            MemorySize => todo!(),
            MemoryGrow => todo!(),
            MemoryInit(_) => todo!(),
            DataDrop(_) => todo!(),
            MemoryCopy => todo!(),
            MemoryFill => todo!(),
            I32Const(i) => return write!(f, "i32.const {}", i),
            I64Const(i) => return write!(f, "i64.const {}", i),
            F32Const(z) => return write!(f, "f32.const {}", z),
            F64Const(z) => return write!(f, "f64.const {}", z),
            I32Eqz => "i32.eqz",
            I32Eq => "i32.eq",
            I32Ne => "i32.ne",
            I32LtS => todo!(),
            I32LtU => todo!(),
            I32GtS => todo!(),
            I32GtU => todo!(),
            I32LeS => todo!(),
            I32LeU => todo!(),
            I32GeS => todo!(),
            I32GeU => todo!(),
            I64Eqz => todo!(),
            I64Eq => todo!(),
            I64Ne => todo!(),
            I64LtS => todo!(),
            I64LtU => todo!(),
            I64GtS => todo!(),
            I64GtU => todo!(),
            I64LeS => todo!(),
            I64LeU => todo!(),
            I64GeS => todo!(),
            I64GeU => todo!(),
            F32Eq => todo!(),
            F32Ne => todo!(),
            F32Lt => todo!(),
            F32Gt => todo!(),
            F32Le => todo!(),
            F32Ge => todo!(),
            F64Eq => todo!(),
            F64Ne => todo!(),
            F64Lt => todo!(),
            F64Gt => todo!(),
            F64Le => todo!(),
            F64Ge => todo!(),
            I32Clz => todo!(),
            I32Ctz => todo!(),
            I32Popcnt => todo!(),
            I32Add => "i32.add",
            I32Sub => todo!(),
            I32Mul => todo!(),
            I32DivS => todo!(),
            I32DivU => todo!(),
            I32RemS => todo!(),
            I32RemU => todo!(),
            I32And => todo!(),
            I32Or => todo!(),
            I32Xor => todo!(),
            I32Shl => todo!(),
            I32ShrS => todo!(),
            I32ShrU => todo!(),
            I32Rotl => todo!(),
            I32Rotr => todo!(),
            I64Clz => todo!(),
            I64Ctz => todo!(),
            I64Popcnt => todo!(),
            I64Add => todo!(),
            I64Sub => todo!(),
            I64Mul => todo!(),
            I64DivS => todo!(),
            I64DivU => todo!(),
            I64RemS => todo!(),
            I64RemU => todo!(),
            I64And => todo!(),
            I64Or => todo!(),
            I64Xor => todo!(),
            I64Shl => todo!(),
            I64ShrS => todo!(),
            I64ShrU => todo!(),
            I64Rotl => todo!(),
            I64Rotr => todo!(),
            F32Abs => todo!(),
            F32Neg => todo!(),
            F32Ceil => todo!(),
            F32Floor => todo!(),
            F32Trunc => todo!(),
            F32Nearest => todo!(),
            F32Sqrt => todo!(),
            F32Add => todo!(),
            F32Sub => todo!(),
            F32Mul => todo!(),
            F32Div => todo!(),
            F32Min => todo!(),
            F32Max => todo!(),
            F32Copysign => todo!(),
            F64Abs => todo!(),
            F64Neg => todo!(),
            F64Ceil => todo!(),
            F64Floor => todo!(),
            F64Trunc => todo!(),
            F64Nearest => todo!(),
            F64Sqrt => todo!(),
            F64Add => todo!(),
            F64Sub => todo!(),
            F64Mul => todo!(),
            F64Div => todo!(),
            F64Min => todo!(),
            F64Max => todo!(),
            F64Copysign => todo!(),
            I32WrapI64 => todo!(),
            I32TruncF32S => todo!(),
            I32TruncF32U => todo!(),
            I32TruncF64S => todo!(),
            I32TruncF64U => todo!(),
            I64ExtendI32S => todo!(),
            I64ExtendI32U => todo!(),
            I64TruncF32S => todo!(),
            I64TruncF32U => todo!(),
            I64TruncF64S => todo!(),
            I64TruncF64U => todo!(),
            F32ConvertI32S => todo!(),
            F32ConvertI32U => todo!(),
            F32ConvertI64S => todo!(),
            F32ConvertI64U => todo!(),
            F32DemoteF64 => todo!(),
            F64ConvertI32S => todo!(),
            F64ConvertI32U => todo!(),
            F64ConvertI64S => todo!(),
            F64ConvertI64U => todo!(),
            F64PromoteF32 => todo!(),
            I32ReinterpretF32 => todo!(),
            I64ReinterpretF64 => todo!(),
            F32ReinterpretI32 => todo!(),
            F64ReinterpretI64 => todo!(),
            I32Extend8S => todo!(),
            I32Extend16S => todo!(),
            I64Extend8S => todo!(),
            I64Extend16S => todo!(),
            I64Extend32S => todo!(),
            I32TruncSatF32S => todo!(),
            I32TruncSatF32U => todo!(),
            I32TruncSatF64S => todo!(),
            I32TruncSatF64U => todo!(),
            I64TruncSatF32S => todo!(),
            I64TruncSatF32U => todo!(),
            I64TruncSatF64S => todo!(),
            I64TruncSatF64U => todo!(),
        };
        f.write_str(s)
    }
}

#[derive(Debug, Default)]
pub struct Instruction {
    len: u8,
    instr: InstrKind,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.instr.fmt(f)
    }
}

impl yaxpeax_arch::Instruction for Instruction {
    fn well_defined(&self) -> bool {
        // !matches!(self.instr, InstrKind::Invalid)
        true
    }
}

impl LengthedInstruction for Instruction {
    type Unit = AddressDiff<u32>;

    fn len(&self) -> Self::Unit {
        AddressDiff::from_const(self.len.into())
    }

    fn min_size() -> Self::Unit {
        AddressDiff::from_const(1)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldDescription {
    id: u32,
    kind: FieldDescriptionKind,
}

impl fmt::Display for FieldDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldDescriptionKind {
    Boundary(&'static str),
    Blocktype,
    Idx(&'static str),
    VecLen,
    Type(&'static str),
    MemArg,
    Opcode,
    Constant(NumType),
}

impl annotation::FieldDescription for FieldDescription {
    fn id(&self) -> u32 {
        self.id
    }

    fn is_separator(&self) -> bool {
        false
    }
}

fn read_idx(
    words: &mut impl Reader<u32, u8>,
    sink: &mut impl DescriptionSink<FieldDescription>,
    idxkind: &'static str,
) -> Result<u32, DecodeError> {
    let (idx, nbytes) = values::read_leb128(words)?;
    record!(sink, words, nbytes as u32 * 8, 1, Idx(idxkind));
    Ok(idx)
}

fn read_vec<
    R: Reader<u32, u8>,
    S: DescriptionSink<FieldDescription>,
    F: FnMut(&mut R, &mut S) -> Result<T, DecodeError>,
    T,
>(
    words: &mut R,
    sink: &mut S,
    mut f: F,
) -> Result<Box<[T]>, DecodeError> {
    let (len, nbytes) = values::read_leb128::<u32, _>(words)?;
    record!(sink, words, nbytes as u32 * 8, 1, VecLen);
    let mut v = Vec::with_capacity(len as usize);
    for _ in 0..len {
        v.push(f(words, sink)?)
    }
    Ok(v.into_boxed_slice())
}

fn read_type<T>(
    words: &mut impl Reader<u32, u8>,
    sink: &mut impl DescriptionSink<FieldDescription>,
    typ: &'static str,
    parse: impl FnOnce(u8) -> Option<T>,
) -> Result<T, DecodeError> {
    let b = words.next()?;
    let parsed = parse(b).ok_or(DecodeError::BadType)?;
    record!(sink, words, 8, 1, Type(typ));
    Ok(parsed)
}

fn read_instrkind<R: Reader<u32, u8>, S: DescriptionSink<FieldDescription>>(
    words: &mut R,
    sink: &mut S,
) -> Result<InstrKind, DecodeError> {
    let mut simple_ins = true;
    let mut _opc_done = |words: &mut R, sink: &mut S| {
        if core::mem::take(&mut simple_ins) {
            let end = words.offset() as u32 * 8 - 1;
            sink.record(
                0,
                end,
                FieldDescription {
                    id: end,
                    kind: FieldDescriptionKind::Opcode,
                },
            );
            record!(sink, words, 1, 1, Boundary(""));
        }
    };
    macro_rules! opc_done {
        () => {
            _opc_done(words, sink)
        };
    }
    macro_rules! onearg {
        ($var:ident, $readf:expr) => {{
            opc_done!();
            InstrKind::$var($readf(words, sink)?)
        }};
    }
    use InstrKind::*;
    let make_readidx = |k| move |w: &mut R, s: &mut S| read_idx(w, s, k);
    let read_labeli = make_readidx("label");
    let read_locali = make_readidx("local");
    let read_globali = make_readidx("global");
    let read_funci = make_readidx("func");
    let read_datai = make_readidx("data");
    let read_elemi = make_readidx("elem");
    let read_tablei = make_readidx("table");
    let read_typei = make_readidx("type");
    let kind = match words.next()? {
        0x00 => Unreachable,
        0x01 => Nop,
        0x02 => onearg!(Block, Blocktype::read),
        0x03 => onearg!(Loop, Blocktype::read),
        0x04 => onearg!(If, Blocktype::read),
        0x05 => Else,
        0x0B => End,
        0x0C => onearg!(Br, read_labeli),
        0x0D => onearg!(BrIf, read_labeli),
        0x0E => {
            opc_done!();
            BrTable(
                read_vec(words, sink, read_labeli)?,
                read_labeli(words, sink)?,
            )
        }
        0x0F => Return,
        0x10 => {
            opc_done!();
            Call(read_funci(words, sink)?)
        }
        0x11 => {
            opc_done!();
            let y = read_tablei(words, sink)?;
            let x = read_typei(words, sink)?;
            CallIndirect(x, y)
        }
        0xD0 => {
            opc_done!();
            RefNull(read_type(words, sink, "ref", RefType::parse)?)
        }
        0xD1 => RefIsNull,
        0xD2 => onearg!(RefFunc, read_funci),
        0x1A => Drop,
        0x1B => Select,
        0x1C => {
            opc_done!();
            SelectN(read_vec(words, sink, |w, s| {
                read_type(w, s, "val", ValType::parse)
            })?)
        }
        0x20 => onearg!(LocalGet, read_locali),
        0x21 => onearg!(LocalSet, read_locali),
        0x22 => onearg!(LocalTee, read_locali),
        0x23 => onearg!(GlobalGet, read_globali),
        0x24 => onearg!(GlobalSet, read_globali),
        0x25 => onearg!(TableGet, read_tablei),
        0x26 => onearg!(TableSet, read_tablei),
        0x28 => onearg!(I32Load, read_memarg),
        0x29 => onearg!(I64Load, read_memarg),
        0x2A => onearg!(F32Load, read_memarg),
        0x2B => onearg!(F64Load, read_memarg),
        0x2C => onearg!(I32Load8S, read_memarg),
        0x2D => onearg!(I32Load8U, read_memarg),
        0x2E => onearg!(I32Load16S, read_memarg),
        0x2F => onearg!(I32Load16U, read_memarg),
        0x30 => onearg!(I64Load8S, read_memarg),
        0x31 => onearg!(I64Load8U, read_memarg),
        0x32 => onearg!(I64Load16S, read_memarg),
        0x33 => onearg!(I64Load16U, read_memarg),
        0x34 => onearg!(I64Load32S, read_memarg),
        0x35 => onearg!(I64Load32U, read_memarg),
        0x36 => onearg!(I32Store, read_memarg),
        0x37 => onearg!(I64Store, read_memarg),
        0x38 => onearg!(F32Store, read_memarg),
        0x39 => onearg!(F64Store, read_memarg),
        0x3A => onearg!(I32Store8, read_memarg),
        0x3B => onearg!(I32Store16, read_memarg),
        0x3C => onearg!(I64Store8, read_memarg),
        0x3D => onearg!(I64Store16, read_memarg),
        0x3E => onearg!(I64Store32, read_memarg),
        0x3F => {
            opc_done!();
            read_memidx(words, sink)?;
            MemorySize
        }
        0x40 => {
            opc_done!();
            read_memidx(words, sink)?;
            MemoryGrow
        }
        0x41 => {
            opc_done!();
            let (i, nbytes) = values::read_leb128(words)?;
            record!(sink, words, nbytes as u32 * 8, 1, Constant(NumType::I32));
            I32Const(i)
        }
        0x42 => {
            opc_done!();
            let (i, nbytes) = values::read_leb128(words)?;
            record!(sink, words, nbytes as u32 * 8, 1, Constant(NumType::I64));
            I64Const(i)
        }
        0x43 => {
            opc_done!();
            let f = values::read_f32(words)?;
            record!(sink, words, 32, 1, Constant(NumType::F32));
            F32Const(f)
        }
        0x44 => {
            opc_done!();
            let f = values::read_f64(words)?;
            record!(sink, words, 64, 1, Constant(NumType::F64));
            F64Const(f)
        }

        0x45 => I32Eqz,
        0x46 => I32Eq,
        0x47 => I32Ne,
        0x48 => I32LtS,
        0x49 => I32LtU,
        0x4A => I32GtS,
        0x4B => I32GtU,
        0x4C => I32LeS,
        0x4D => I32LeU,
        0x4E => I32GeS,
        0x4F => I32GeU,

        0x50 => I64Eqz,
        0x51 => I64Eq,
        0x52 => I64Ne,
        0x53 => I64LtS,
        0x54 => I64LtU,
        0x55 => I64GtS,
        0x56 => I64GtU,
        0x57 => I64LeS,
        0x58 => I64LeU,
        0x59 => I64GeS,
        0x5A => I64GeU,

        0x5B => F32Eq,
        0x5C => F32Ne,
        0x5D => F32Lt,
        0x5E => F32Gt,
        0x5F => F32Le,
        0x60 => F32Ge,

        0x61 => F64Eq,
        0x62 => F64Ne,
        0x63 => F64Lt,
        0x64 => F64Gt,
        0x65 => F64Le,
        0x66 => F64Ge,

        0x67 => I32Clz,
        0x68 => I32Ctz,
        0x69 => I32Popcnt,
        0x6A => I32Add,
        0x6B => I32Sub,
        0x6C => I32Mul,
        0x6D => I32DivS,
        0x6E => I32DivU,
        0x6F => I32RemS,
        0x70 => I32RemU,
        0x71 => I32And,
        0x72 => I32Or,
        0x73 => I32Xor,
        0x74 => I32Shl,
        0x75 => I32ShrS,
        0x76 => I32ShrU,
        0x77 => I32Rotl,
        0x78 => I32Rotr,

        0x79 => I64Clz,
        0x7A => I64Ctz,
        0x7B => I64Popcnt,
        0x7C => I64Add,
        0x7D => I64Sub,
        0x7E => I64Mul,
        0x7F => I64DivS,
        0x80 => I64DivU,
        0x81 => I64RemS,
        0x82 => I64RemU,
        0x83 => I64And,
        0x84 => I64Or,
        0x85 => I64Xor,
        0x86 => I64Shl,
        0x87 => I64ShrS,
        0x88 => I64ShrU,
        0x89 => I64Rotl,
        0x8A => I64Rotr,

        0x8B => F32Abs,
        0x8C => F32Neg,
        0x8D => F32Ceil,
        0x8E => F32Floor,
        0x8F => F32Trunc,
        0x90 => F32Nearest,
        0x91 => F32Sqrt,
        0x92 => F32Add,
        0x93 => F32Sub,
        0x94 => F32Mul,
        0x95 => F32Div,
        0x96 => F32Min,
        0x97 => F32Max,
        0x98 => F32Copysign,

        0x99 => F64Abs,
        0x9A => F64Neg,
        0x9B => F64Ceil,
        0x9C => F64Floor,
        0x9D => F64Trunc,
        0x9E => F64Nearest,
        0x9F => F64Sqrt,
        0xA0 => F64Add,
        0xA1 => F64Sub,
        0xA2 => F64Mul,
        0xA3 => F64Div,
        0xA4 => F64Min,
        0xA5 => F64Max,
        0xA6 => F64Copysign,

        0xA7 => I32WrapI64,
        0xA8 => I32TruncF32S,
        0xA9 => I32TruncF32U,
        0xAA => I32TruncF64S,
        0xAB => I32TruncF64U,
        0xAC => I64ExtendI32S,
        0xAD => I64ExtendI32U,
        0xAE => I64TruncF32S,
        0xAF => I64TruncF32U,
        0xB0 => I64TruncF64S,
        0xB1 => I64TruncF64U,
        0xB2 => F32ConvertI32S,
        0xB3 => F32ConvertI32U,
        0xB4 => F32ConvertI64S,
        0xB5 => F32ConvertI64U,
        0xB6 => F32DemoteF64,
        0xB7 => F64ConvertI32S,
        0xB8 => F64ConvertI32U,
        0xB9 => F64ConvertI64S,
        0xBA => F64ConvertI64U,
        0xBB => F64PromoteF32,
        0xBC => I32ReinterpretF32,
        0xBD => I64ReinterpretF64,
        0xBE => F32ReinterpretI32,
        0xBF => F64ReinterpretI64,

        0xC0 => I32Extend8S,
        0xC1 => I32Extend16S,
        0xC2 => I64Extend8S,
        0xC3 => I64Extend16S,
        0xC4 => I64Extend32S,

        0xFC => {
            let (b2, _) = values::read_leb128::<u32, _>(words)?;
            match b2 {
                0 => I32TruncSatF32S,
                1 => I32TruncSatF32U,
                2 => I32TruncSatF64S,
                3 => I32TruncSatF64U,
                4 => I64TruncSatF32S,
                5 => I64TruncSatF32U,
                6 => I64TruncSatF64S,
                7 => I64TruncSatF64U,
                8 => {
                    opc_done!();
                    let x = read_datai(words, sink)?;
                    read_memidx(words, sink)?;
                    MemoryInit(x)
                }
                9 => onearg!(DataDrop, read_datai),
                10 => {
                    opc_done!();
                    read_memidx(words, sink)?;
                    read_memidx(words, sink)?;
                    MemoryCopy
                }
                11 => {
                    opc_done!();
                    read_memidx(words, sink)?;
                    MemoryFill
                }
                12 => {
                    opc_done!();
                    let y = read_elemi(words, sink)?;
                    let x = read_tablei(words, sink)?;
                    TableInit(x, y)
                }
                13 => onearg!(ElemDrop, read_elemi),
                14 => {
                    opc_done!();
                    let x = read_tablei(words, sink)?;
                    let y = read_tablei(words, sink)?;
                    TableCopy(x, y)
                }
                15 => onearg!(TableGrow, read_tablei),
                16 => onearg!(TableSize, read_tablei),
                17 => onearg!(TableFill, read_tablei),
                _ => return Err(DecodeError::BadOpcode),
            }
        }
        _ => return Err(DecodeError::BadOpcode),
    };
    opc_done!();
    Ok(kind)
}

fn read_memidx(
    words: &mut impl Reader<u32, u8>,
    sink: &mut impl DescriptionSink<FieldDescription>,
) -> Result<(), DecodeError> {
    let n = words.next()?;
    if n != 0 {
        return Err(DecodeError::NonzeroMemIdx);
    }
    record!(sink, words, 8, 1, Idx("(reserved) mem"));
    Ok(())
}

pub fn read_instr<R: Reader<u32, u8>, S: DescriptionSink<FieldDescription>>(
    words: &mut R,
    sink: &mut S,
) -> Result<Instruction, DecodeError> {
    words.mark();
    let instr = read_instrkind(words, sink)?;
    let len = words.offset() as u8;
    Ok(Instruction { instr, len })
}
