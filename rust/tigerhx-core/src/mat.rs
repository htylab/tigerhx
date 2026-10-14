//! MATLAB 5 (.mat) files with compressed variables, as scipy's savemat
//! writes them: numeric arrays, scalars and strings, column-major data.

use std::io::Write;
use std::path::Path;

use flate2::write::ZlibEncoder;
use flate2::Compression;

use crate::error::Result;
use crate::fsutil::write_atomic;

const MI_INT8: u32 = 1;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_SINGLE: u32 = 7;
const MI_UINT32: u32 = 6;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;

const MX_CHAR: u32 = 4;
const MX_DOUBLE: u32 = 6;
const MX_SINGLE: u32 = 7;
const MX_INT64: u32 = 14;

/// Values of a variable, in column-major (NIfTI) order.
pub enum Data<'a> {
    /// float32 voxels stored as double
    F32AsDouble(&'a [f32]),
    F64(Vec<f64>),
    F32(Vec<f32>),
    /// uint8 labels stored as int64 (what the original GUI wrote)
    U8AsInt64(&'a [u8]),
    I64(Vec<i64>),
    Str(String),
}

pub struct Var<'a> {
    pub name: &'a str,
    /// dimensions; 1D values are written as a row (1 x n), like savemat
    pub dims: Vec<usize>,
    pub data: Data<'a>,
}

impl<'a> Var<'a> {
    pub fn new(name: &'a str, dims: &[usize], data: Data<'a>) -> Self {
        let dims = match dims.len() {
            0 => vec![1, 1],
            1 => vec![1, dims[0]],
            _ => dims.to_vec(),
        };
        Var { name, dims, data }
    }
    pub fn scalar_i64(name: &'a str, v: i64) -> Self {
        Var::new(name, &[], Data::I64(vec![v]))
    }
    pub fn string(name: &'a str, s: &str) -> Self {
        let n = s.encode_utf16().count();
        Var::new(name, &[1, n], Data::Str(s.to_string()))
    }
}

fn pad8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

fn tag(out: &mut Vec<u8>, ty: u32, n: usize) {
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
}

fn pad(out: &mut Vec<u8>, n: usize) {
    out.resize(out.len() + pad8(n) - n, 0);
}

impl Var<'_> {
    fn class_and_type(&self) -> (u32, u32, usize) {
        match &self.data {
            Data::F32AsDouble(_) | Data::F64(_) => (MX_DOUBLE, MI_DOUBLE, 8),
            Data::F32(_) => (MX_SINGLE, MI_SINGLE, 4),
            Data::U8AsInt64(_) | Data::I64(_) => (MX_INT64, MI_INT64, 8),
            Data::Str(_) => (MX_CHAR, MI_UINT16, 2),
        }
    }

    fn count(&self) -> usize {
        match &self.data {
            Data::F32AsDouble(v) => v.len(),
            Data::F64(v) => v.len(),
            Data::F32(v) => v.len(),
            Data::U8AsInt64(v) => v.len(),
            Data::I64(v) => v.len(),
            Data::Str(s) => s.encode_utf16().count(),
        }
    }

    /// The miMATRIX element, streamed into `out`.
    fn write_matrix(&self, out: &mut impl Write) -> std::io::Result<()> {
        let (class, ty, size) = self.class_and_type();
        let n = self.count();
        debug_assert_eq!(n, self.dims.iter().product::<usize>());
        let mut head = Vec::new();
        // array flags
        tag(&mut head, MI_UINT32, 8);
        head.extend_from_slice(&class.to_le_bytes());
        head.extend_from_slice(&0u32.to_le_bytes());
        // dimensions
        tag(&mut head, MI_INT32, 4 * self.dims.len());
        for &d in &self.dims {
            head.extend_from_slice(&(d as i32).to_le_bytes());
        }
        pad(&mut head, 4 * self.dims.len());
        // name (small element format when it fits in 4 bytes)
        let name = self.name.as_bytes();
        if name.len() <= 4 {
            head.extend_from_slice(&(MI_INT8 as u16).to_le_bytes());
            head.extend_from_slice(&(name.len() as u16).to_le_bytes());
            head.extend_from_slice(name);
            head.resize(head.len() + 4 - name.len(), 0);
        } else {
            tag(&mut head, MI_INT8, name.len());
            head.extend_from_slice(name);
            pad(&mut head, name.len());
        }
        let nbytes = n * size;
        tag(&mut head, ty, nbytes);
        let mut total = Vec::new();
        tag(&mut total, MI_MATRIX, head.len() + pad8(nbytes));
        out.write_all(&total)?;
        out.write_all(&head)?;
        let mut buf = Vec::with_capacity(1 << 16);
        let flush = |buf: &mut Vec<u8>, out: &mut dyn Write| -> std::io::Result<()> {
            out.write_all(buf)?;
            buf.clear();
            Ok(())
        };
        macro_rules! stream {
            ($iter:expr) => {
                for v in $iter {
                    buf.extend_from_slice(&v.to_le_bytes());
                    if buf.len() >= 1 << 16 {
                        flush(&mut buf, out)?;
                    }
                }
            };
        }
        match &self.data {
            Data::F32AsDouble(v) => stream!(v.iter().map(|&x| x as f64)),
            Data::F64(v) => stream!(v.iter().copied()),
            Data::F32(v) => stream!(v.iter().copied()),
            Data::U8AsInt64(v) => stream!(v.iter().map(|&x| x as i64)),
            Data::I64(v) => stream!(v.iter().copied()),
            Data::Str(s) => stream!(s.encode_utf16()),
        }
        buf.resize(buf.len() + pad8(nbytes) - nbytes, 0);
        flush(&mut buf, out)
    }
}

/// Write the variables to a .mat file (through a temporary file).
pub fn save_mat(path: &Path, vars: &[Var<'_>]) -> Result<()> {
    write_atomic(path, |f| {
        let mut header = format!("MATLAB 5.0 MAT-file, Platform: {}, Created by: TigerHx", std::env::consts::OS)
            .into_bytes();
        header.resize(116, b' ');
        header.extend_from_slice(&[0u8; 8]); // no subsystem data
        header.extend_from_slice(&0x0100u16.to_le_bytes());
        header.extend_from_slice(b"IM");
        f.write_all(&header)?;
        for var in vars {
            let mut enc = ZlibEncoder::new(Vec::new(), Compression::new(1));
            var.write_matrix(&mut enc)?;
            let z = enc.finish()?;
            let mut t = Vec::new();
            tag(&mut t, MI_COMPRESSED, z.len());
            f.write_all(&t)?;
            f.write_all(&z)?;
        }
        Ok(())
    })
}
