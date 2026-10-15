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

// ---------------------------------------------------------------- reading

/// A variable read from a .mat file: dimensions (column-major) and values.
#[derive(Clone, Debug, PartialEq)]
pub struct MatVar {
    pub dims: Vec<usize>,
    pub data: MatData,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MatData {
    /// any numeric or logical class, as f64
    Num(Vec<f64>),
    Str(String),
}

const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT64: u32 = 13;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;

fn le_u32(b: &[u8], off: usize) -> Option<u32> {
    b.get(off..off + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

/// One data element: (type, payload, position after it).
fn element(b: &[u8], pos: usize) -> Result<(u32, &[u8], usize)> {
    let first = le_u32(b, pos).ok_or_else(|| crate::Error::invalid("truncated .mat element"))?;
    if first >> 16 != 0 {
        // small data element: type and size in one word, data in the next 4 bytes
        let (ty, n) = (first & 0xffff, (first >> 16) as usize);
        let data = b.get(pos + 4..pos + 4 + n).ok_or_else(|| crate::Error::invalid("truncated .mat element"))?;
        return Ok((ty, data, pos + 8));
    }
    let n = le_u32(b, pos + 4).ok_or_else(|| crate::Error::invalid("truncated .mat element"))? as usize;
    let data = b.get(pos + 8..pos + 8 + n).ok_or_else(|| crate::Error::invalid("truncated .mat element"))?;
    let next = if first == MI_COMPRESSED { pos + 8 + n } else { pos + 8 + pad8(n) };
    Ok((first, data, next))
}

fn numbers(ty: u32, d: &[u8]) -> Option<Vec<f64>> {
    macro_rules! conv {
        ($t:ty, $n:expr) => {
            d.chunks_exact($n).map(|c| <$t>::from_le_bytes(c.try_into().unwrap()) as f64).collect()
        };
    }
    Some(match ty {
        MI_INT8 => d.iter().map(|&v| v as i8 as f64).collect(),
        MI_UINT8 => d.iter().map(|&v| v as f64).collect(),
        MI_INT16 => conv!(i16, 2),
        MI_UINT16 => conv!(u16, 2),
        MI_INT32 => conv!(i32, 4),
        MI_UINT32 => conv!(u32, 4),
        MI_SINGLE => conv!(f32, 4),
        MI_DOUBLE => conv!(f64, 8),
        MI_INT64 => conv!(i64, 8),
        MI_UINT64 => conv!(u64, 8),
        _ => return None,
    })
}

/// A miMATRIX element: its name and value (None for classes TigerHx does
/// not read: cells, structs, objects, sparse matrices).
fn matrix(b: &[u8]) -> Result<Option<(String, MatVar)>> {
    let (_, flags, pos) = element(b, 0)?;
    let class = le_u32(flags, 0).unwrap_or(0) & 0xff;
    let (_, dims_raw, pos) = element(b, pos)?;
    let dims: Vec<usize> =
        dims_raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()).max(0) as usize).collect();
    let (_, name, pos) = element(b, pos)?;
    let name = String::from_utf8_lossy(name).into_owned();
    if !matches!(class, 4 | 6..=15) {
        return Ok(None);
    }
    let (ty, d, _) = if dims.iter().product::<usize>() == 0 { (MI_DOUBLE, &[][..], 0) } else { element(b, pos)? };
    let data = if class == MX_CHAR {
        MatData::Str(match ty {
            MI_UTF8 | MI_INT8 | MI_UINT8 => String::from_utf8_lossy(d).into_owned(),
            MI_UTF16 | MI_UINT16 => String::from_utf16_lossy(
                &d.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<_>>(),
            ),
            _ => String::new(),
        })
    } else {
        match numbers(ty, d) {
            Some(v) => MatData::Num(v),
            None => return Ok(None),
        }
    };
    Ok(Some((name, MatVar { dims, data })))
}

/// The variables of a MATLAB 5 file (little-endian, compressed or not), as
/// scipy's loadmat reads them.
pub fn load_mat(path: &Path) -> Result<std::collections::HashMap<String, MatVar>> {
    let b = std::fs::read(path).map_err(|e| crate::Error::from(e).context(path.display()))?;
    if b.len() < 128 || &b[126..128] != b"IM" {
        return Err(crate::Error::invalid(format!("{}: not a little-endian MATLAB 5 file", path.display())));
    }
    let mut vars = std::collections::HashMap::new();
    let mut pos = 128;
    while pos + 8 <= b.len() {
        let (ty, data, next) = element(&b, pos).map_err(|e| e.context(path.display()))?;
        let inflated;
        let m = match ty {
            MI_COMPRESSED => {
                use std::io::Read;
                let mut out = Vec::new();
                flate2::read::ZlibDecoder::new(data)
                    .read_to_end(&mut out)
                    .map_err(|e| crate::Error::invalid(format!("{}: {e}", path.display())))?;
                inflated = out;
                let (ty, inner, _) = element(&inflated, 0)?;
                if ty == MI_MATRIX { Some(inner) } else { None }
            }
            MI_MATRIX => Some(data),
            _ => None,
        };
        if let Some(m) = m {
            if let Some((name, var)) = matrix(m).map_err(|e| e.context(path.display()))? {
                vars.insert(name, var);
            }
        }
        pos = next;
    }
    Ok(vars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("tigerhx-mat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.mat");
        let img = [0.5f32, 1.5, -2.0, 3.25, 4.0, 5.0];
        let seg = [0u8, 1, 2, 3, 2, 1];
        save_mat(&p, &[
            Var::new("input_img", &[1, 2, 3], Data::F32AsDouble(&img)),
            Var::new("Seg", &[1, 2, 3], Data::U8AsInt64(&seg)),
            Var::scalar_i64("ED", 7),
            Var::string("model", "cine4d_v0006_xy_cine.onnx"),
            Var::new("v", &[4], Data::F32(vec![1.5, 1.5, 8.0, 1.0])),
        ])
        .unwrap();
        let m = load_mat(&p).unwrap();
        assert_eq!(m["input_img"].dims, vec![1, 2, 3]);
        assert_eq!(m["input_img"].data, MatData::Num(img.iter().map(|&v| v as f64).collect()));
        assert_eq!(m["Seg"].data, MatData::Num(seg.iter().map(|&v| v as f64).collect()));
        assert_eq!(m["ED"].data, MatData::Num(vec![7.0]));
        assert_eq!(m["model"].data, MatData::Str("cine4d_v0006_xy_cine.onnx".into()));
        assert_eq!(m["v"].dims, vec![1, 4]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
