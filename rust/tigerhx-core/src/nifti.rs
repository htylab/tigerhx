//! NIfTI-1 / NIfTI-2 files (.nii, .nii.gz).
//!
//! Only what TigerHx needs: reading images as f32 (scaled like nibabel's
//! get_fdata), reading label maps, writing label maps that keep the source
//! header (geometry, orientation, units) and building the uncompressed
//! display volumes for the viewer.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use crate::error::{Error, Result};
use crate::invalid;
use crate::volume::Vol;

const DT_UINT8: i16 = 2;
const DT_INT16: i16 = 4;
const DT_INT32: i16 = 8;
const DT_FLOAT32: i16 = 16;
const DT_FLOAT64: i16 = 64;
const DT_INT8: i16 = 256;
const DT_UINT16: i16 = 512;
const DT_UINT32: i16 = 768;
const DT_INT64: i16 = 1024;
const DT_UINT64: i16 = 1280;

/// The raw header of a NIfTI file plus the fields TigerHx reads from it.
#[derive(Clone, Debug)]
pub struct Header {
    /// 348 (NIfTI-1) or 540 (NIfTI-2) header bytes in the file's byte order.
    raw: Vec<u8>,
    big_endian: bool,
    /// Image dimensions, dim[1..=dim[0]].
    pub dims: Vec<usize>,
    /// Voxel sizes matching `dims` (nibabel's header.get_zooms()).
    pub pixdim: Vec<f64>,
    pub datatype: i16,
    vox_offset: usize,
    scl_slope: f64,
    scl_inter: f64,
}

impl Header {
    pub fn version(&self) -> u8 {
        if self.raw.len() == 540 { 2 } else { 1 }
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Zooms padded to 4 values (x, y, slice spacing, frame), as load_nifti.
    pub fn zooms4(&self) -> [f64; 4] {
        let mut z = [1.0; 4];
        for (i, v) in self.pixdim.iter().take(4).enumerate() {
            z[i] = *v;
        }
        z
    }

    fn parse(raw: &[u8]) -> Result<Header> {
        if raw.len() < 348 {
            return invalid!("not a NIfTI file (too short)");
        }
        let le = i32::from_le_bytes(raw[0..4].try_into().unwrap());
        let be = i32::from_be_bytes(raw[0..4].try_into().unwrap());
        let (size, big_endian) = match (le, be) {
            (348, _) => (348, false),
            (_, 348) => (348, true),
            (540, _) => (540, false),
            (_, 540) => (540, true),
            _ => return invalid!("not a NIfTI file (bad header size)"),
        };
        if raw.len() < size {
            return invalid!("not a NIfTI file (truncated header)");
        }
        let r = Reader { raw, big_endian };
        let mut h = Header {
            raw: raw[..size].to_vec(),
            big_endian,
            dims: Vec::new(),
            pixdim: Vec::new(),
            datatype: 0,
            vox_offset: 0,
            scl_slope: 1.0,
            scl_inter: 0.0,
        };
        let (ndim, dim, pix): (usize, Vec<i64>, Vec<f64>) = if size == 348 {
            let ndim = r.i16(40) as i64;
            let dim = (1..=7).map(|i| r.i16(40 + 2 * i) as i64).collect();
            let pix = (1..=7).map(|i| r.f32(76 + 4 * i) as f64).collect();
            h.datatype = r.i16(70);
            h.vox_offset = r.f32(108).max(0.0) as usize;
            h.scl_slope = r.f32(112) as f64;
            h.scl_inter = r.f32(116) as f64;
            (ndim.clamp(0, 7) as usize, dim, pix)
        } else {
            let ndim = r.i64(16);
            let dim = (1..=7).map(|i| r.i64(16 + 8 * i)).collect();
            let pix = (1..=7).map(|i| r.f64(104 + 8 * i)).collect();
            h.datatype = r.i16(12);
            h.vox_offset = r.i64(168).max(0) as usize;
            h.scl_slope = r.f64(176);
            h.scl_inter = r.f64(184);
            (ndim.clamp(0, 7) as usize, dim, pix)
        };
        if ndim == 0 {
            return invalid!("NIfTI header has no dimensions");
        }
        for &d in &dim[..ndim] {
            if d < 1 {
                return invalid!("NIfTI header has an invalid dimension {d}");
            }
        }
        h.dims = dim[..ndim].iter().map(|&d| d as usize).collect();
        h.pixdim = pix[..ndim].iter().map(|v| v.abs()).collect();
        Ok(h)
    }

    /// This header for a uint8 label map of shape `dims` (no scaling).
    fn for_labels(&self, dims: &[usize]) -> Header {
        let mut h = self.clone();
        let w = Writer { big_endian: h.big_endian };
        if h.version() == 1 {
            w.i16(&mut h.raw, 40, dims.len() as i16);
            for i in 1..=7 {
                let d = dims.get(i - 1).copied().unwrap_or(1);
                w.i16(&mut h.raw, 40 + 2 * i, d as i16);
            }
            w.i16(&mut h.raw, 70, DT_UINT8);
            w.i16(&mut h.raw, 72, 8);
            w.f32(&mut h.raw, 108, 352.0);
            w.f32(&mut h.raw, 112, 1.0);
            w.f32(&mut h.raw, 116, 0.0);
            w.f32(&mut h.raw, 124, 0.0); // cal_max
            w.f32(&mut h.raw, 128, 0.0); // cal_min
            h.raw[344..348].copy_from_slice(b"n+1\0");
            h.vox_offset = 352;
        } else {
            w.i64(&mut h.raw, 16, dims.len() as i64);
            for i in 1..=7 {
                let d = dims.get(i - 1).copied().unwrap_or(1);
                w.i64(&mut h.raw, 16 + 8 * i, d as i64);
            }
            w.i16(&mut h.raw, 12, DT_UINT8);
            w.i16(&mut h.raw, 14, 8);
            w.i64(&mut h.raw, 168, 544);
            w.f64(&mut h.raw, 176, 1.0);
            w.f64(&mut h.raw, 184, 0.0);
            w.f64(&mut h.raw, 192, 0.0); // cal_max
            w.f64(&mut h.raw, 200, 0.0); // cal_min
            h.raw[4..12].copy_from_slice(b"n+2\0\r\n\x1a\n");
            h.vox_offset = 544;
        }
        h.dims = dims.to_vec();
        h.pixdim.resize(dims.len(), 1.0);
        h.datatype = DT_UINT8;
        h.scl_slope = 1.0;
        h.scl_inter = 0.0;
        h
    }
}

struct Reader<'a> {
    raw: &'a [u8],
    big_endian: bool,
}

macro_rules! read_fn {
    ($name:ident, $t:ty, $n:expr) => {
        fn $name(&self, off: usize) -> $t {
            let b: [u8; $n] = self.raw[off..off + $n].try_into().unwrap();
            if self.big_endian { <$t>::from_be_bytes(b) } else { <$t>::from_le_bytes(b) }
        }
    };
}

impl Reader<'_> {
    read_fn!(i16, i16, 2);
    read_fn!(i64, i64, 8);
    read_fn!(f32, f32, 4);
    read_fn!(f64, f64, 8);
}

struct Writer {
    big_endian: bool,
}

macro_rules! write_fn {
    ($name:ident, $t:ty) => {
        fn $name(&self, raw: &mut [u8], off: usize, v: $t) {
            let b = if self.big_endian { v.to_be_bytes() } else { v.to_le_bytes() };
            raw[off..off + b.len()].copy_from_slice(&b);
        }
    };
}

impl Writer {
    write_fn!(i16, i16);
    write_fn!(i32, i32);
    write_fn!(i64, i64);
    write_fn!(f32, f32);
    write_fn!(f64, f64);
}

fn is_gz(bytes: &[u8]) -> bool {
    bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b
}

fn read_file(path: &Path, limit: Option<usize>) -> Result<Vec<u8>> {
    let mut f = File::open(path).map_err(|e| Error::from(e).context(path.display()))?;
    let mut magic = [0u8; 2];
    let n = f.read(&mut magic)?;
    let mut out = Vec::new();
    let reader: Box<dyn Read> = if is_gz(&magic[..n]) {
        Box::new(MultiGzDecoder::new(std::io::Cursor::new(magic[..n].to_vec()).chain(f)))
    } else {
        Box::new(std::io::Cursor::new(magic[..n].to_vec()).chain(f))
    };
    match limit {
        Some(l) => reader.take(l as u64).read_to_end(&mut out),
        None => { let mut r = reader; r.read_to_end(&mut out) }
    }
    .map_err(|e| Error::invalid(format!("{}: {e}", path.display())))?;
    Ok(out)
}

/// Only the header (reads at most the first 540 bytes of data).
pub fn read_header(path: &Path) -> Result<Header> {
    let raw = read_file(path, Some(540))?;
    Header::parse(&raw).map_err(|e| e.context(path.display()))
}

fn elem_size(datatype: i16) -> Result<usize> {
    Ok(match datatype {
        DT_UINT8 | DT_INT8 => 1,
        DT_INT16 | DT_UINT16 => 2,
        DT_INT32 | DT_UINT32 | DT_FLOAT32 => 4,
        DT_INT64 | DT_UINT64 | DT_FLOAT64 => 8,
        other => return invalid!("unsupported NIfTI data type {other}"),
    })
}

/// Decode the voxel values as f64 (unscaled) and hand each to `put`.
fn decode(h: &Header, bytes: &[u8], mut put: impl FnMut(usize, f64)) -> Result<()> {
    let es = elem_size(h.datatype)?;
    let n: usize = h.dims.iter().product();
    let start = h.vox_offset;
    if bytes.len() < start + n * es {
        return invalid!("NIfTI data is truncated");
    }
    let data = &bytes[start..start + n * es];
    let be = h.big_endian;
    macro_rules! each {
        ($t:ty) => {
            for (i, c) in data.chunks_exact(std::mem::size_of::<$t>()).enumerate() {
                let b = c.try_into().unwrap();
                let v = if be { <$t>::from_be_bytes(b) } else { <$t>::from_le_bytes(b) };
                put(i, v as f64);
            }
        };
    }
    match h.datatype {
        DT_UINT8 => each!(u8),
        DT_INT8 => each!(i8),
        DT_INT16 => each!(i16),
        DT_UINT16 => each!(u16),
        DT_INT32 => each!(i32),
        DT_UINT32 => each!(u32),
        DT_FLOAT32 => each!(f32),
        DT_INT64 => each!(i64),
        DT_UINT64 => each!(u64),
        DT_FLOAT64 => each!(f64),
        _ => unreachable!(),
    }
    Ok(())
}

fn shape4(h: &Header) -> Result<[usize; 4]> {
    if h.ndim() > 4 {
        return invalid!("Expected a 3D or 4D image, got shape {:?}", h.dims);
    }
    let mut s = [1usize; 4];
    for (i, &d) in h.dims.iter().enumerate() {
        s[i] = d;
    }
    Ok(s)
}

/// An image as f32, always 4D (x, y, slice, frame), scaled with
/// scl_slope/scl_inter like nibabel's get_fdata.
pub fn load(path: &Path) -> Result<(Vol<f32>, Header)> {
    let bytes = read_file(path, None)?;
    let h = Header::parse(&bytes).map_err(|e| e.context(path.display()))?;
    let shape = shape4(&h)?;
    let (slope, inter) = if h.scl_slope.is_finite() && h.scl_slope != 0.0 {
        (h.scl_slope, if h.scl_inter.is_finite() { h.scl_inter } else { 0.0 })
    } else {
        (1.0, 0.0)
    };
    let mut vol = Vol::<f32>::zeros(shape);
    decode(&h, &bytes, |i, v| vol.data[i] = (v * slope + inter) as f32)
        .map_err(|e| e.context(path.display()))?;
    Ok((vol, h))
}

/// A label map as uint8, always 4D (values cast like numpy's astype(uint8)).
pub fn load_labels(path: &Path) -> Result<Vol<u8>> {
    let (vol, _) = load(path)?;
    Ok(vol.map(|v| if v.is_finite() { (v as i64) as u8 } else { 0 }))
}

/// A NIfTI-1 header with the affine diag(zooms), for label maps of an image
/// that has no NIfTI file (an original-GUI .mat result), as nibabel writes
/// Nifti1Image(labels, np.diag(zooms + [1])).
pub fn plain_header(zooms: [f64; 3]) -> Header {
    let mut raw = vec![0u8; 348];
    let w = Writer { big_endian: false };
    w.i32(&mut raw, 0, 348);
    w.i16(&mut raw, 40, 3);
    for i in 1..=7 {
        w.i16(&mut raw, 40 + 2 * i, 1);
    }
    w.i16(&mut raw, 70, DT_UINT8);
    w.i16(&mut raw, 72, 8);
    let pix = [1.0, zooms[0], zooms[1], zooms[2], 1.0, 1.0, 1.0, 1.0];
    for (i, p) in pix.iter().enumerate() {
        w.f32(&mut raw, 76 + 4 * i, *p as f32);
    }
    w.f32(&mut raw, 108, 352.0);
    w.f32(&mut raw, 112, 1.0);
    w.i16(&mut raw, 252, 2); // qform: aligned, no rotation
    w.i16(&mut raw, 254, 2); // sform: the affine
    for r in 0..3 {
        w.f32(&mut raw, 280 + 16 * r + 4 * r, zooms[r] as f32);
    }
    raw[344..348].copy_from_slice(b"n+1\0");
    Header::parse(&raw).expect("a valid header")
}

/// Save a uint8 label map with the header (geometry) of the source image.
/// `ndim` is the number of dimensions to write (3 drops the frame axis,
/// which must then be 1).
pub fn save_labels(labels: &Vol<u8>, ndim: usize, source: &Header, path: &Path) -> Result<()> {
    let dims: Vec<usize> = labels.shape[..ndim.clamp(1, 4)].to_vec();
    if dims.iter().product::<usize>() != labels.data.len() {
        return invalid!("label map shape {:?} cannot be written as {}D", labels.shape, ndim);
    }
    let h = source.for_labels(&dims);
    let mut buf = Vec::with_capacity(h.vox_offset + labels.data.len());
    buf.extend_from_slice(&h.raw);
    buf.resize(h.vox_offset, 0); // no extensions
    buf.extend_from_slice(&labels.data);
    write_maybe_gz(path, &buf)
}

fn write_maybe_gz(path: &Path, bytes: &[u8]) -> Result<()> {
    let gz = path.to_string_lossy().to_lowercase().ends_with(".gz");
    crate::fsutil::write_atomic(path, |f| {
        if gz {
            let mut enc = GzEncoder::new(f, Compression::new(1));
            enc.write_all(bytes)?;
            enc.finish()?;
        } else {
            f.write_all(bytes)?;
        }
        Ok(())
    })
}

/// Voxel data for a display volume.
pub enum DisplayData<'a> {
    F32(&'a [f32]),
    U8(&'a [u8]),
}

/// Uncompressed NIfTI-1 bytes with an axis-aligned affine for the viewer.
///
/// Voxel (i, j, k) maps to screen row i, column j and slice k in an axial
/// view, which matches how the original GUI drew arrays with imshow.  Float
/// images holding only integers within the int16 range are stored as int16.
pub fn display_bytes(data: DisplayData<'_>, shape: [usize; 4], zooms: [f64; 4]) -> Vec<u8> {
    let (dx, dy, dz) = (zooms[0], zooms[1], zooms[2]);
    let mut raw = vec![0u8; 348];
    let w = Writer { big_endian: false };
    w.i32(&mut raw, 0, 348);
    let ndim = if shape[3] > 1 { 4 } else { 3 };
    w.i16(&mut raw, 40, ndim);
    for i in 1..=7 {
        let d = if i <= 4 { shape[i - 1] } else { 1 };
        w.i16(&mut raw, 40 + 2 * i, d as i16);
    }
    let (datatype, bitpix, body): (i16, i16, Vec<u8>) = match data {
        DisplayData::U8(v) => (DT_UINT8, 8, v.to_vec()),
        DisplayData::F32(v) => {
            let ints = v.iter().all(|&x| x.fract() == 0.0 && (-32768.0..=32767.0).contains(&x));
            if ints {
                (DT_INT16, 16, v.iter().flat_map(|&x| (x as i16).to_le_bytes()).collect())
            } else {
                (DT_FLOAT32, 32, v.iter().flat_map(|&x| x.to_le_bytes()).collect())
            }
        }
    };
    w.i16(&mut raw, 70, datatype);
    w.i16(&mut raw, 72, bitpix);
    let pix = [1.0, dx, dy, dz, 1.0, 1.0, 1.0, 1.0];
    for (i, p) in pix.iter().enumerate() {
        w.f32(&mut raw, 76 + 4 * i, *p as f32);
    }
    w.f32(&mut raw, 108, 352.0);
    w.f32(&mut raw, 112, 1.0);
    raw[123] = 2 | 8; // mm, sec
    // qform: rotation of -90 degrees about z; sform: the full affine
    w.i16(&mut raw, 252, 0);
    w.i16(&mut raw, 254, 2);
    w.f32(&mut raw, 264, -std::f32::consts::FRAC_1_SQRT_2);
    let srow = [[0.0, dy, 0.0, 0.0], [-dx, 0.0, 0.0, 0.0], [0.0, 0.0, dz, 0.0]];
    for (r, row) in srow.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            w.f32(&mut raw, 280 + 16 * r + 4 * c, *v as f32);
        }
    }
    raw[344..348].copy_from_slice(b"n+1\0");
    let mut out = raw;
    out.resize(352, 0);
    out.extend_from_slice(&body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_roundtrip() {
        let vol = Vol::from_vec((0..24).map(|v| v as f32).collect(), [2, 3, 4, 1]);
        let bytes = display_bytes(DisplayData::F32(&vol.data), vol.shape, [1.5, 2.0, 8.0, 1.0]);
        let h = Header::parse(&bytes).unwrap();
        assert_eq!(h.dims, vec![2, 3, 4]);
        assert_eq!(h.datatype, DT_INT16);
        assert_eq!(h.zooms4(), [1.5, 2.0, 8.0, 1.0]);
        let mut back = vec![0.0; 24];
        decode(&h, &bytes, |i, v| back[i] = v as f32).unwrap();
        assert_eq!(back, vol.data);
    }

    #[test]
    fn labels_keep_source_header() {
        let dir = std::env::temp_dir().join(format!("tigerhx-nifti-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = display_bytes(DisplayData::F32(&[0.5; 12]), [2, 3, 2, 1], [1.25, 1.25, 10.0, 1.0]);
        let h = Header::parse(&src).unwrap();
        let labels = Vol::from_vec((0..12).map(|v| v as u8).collect(), [2, 3, 2, 1]);
        let p = dir.join("seg.nii.gz");
        save_labels(&labels, 3, &h, &p).unwrap();
        let back = load_labels(&p).unwrap();
        assert_eq!(back, labels);
        let h2 = read_header(&p).unwrap();
        assert_eq!(h2.dims, vec![2, 3, 2]);
        assert_eq!(h2.zooms4(), [1.25, 1.25, 10.0, 1.0]);
        assert_eq!(h2.raw[280..344], h.raw[280..344]); // sform kept
        std::fs::remove_dir_all(&dir).ok();
    }
}
